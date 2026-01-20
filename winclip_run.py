#!/usr/bin/env python3
"""
WinCLIP++: Explainable Anomaly Detection (foreground-aware) + MVTec benchmark
--------------------------------------------------------------------------------
What this script fixes vs the previous version:
- Background false positives: strong CLIP-based foreground (objectness) mask
- Clearer regions: component-based boxes on the *object* + better visualization
- Explainability: ALWAYS returns an explanation; prefers LLaVA via vLLM OpenAI server
  (http://localhost:8000/v1/chat/completions), with safe fallbacks.

Requirements (typical):
  pip install torch open-clip-torch opencv-python pillow tqdm scikit-learn matplotlib requests

Usage:
  # Single image + explanation
  python winclip_run.py --image datasets/screw/test/scratch_head/002.png --class screw

  # Full MVTec benchmark (expects ../mvtec_anomaly_detection)
  python winclip_run.py --benchmark --mvtec_root ../mvtec_anomaly_detection
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
import torch
import torch.nn.functional as F
import warnings
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

import open_clip

warnings.filterwarnings("ignore")


# ------------------------------- utilities ---------------------------------

def _b64_png(arr_rgb: np.ndarray) -> str:
    img = Image.fromarray(arr_rgb)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _robust_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust z-score using MAD computed on mask region."""
    v = x[mask]
    if v.size < 16:
        return x
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad < 1e-6:
        return x - med
    return (x - med) / (mad * 1.4826)


def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    # skip background (0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = (labels == idx).astype(np.uint8)
    return out


def _grid_size_from_vit(patches: torch.Tensor) -> int:
    # patches: (B, N, D) where N = grid^2
    n = patches.shape[1]
    g = int(round(np.sqrt(n)))
    return g


def _resize_to(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _bbox_pad(x1, y1, x2, y2, pad, W, H):
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(W - 1, x2 + pad),
        min(H - 1, y2 + pad),
    )


def _pos_desc(cx: float, cy: float) -> str:
    # cx,cy are in [0,1]
    v = "top" if cy < 0.33 else ("middle" if cy < 0.67 else "bottom")
    h = "left" if cx < 0.33 else ("center" if cx < 0.67 else "right")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


# ------------------------------- data types --------------------------------

@dataclass
class Region:
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2 (inclusive)
    score: float


@dataclass
class AnomalyResult:
    anomaly_map: np.ndarray          # (H,W) float32
    image_score: float
    foreground_mask: np.ndarray      # (H,W) bool
    regions: List[Region]


# ----------------------------- WinCLIP++ core ------------------------------

class WinCLIP:
    """
    CLIP patch scoring + normal memory (nearest neighbor) + foreground gating.
    Multi-layer token fusion for sharper maps.
    """
    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B-16", clip_pretrained: str = "openai",
                 layers: Optional[List[int]] = None, img_size: int = 224):
        self.device = device
        self.img_size = img_size

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Multi-layer token capture
        n_blocks = len(self.model.visual.transformer.resblocks)
        self.layers = layers or [n_blocks - 3, n_blocks - 2, n_blocks - 1]
        self._feat: Dict[str, torch.Tensor] = {}
        for li in self.layers:
            self.model.visual.transformer.resblocks[li].register_forward_hook(self._hook(f"l{li}"))

        # projection from visual blocks to text embedding space
        self.visual_proj = self.model.visual.proj

        # figure embed dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size, device=device)
            emb = self.model.encode_image(dummy)
        self.embed_dim = emb.shape[-1]

        self.normal_memory: Optional[torch.Tensor] = None
        self._cached_bg_anchor: Optional[torch.Tensor] = None

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            self._feat[name] = out
        return fn

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts).to(self.device)
        t = self.model.encode_text(tok)
        return F.normalize(t, dim=-1)

    def build_text_anchors(self, class_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # robust templates to reduce prompt sensitivity
        normal = [
            f"a photo of a flawless {class_name}",
            f"a photo of a perfect {class_name}",
            f"a photo of a {class_name} without defects",
            f"a close-up photo of a clean {class_name}",
        ]
        anomalous = [
            f"a photo of a {class_name} with defects",
            f"a photo of a damaged {class_name}",
            f"a photo of a broken {class_name}",
            f"a close-up photo of a defective {class_name}",
        ]
        n = self.encode_text(normal).mean(0)
        a = self.encode_text(anomalous).mean(0)
        return F.normalize(n, dim=-1), F.normalize(a, dim=-1)

    def _bg_anchor(self) -> torch.Tensor:
        if self._cached_bg_anchor is not None:
            return self._cached_bg_anchor
        bg = [
            "a photo of the background",
            "a photo of an empty surface",
            "a photo of a plain tabletop background",
            "a photo with no object present",
        ]
        self._cached_bg_anchor = F.normalize(self.encode_text(bg).mean(0), dim=-1)
        return self._cached_bg_anchor

    @torch.no_grad()
    def _encode_image_tokens(self, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict layer_name -> patch tokens in text-embedding space: (B, N, D)
        """
        self._feat.clear()
        _ = self.model.encode_image(img_t)

        out: Dict[str, torch.Tensor] = {}
        for k, feat in self._feat.items():
            # OpenCLIP: (seq, batch, dim) OR (batch, seq, dim) depending on version
            if feat.dim() == 3 and feat.shape[0] != img_t.shape[0] and feat.shape[1] == img_t.shape[0]:
                feat = feat.permute(1, 0, 2)  # (B, seq, dim)
            elif feat.dim() == 3 and feat.shape[0] == img_t.shape[0]:
                # already (B, seq, dim)
                pass
            else:
                continue

            patches = feat[:, 1:, :]  # remove CLS -> (B, N, dim)
            if self.visual_proj is not None:
                B, N, D = patches.shape
                patches = (patches.reshape(B * N, D) @ self.visual_proj).reshape(B, N, self.embed_dim)
            patches = F.normalize(patches, dim=-1)
            out[k] = patches
        return out

    def _foreground_mask_from_tokens(self, tokens_by_layer: Dict[str, torch.Tensor],
                                     class_name: str,
                                     img_rgb_224: np.ndarray,
                                     obj_thresh: Optional[float] = None) -> np.ndarray:
        """
        Strong foreground gating:
        - CLIP objectness per patch = sim(object_prompt) - sim(background_prompt)
        - adaptive threshold (Otsu on objectness map) + morphology + keep largest component
        - optional sanity fusion with simple intensity-based mask
        Returns mask at 224x224 bool.
        """
        # fuse tokens across layers for objectness
        toks = torch.stack([tokens_by_layer[k] for k in sorted(tokens_by_layer.keys())], 0).mean(0).squeeze(0)  # (N,D)
        grid = _grid_size_from_vit(toks.unsqueeze(0))
        if grid * grid != toks.shape[0]:
            grid = int(round(np.sqrt(toks.shape[0])))

        obj_prompts = [
            f"a photo of a {class_name}",
            f"the {class_name} object",
            f"a close-up photo of a {class_name}",
        ]
        obj_anchor = F.normalize(self.encode_text(obj_prompts).mean(0), dim=-1)
        bg_anchor = self._bg_anchor()

        obj_sim = (toks @ obj_anchor).float().cpu().numpy()
        bg_sim = (toks @ bg_anchor).float().cpu().numpy()
        objness = (obj_sim - bg_sim).reshape(grid, grid)

        # normalize to 0..255 for thresholding
        o = objness.copy()
        o = (o - o.min()) / (o.max() - o.min() + 1e-6)
        o_u8 = (o * 255).astype(np.uint8)

        if obj_thresh is None:
            # Otsu threshold on patch grid, fallback to percentile if degenerate
            try:
                _, th = cv2.threshold(o_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr_val = th / 255.0
            except Exception:
                thr_val = float(np.percentile(o, 60))
        else:
            thr_val = obj_thresh

        mask_grid = (o > thr_val).astype(np.uint8)
        # smooth on grid then upscale
        k = 3 if grid <= 14 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_grid = cv2.morphologyEx(mask_grid, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_grid = _keep_largest_component(mask_grid)

        mask_224 = cv2.resize(mask_grid, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # sanity fusion: simple "non-background" mask
        gray = cv2.cvtColor(img_rgb_224, cv2.COLOR_RGB2GRAY)
        grayf = gray.astype(np.float32) / 255.0
        simple = ((grayf > 0.03) & (grayf < 0.97)).astype(np.uint8)
        simple = cv2.morphologyEx(simple, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        simple = cv2.morphologyEx(simple, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        simple = _keep_largest_component(simple)

        # combine: prefer intersection, but if too small, fall back to union
        inter = (mask_224 & simple).astype(np.uint8)
        if inter.sum() < 0.25 * max(1, mask_224.sum()):
            fused = (mask_224 | simple).astype(np.uint8)
        else:
            fused = inter

        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        fused = _keep_largest_component(fused)
        return fused.astype(bool)

    def build_normal_memory(self, train_good: List[Path], class_name: str,
                            max_images: int = 50, coreset_ratio: float = 0.08,
                            min_coreset: int = 256, max_coreset: int = 4000) -> None:
        """
        Build compact memory of normal patch tokens (foreground only).
        """
        good = train_good[:max_images]
        all_patches: List[torch.Tensor] = []
        for p in tqdm(good, desc="Building memory"):
            img_pil = Image.open(p).convert("RGB")
            img_t = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            img_224 = np.array(img_pil.resize((self.img_size, self.img_size), Image.BILINEAR))
            toks_by_layer = self._encode_image_tokens(img_t)
            fg = self._foreground_mask_from_tokens(toks_by_layer, class_name, img_224)

            # select from fused tokens
            fused = torch.stack([toks_by_layer[k] for k in sorted(toks_by_layer.keys())], 0).mean(0).squeeze(0)  # (N,D)
            grid = int(round(np.sqrt(fused.shape[0])))
            fg_g = cv2.resize(fg.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_NEAREST).flatten().astype(bool)

            if fg_g.sum() > 0:
                all_patches.append(fused[torch.from_numpy(np.where(fg_g)[0]).to(self.device)].detach().cpu())

        if not all_patches:
            self.normal_memory = None
            print("WARNING: no foreground patches collected for memory; NN term disabled.")
            return

        all_p = torch.cat(all_patches, 0)  # (M,D)
        n = all_p.shape[0]
        k = int(max(min_coreset, min(max_coreset, n * coreset_ratio)))
        idx = torch.randperm(n)[:k]
        self.normal_memory = all_p[idx].to(self.device)
        print(f"Memory: {self.normal_memory.shape[0]} patches")

    @torch.no_grad()
    def detect(self, image_path: Path, class_name: str,
               normal_anchor: torch.Tensor, anomalous_anchor: torch.Tensor,
               w_text: float = 0.35, w_nn: float = 0.65,
               max_regions: int = 5) -> AnomalyResult:

        img_pil = Image.open(image_path).convert("RGB")
        img_rgb = np.array(img_pil)
        H, W = img_rgb.shape[:2]

        # 224 view for CLIP
        img_224 = np.array(img_pil.resize((self.img_size, self.img_size), Image.BILINEAR))
        img_t = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        toks_by_layer = self._encode_image_tokens(img_t)

        # foreground gating at 224
        fg_224 = self._foreground_mask_from_tokens(toks_by_layer, class_name, img_224)

        # per-layer anomaly maps at 224
        per_maps: List[np.ndarray] = []
        per_conf: List[float] = []

        for k in sorted(toks_by_layer.keys()):
            toks = toks_by_layer[k].squeeze(0)  # (N,D)
            grid = int(round(np.sqrt(toks.shape[0])))

            # text map
            sim_n = (toks @ normal_anchor).float().cpu().numpy()
            sim_a = (toks @ anomalous_anchor).float().cpu().numpy()
            margin = np.abs(sim_a - sim_n)
            text = (sim_a - sim_n).reshape(grid, grid).astype(np.float32)
            text = cv2.resize(text, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

            # NN map
            if self.normal_memory is not None:
                d = torch.cdist(toks, self.normal_memory, p=2)
                nn = d.min(1)[0].float().cpu().numpy().reshape(grid, grid).astype(np.float32)
                nn = cv2.resize(nn, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            else:
                nn = np.zeros((self.img_size, self.img_size), np.float32)

            m = w_text * text + w_nn * nn
            m *= fg_224.astype(np.float32)
            m = _robust_z(m, fg_224)
            m = np.clip(m, -3, 10) * fg_224.astype(np.float32)

            # confidence weight: higher when margin is strong
            conf = float(np.mean(margin))
            per_maps.append(m)
            per_conf.append(conf)

        # fuse layers with softmax weights on confidence
        conf = np.array(per_conf, dtype=np.float32)
        if np.all(conf == 0):
            w = np.ones_like(conf) / len(conf)
        else:
            w = np.exp(conf - conf.max())
            w = w / (w.sum() + 1e-6)

        fused_224 = np.zeros((self.img_size, self.img_size), np.float32)
        for wi, mi in zip(w, per_maps):
            fused_224 += float(wi) * mi

        # light smoothing (foreground-preserving)
        fused_224 = cv2.GaussianBlur(fused_224, (0, 0), sigmaX=2.0)
        fused_224 *= fg_224.astype(np.float32)

        # normalize to [0,1] on foreground for nicer viz + stable thresholding
        fg_vals = fused_224[fg_224]
        if fg_vals.size > 0:
            lo = np.percentile(fg_vals, 5)
            hi = np.percentile(fg_vals, 99.5)
            fused_224 = (fused_224 - lo) / (hi - lo + 1e-6)
        fused_224 = np.clip(fused_224, 0, 1) * fg_224.astype(np.float32)

        # upscale map/mask to original resolution
        anomaly_map = cv2.resize(fused_224, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        fg_mask = cv2.resize(fg_224.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        anomaly_map *= fg_mask.astype(np.float32)

        # image score: max on foreground
        image_score = float(np.max(anomaly_map[fg_mask])) if fg_mask.any() else 0.0

        # regions: connected components of top-percentile anomaly within foreground
        regions = self._regions_from_map(anomaly_map, fg_mask, max_regions=max_regions)

        return AnomalyResult(anomaly_map=anomaly_map, image_score=image_score, foreground_mask=fg_mask, regions=regions)

    @staticmethod
    def _regions_from_map(anom: np.ndarray, fg: np.ndarray, max_regions: int = 5) -> List[Region]:
        if not fg.any():
            return []
        vals = anom[fg]
        if vals.size < 32:
            return []
        thr = np.percentile(vals, 98.5)  # focus on sharp hotspots
        binm = ((anom >= thr) & fg).astype(np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        regs: List[Region] = []
        H, W = anom.shape[:2]
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 20:
                continue
            x1, y1 = x, y
            x2, y2 = x + w - 1, y + h - 1
            # score as max inside component
            comp = (labels == i)
            sc = float(np.max(anom[comp])) if comp.any() else 0.0
            # pad for better crops
            x1, y1, x2, y2 = _bbox_pad(x1, y1, x2, y2, pad=12, W=W, H=H)
            regs.append(Region((x1, y1, x2, y2), sc))

        regs.sort(key=lambda r: r.score, reverse=True)
        return regs[:max_regions]


# --------------------------- LLaVA explainability ---------------------------

class LLaVAExplainer:
    """
    Prefer OpenAI-compatible vLLM server (fast, stable).
    If unavailable, falls back to deterministic explanation (never empty).
    """
    def __init__(self,
                 url: str = "http://localhost:8000/v1/chat/completions",
                 model: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                 timeout: int = 180):
        self.url = url
        self.model = model
        self.timeout = timeout

    def explain(self, img_rgb: np.ndarray, result: AnomalyResult, class_name: str) -> str:
        # Always validate regions: must intersect foreground
        regs = []
        for r in result.regions:
            x1, y1, x2, y2 = r.bbox
            if result.foreground_mask[y1:y2 + 1, x1:x2 + 1].any():
                regs.append(r)
        if not regs:
            return self._fallback(img_rgb.shape[:2], class_name, result, reason="No confident object regions; likely background noise.")

        # Build crops: top K regions + full image
        crops = []
        for i, r in enumerate(regs[:3], 1):
            x1, y1, x2, y2 = r.bbox
            crop = img_rgb[y1:y2 + 1, x1:x2 + 1]
            crops.append((f"Region {i}", crop, r))

        # Ask LLaVA per crop to keep grounded, then synthesize
        per_text = []
        for label, crop, r in crops:
            try:
                t = self._call_llava(crop, class_name, label, r, img_rgb.shape[:2])
                if t:
                    per_text.append(t.strip())
            except Exception:
                # continue, we'll fallback later if needed
                pass

        if per_text:
            # concise aggregation
            summary = self._aggregate(per_text, regs, img_rgb.shape[:2], class_name, result.image_score)
            return summary

        return self._fallback(img_rgb.shape[:2], class_name, result, reason="LLaVA server not reachable or returned empty.")

    def _call_llava(self, crop_rgb: np.ndarray, class_name: str, label: str, r: Region, hw: Tuple[int, int]) -> str:
        H, W = hw
        x1, y1, x2, y2 = r.bbox
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        loc = _pos_desc(cx, cy)

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 160,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"You are inspecting an industrial {class_name} for defects.\n"
                                f"{label} is a crop around a suspected anomaly at the {loc} of the object.\n"
                                f"Describe ONLY what you can see: defect type, texture, shape, and exact location.\n"
                                f"If no defect is visible in this crop, reply exactly: 'no defect visible'.\n"
                                f"Be concise (1-2 sentences)."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _b64_png(crop_rgb)},
                        },
                    ],
                }
            ],
        }

        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        return text

    @staticmethod
    def _aggregate(per_region: List[str], regs: List[Region], hw: Tuple[int, int], class_name: str, score: float) -> str:
        H, W = hw
        lines = []
        for i, (t, r) in enumerate(zip(per_region, regs[: len(per_region)]), 1):
            x1, y1, x2, y2 = r.bbox
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            loc = _pos_desc(cx, cy)
            lines.append(f"- Region {i} ({loc}, score={r.score:.2f}): {t}")

        verdict = "Defect detected" if score > 0.35 else "No strong defect evidence"
        # Keep it short and operator-friendly
        return (
            f"{verdict} on {class_name}. Image anomaly score={score:.3f}.\n"
            "Findings:\n" + "\n".join(lines)
        )

    @staticmethod
    def _fallback(hw: Tuple[int, int], class_name: str, result: AnomalyResult, reason: str) -> str:
        H, W = hw
        if not result.regions:
            return (
                f"Explanation fallback: {reason}\n"
                f"Summary: no clear defect visible on the {class_name}. Image score={result.image_score:.3f}."
            )

        lines = [f"Explanation fallback: {reason}"]
        for i, r in enumerate(result.regions[:3], 1):
            x1, y1, x2, y2 = r.bbox
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            loc = _pos_desc(cx, cy)
            lines.append(
                f"- Region {i} ({loc}, score={r.score:.2f}): localized anomaly hotspot inside object crop; "
                f"inspect for scratch/dent/discoloration."
            )
        lines.append(f"Image score={result.image_score:.3f}.")
        return "\n".join(lines)


# ---------------------------- Visualization/IO ------------------------------

def save_visuals(img_rgb: np.ndarray, result: AnomalyResult, out_dir: Path, stem: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = img_rgb.shape[:2]

    # heat overlay
    heat = (result.anomaly_map * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb[..., ::-1], 0.55, heat_color, 0.45, 0)  # to BGR for OpenCV
    overlay = overlay[..., ::-1]  # back to RGB

    heat_path = out_dir / f"{stem}_heat.png"
    Image.fromarray(overlay).save(heat_path)

    # boxes image
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_rgb)
    ax.imshow(result.anomaly_map, cmap="jet", alpha=0.35)
    for r in result.regions:
        x1, y1, x2, y2 = r.bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 6), f"{r.score:.2f}", color="red", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.axis("off")
    ax.set_title(f"Regions (score={result.image_score:.3f})")
    boxes_path = out_dir / f"{stem}_boxes.png"
    plt.tight_layout()
    plt.savefig(boxes_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"heat": str(heat_path), "boxes": str(boxes_path)}


def save_report(out_dir: Path, stem: str, image_path: Path, class_name: str, result: AnomalyResult, explanation: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("ANOMALY DETECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"Image: {image_path.name}")
    lines.append(f"Class: {class_name}")
    lines.append(f"Anomaly Score: {result.image_score:.4f}")
    lines.append("")
    lines.append(f"DETECTED REGIONS ({len(result.regions)}):")
    for i, r in enumerate(result.regions, 1):
        x1, y1, x2, y2 = r.bbox
        lines.append(f"  Region {i}: bbox=({x1},{y1})-({x2},{y2}), score={r.score:.3f}")
    lines.append("")
    lines.append("EXPLANATION:")
    lines.append(explanation.strip() if explanation and explanation.strip() else "Explanation fallback: no text generated.")
    rep = "\n".join(lines) + "\n"
    rp = out_dir / f"{stem}_report.txt"
    rp.write_text(rep, encoding="utf-8")
    return str(rp)


# ------------------------------ MVTec Benchmark -----------------------------

class MVTecBenchmark:
    def __init__(self, mvtec_root: Path, device: str = "cuda"):
        self.root = Path(mvtec_root)
        self.win = WinCLIP(device=device)

    def _class_data(self, cls: str) -> Dict[str, List[Path]]:
        d = self.root / cls
        train_good = sorted((d / "train" / "good").glob("*.png"))
        test_good = sorted((d / "test" / "good").glob("*.png"))
        test_def = []
        for sub in (d / "test").iterdir():
            if sub.is_dir() and sub.name != "good":
                test_def += sorted(sub.glob("*.png"))
        return {"train": train_good, "good": test_good, "def": test_def}

    def _gt_mask(self, img_path: Path) -> Optional[np.ndarray]:
        parts = img_path.parts
        if "test" not in parts:
            return None
        ti = parts.index("test")
        defect = parts[ti + 1]
        img = Image.open(img_path)
        if defect == "good":
            return np.zeros((img.height, img.width), dtype=bool)
        gt = Path(*parts[:ti]) / "ground_truth" / defect / f"{img_path.stem}_mask.png"
        if not gt.exists():
            return None
        m = np.array(Image.open(gt).convert("L"))
        return (m > 127).astype(bool)

    def evaluate_class(self, cls: str, max_train: int = 50) -> Dict[str, float]:
        print(f"\n{'='*60}\nEvaluating: {cls}\n{'='*60}")
        data = self._class_data(cls)
        if not data["train"]:
            print("No training images; skipping.")
            return {}

        nA, aA = self.win.build_text_anchors(cls)
        self.win.build_normal_memory(data["train"], cls, max_images=max_train)

        img_scores: List[float] = []
        img_labels: List[int] = []
        pix_scores: List[np.ndarray] = []
        pix_labels: List[np.ndarray] = []

        for p in tqdm(data["good"], desc="Good"):
            r = self.win.detect(p, cls, nA, aA)
            img_scores.append(r.image_score)
            img_labels.append(0)
            gt = self._gt_mask(p)
            if gt is not None:
                pix_scores.append(r.anomaly_map.flatten())
                pix_labels.append(gt.flatten())

        for p in tqdm(data["def"], desc="Defect"):
            r = self.win.detect(p, cls, nA, aA)
            img_scores.append(r.image_score)
            img_labels.append(1)
            gt = self._gt_mask(p)
            if gt is not None:
                pix_scores.append(r.anomaly_map.flatten())
                pix_labels.append(gt.flatten())

        out: Dict[str, float] = {}
        if len(set(img_labels)) > 1:
            out["img_auroc"] = roc_auc_score(img_labels, img_scores) * 100.0
        else:
            out["img_auroc"] = 0.0

        if pix_scores:
            ps = np.concatenate(pix_scores, 0)
            pl = np.concatenate(pix_labels, 0)
            if len(set(pl)) > 1:
                out["pix_auroc"] = roc_auc_score(pl, ps) * 100.0
                out["pix_ap"] = average_precision_score(pl, ps) * 100.0
            else:
                out["pix_auroc"] = out["pix_ap"] = 0.0

        print(f"Results: Img-AUROC={out.get('img_auroc',0):.1f}% "
              f"Pix-AUROC={out.get('pix_auroc',0):.1f}% Pix-AP={out.get('pix_ap',0):.1f}%")
        return out

    def run(self, classes: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        default = [
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
            "transistor", "wood", "zipper",
        ]
        classes = classes or default

        results: Dict[str, Dict[str, float]] = {}
        for c in classes:
            if not (self.root / c).exists():
                continue
            try:
                results[c] = self.evaluate_class(c)
            except Exception as e:
                print(f"Error on {c}: {e}")
                results[c] = {}

        print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
        print(f"{'Class':<15} {'Img-AUROC':<12} {'Pix-AUROC':<12} {'Pix-AP':<12}")
        print("-" * 70)
        img, pA, pP = [], [], []
        for c, m in results.items():
            ia = m.get("img_auroc", 0.0)
            pa = m.get("pix_auroc", 0.0)
            pp = m.get("pix_ap", 0.0)
            print(f"{c:<15} {ia:<12.1f} {pa:<12.1f} {pp:<12.1f}")
            if ia > 0: img.append(ia)
            if pa > 0: pA.append(pa)
            if pp > 0: pP.append(pp)
        print("-" * 70)
        print(f"{'MEAN':<15} {np.mean(img) if img else 0:<12.1f} "
              f"{np.mean(pA) if pA else 0:<12.1f} {np.mean(pP) if pP else 0:<12.1f}")
        print("=" * 70)
        return results


# ---------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="single image path to run")
    ap.add_argument("--class", dest="class_name", type=str, default=None, help="mvtec class name (e.g., screw)")
    ap.add_argument("--out", type=str, default="outputs", help="output directory")
    ap.add_argument("--llava_url", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI chat endpoint")
    ap.add_argument("--llava_model", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", help="LLaVA model id")
    ap.add_argument("--no_llava", action="store_true", help="disable llava calls")
    ap.add_argument("--benchmark", action="store_true", help="run full MVTec benchmark")
    ap.add_argument("--mvtec_root", type=str, default="../mvtec_anomaly_detection", help="MVTec root folder")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    out_dir = Path(args.out)

    if args.benchmark:
        mvtec_root = Path(args.mvtec_root)
        if not mvtec_root.exists():
            raise SystemExit(f"MVTec not found at: {mvtec_root.resolve()}")
        bench = MVTecBenchmark(mvtec_root, device=device)
        bench.run()
        return

    if not args.image or not args.class_name:
        raise SystemExit("Provide --image <path> and --class <mvtec_class> OR use --benchmark")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    cls = args.class_name
    win = WinCLIP(device=device)
    nA, aA = win.build_text_anchors(cls)

    # OPTIONAL: If you're running within MVTec, try to build memory from train/good automatically.
    # This drastically reduces false positives on object texture.
    try:
        mvtec_root = Path(args.mvtec_root)
        train_good = sorted((mvtec_root / cls / "train" / "good").glob("*.png"))
        if train_good:
            win.build_normal_memory(train_good, cls, max_images=50)
    except Exception:
        pass

    res = win.detect(image_path, cls, nA, aA)

    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    expl = "LLaVA disabled."
    if not args.no_llava:
        explainer = LLaVAExplainer(url=args.llava_url, model=args.llava_model)
        expl = explainer.explain(img_rgb, res, cls)

    # ensure never empty
    if not expl or len(expl.strip()) < 8:
        expl = LLaVAExplainer._fallback(img_rgb.shape[:2], cls, res, reason="Empty explanation string.")

    stem = image_path.stem
    vis = save_visuals(img_rgb, res, out_dir, stem)
    rep = save_report(out_dir, stem, image_path, cls, res, expl)

    print("\n" + "=" * 60)
    print(f"Image: {image_path}")
    print(f"Class: {cls}")
    print(f"Score: {res.image_score:.4f}")
    print(f"Regions: {len(res.regions)}")
    print(f"Saved: {vis['heat']}\n       {vis['boxes']}\n       {rep}")
    print("=" * 60)
    print("\nEXPLANATION:\n" + expl)


if __name__ == "__main__":
    main()
