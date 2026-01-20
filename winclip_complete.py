#!/usr/bin/env python3
"""
WinCLIP: Minimal Anomaly Detection with Explainability
- Foreground-only scoring + normal memory + multi-layer fusion
- Full MVTec benchmark
- LLaVA-based textual explanations
"""

import torch
import torch.nn.functional as F
import open_clip
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
warnings.filterwarnings('ignore')


@dataclass
class AnomalyResult:
    anomaly_map: np.ndarray
    image_score: float
    foreground_mask: np.ndarray
    hotspots: List[Tuple[int, int, float]] = None  # (x, y, score)


class WinCLIP:
    def __init__(self, device="cuda", layers=None):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        
        # Multi-layer extraction
        num_layers = len(self.model.visual.transformer.resblocks)
        self.layers = layers or [num_layers-3, num_layers-2, num_layers-1]
        self.features = {}
        for i in self.layers:
            self.model.visual.transformer.resblocks[i].register_forward_hook(
                self._get_hook(f'layer_{i}'))
        
        # Get embedding dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = self.model.encode_image(dummy)
            self.embed_dim = out.shape[-1]
        
        self.visual_proj = self.model.visual.proj
        self.normal_memory = None
        self.img_size = 224
        self.grid_size = 14
    
    def _get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    @torch.no_grad()
    def encode_text(self, prompts: List[str]):
        tokens = self.tokenizer(prompts).to(self.device)
        emb = self.model.encode_text(tokens)
        return F.normalize(emb, dim=-1)
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        self.features.clear()
        _ = self.model.encode_image(image)
        
        patch_features = {}
        for name, feat in self.features.items():
            # OpenCLIP: (seq, batch, dim) -> (batch, seq, dim)
            if feat.dim() == 3 and feat.shape[1] == image.shape[0]:
                feat = feat.permute(1, 0, 2)
            
            patches = feat[:, 1:, :]  # Remove CLS
            if self.visual_proj is not None:
                B, N, D = patches.shape
                patches = patches.reshape(B*N, D) @ self.visual_proj
                patches = patches.reshape(B, N, self.embed_dim)
            patches = F.normalize(patches, dim=-1)
            patch_features[name] = patches
        return patch_features
    
    def build_text_anchors(self, class_name: str):
        normal = [f"a photo of a flawless {class_name}",
                  f"a photo of a perfect {class_name}",
                  f"a photo of a {class_name} without defects"]
        anomalous = [f"a photo of a {class_name} with defects",
                     f"a photo of a damaged {class_name}",
                     f"a photo of a broken {class_name}"]
        normal_anchor = self.encode_text(normal).mean(dim=0)
        anomalous_anchor = self.encode_text(anomalous).mean(dim=0)
        return F.normalize(normal_anchor, dim=-1), F.normalize(anomalous_anchor, dim=-1)
    
    def compute_foreground_mask(self, image: np.ndarray, threshold=0.05):
        """Fallback OpenCV-based foreground mask."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask = (gray > threshold) & (gray < (1.0 - threshold))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 1, thickness=-1)
        return mask.astype(bool)
    
    def compute_clip_objectness_mask(self, patch_features_dict, global_embedding, threshold=0.15):
        """CLIP-native objectness: patches similar to global image are object patches."""
        # Use fused multi-layer features
        fused = torch.stack([patch_features_dict[k] for k in sorted(patch_features_dict.keys())], 
                           dim=0).mean(dim=0).squeeze(0)  # (N, D)
        
        # Compute patch-to-global similarity
        patch_to_global_sim = (fused @ global_embedding).cpu().numpy()  # (N,)
        
        # Patches with high similarity to global image are object patches
        # Use a soft threshold to be more permissive
        object_mask_flat = patch_to_global_sim > threshold
        object_mask = object_mask_flat.reshape(self.grid_size, self.grid_size)
        
        # Resize to image size
        object_mask_resized = cv2.resize(object_mask.astype(np.uint8), 
                                        (self.img_size, self.img_size),
                                        interpolation=cv2.INTER_NEAREST)
        # Light morphology for smoothness (smaller kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask_resized = cv2.morphologyEx(object_mask_resized, cv2.MORPH_CLOSE, kernel)
        
        return object_mask_resized.astype(bool), object_mask_flat
    
    def build_normal_memory(self, image_paths, normal_anchor, anomalous_anchor, 
                           coreset_ratio=0.1, max_images=50):
        all_patches = []
        for img_path in tqdm(image_paths[:max_images], desc="Building memory"):
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            img_np = np.array(img_pil)
            fg_mask = self.compute_foreground_mask(img_np)
            fg_mask_resized = cv2.resize(fg_mask.astype(np.uint8), 
                                        (self.grid_size, self.grid_size),
                                        interpolation=cv2.INTER_NEAREST).flatten()
            patch_features = self.encode_image(img_tensor)
            fused = torch.stack([patch_features[k] for k in sorted(patch_features.keys())], 
                               dim=0).mean(dim=0).squeeze(0)
            fg_indices = np.where(fg_mask_resized)[0]
            if len(fg_indices) > 0:
                all_patches.append(fused[fg_indices].cpu())
        
        all_patches = torch.cat(all_patches, dim=0)
        n_coreset = max(int(len(all_patches) * coreset_ratio), 100)
        n_coreset = min(n_coreset, len(all_patches))
        indices = torch.randperm(len(all_patches))[:n_coreset]
        self.normal_memory = all_patches[indices].to(self.device)
        print(f"Memory: {len(self.normal_memory)} patches")
    
    def detect(self, image_path: Path, normal_anchor, anomalous_anchor, 
              text_weight=0.3, nn_weight=0.7):
        # Load and prepare
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        
        # Compute foreground mask (proven approach)
        fg_mask = self.compute_foreground_mask(img_np)
        fg_mask_resized = cv2.resize(fg_mask.astype(np.uint8), (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Extract features
        patch_features = self.encode_image(img_tensor)
        fused = torch.stack([patch_features[k] for k in sorted(patch_features.keys())], 
                           dim=0).mean(dim=0).squeeze(0)  # (N, D)
        
        # Text similarities (on all patches)
        sim_normal = (fused @ normal_anchor).cpu().numpy()
        sim_anomalous = (fused @ anomalous_anchor).cpu().numpy()
        text_map = (sim_anomalous - sim_normal).reshape(self.grid_size, self.grid_size)
        text_map = cv2.resize(text_map, (self.img_size, self.img_size), 
                             interpolation=cv2.INTER_CUBIC)
        
        # NN map (on all patches)
        if self.normal_memory is not None:
            distances = torch.cdist(fused, self.normal_memory, p=2)
            min_dist = distances.min(dim=1)[0].cpu().numpy()
            nn_map = min_dist.reshape(self.grid_size, self.grid_size)
            nn_map = cv2.resize(nn_map, (self.img_size, self.img_size), 
                               interpolation=cv2.INTER_CUBIC)
        else:
            nn_map = np.zeros((self.img_size, self.img_size))
        
        # Fuse maps
        fused_map = text_weight * text_map + nn_weight * nn_map
        
        # Apply foreground mask - hard suppression of background
        fused_map = fused_map * fg_mask_resized
        fused_map[~fg_mask_resized] = -999
        
        # Robust normalize (only on foreground)
        fg_values = fused_map[fg_mask_resized]
        if len(fg_values) > 0:
            median = np.median(fg_values)
            mad = np.median(np.abs(fg_values - median))
            if mad > 1e-6:
                fused_map = (fused_map - median) / (mad * 1.4826)
        fused_map = np.clip(fused_map, -3, 10) * fg_mask_resized
        
        # Smooth (preserving foreground boundaries)
        smoothed = cv2.GaussianBlur(fused_map.astype(np.float32), (0, 0), sigmaX=4.0)
        smoothed = smoothed * fg_mask_resized
        
        # Score (only from foreground)
        fg_scores = smoothed[fg_mask_resized]
        image_score = float(np.max(fg_scores)) if len(fg_scores) > 0 else 0.0
        
        # Find top hotspots (ONLY on foreground)
        hotspots = []
        if len(fg_scores) > 0:
            threshold = np.percentile(fg_scores, 95)
            hotspot_mask = (smoothed > threshold) & fg_mask_resized
            contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Validate: only accept if on foreground
                        if fg_mask_resized[cy, cx]:
                            score = float(smoothed[cy, cx])
                            hotspots.append((cx, cy, score))
            hotspots = sorted(hotspots, key=lambda x: x[2], reverse=True)[:5]
        
        return AnomalyResult(smoothed, image_score, fg_mask_resized, hotspots)


class LLaVAExplainer:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
                 gpu_util=0.25, auto_start=True):
        self.model_name = model_name
        self.gpu_util = gpu_util
        self.llm = None
        self.sampling = None
        self.enabled = False
        
        if auto_start:
            self._init_llava()
    
    def _init_llava(self, retries=3):
        """Initialize LLaVA with retry logic."""
        import time
        for attempt in range(retries):
            try:
                from vllm import LLM, SamplingParams
                print(f"Loading LLaVA (attempt {attempt+1}/{retries})...")
                self.llm = LLM(model=self.model_name,
                              gpu_memory_utilization=self.gpu_util, 
                              max_model_len=2048,
                              trust_remote_code=True)
                self.sampling = SamplingParams(temperature=0.2, max_tokens=200, 
                                              stop=["\n\n", "<|endoftext|>"])
                self.enabled = True
                print("✓ LLaVA loaded successfully")
                return True
            except Exception as e:
                print(f"LLaVA init attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("LLaVA unavailable, using deterministic fallback")
                    self.enabled = False
        return False
    
    def explain(self, image: np.ndarray, result: AnomalyResult, class_name: str, 
               heatmap: np.ndarray = None):
        """Generate explanation - GUARANTEED non-empty output."""
        # Validate hotspots are on object
        valid_hotspots = [
            (x, y, s) for x, y, s in result.hotspots 
            if result.foreground_mask[y, x]
        ]
        
        if not valid_hotspots:
            return f"No defect visible on the {class_name} object. All detected regions were background noise or below confidence threshold."
        
        # Update result with validated hotspots
        result.hotspots = valid_hotspots
        
        # Try LLaVA if enabled
        if self.enabled and self.llm is not None:
            try:
                explanation = self._llava_explain(image, result, class_name, heatmap)
                if explanation and len(explanation.strip()) > 10:
                    return explanation
            except Exception as e:
                print(f"LLaVA inference failed: {e}, using fallback")
        
        # Deterministic fallback - ALWAYS produces output
        return self._deterministic_fallback(result, class_name, image.shape)
    
    def _llava_explain(self, image: np.ndarray, result: AnomalyResult, 
                      class_name: str, heatmap: np.ndarray):
        """Call LLaVA with proper formatting."""
        import base64
        from io import BytesIO
        
        # Convert image to base64 data URL
        img_pil = Image.fromarray(image)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_url = f"data:image/png;base64,{img_base64}"
        
        # Build prompt for industrial inspection
        hotspot_locs = ", ".join([f"({x},{y})" for x, y, _ in result.hotspots[:3]])
        prompt = (
            f"<image>\n\nYou are inspecting an industrial {class_name} for defects. "
            f"Anomaly detected at positions: {hotspot_locs}. "
            f"Describe the abnormal region precisely: shape, location, texture, severity. "
            f"If nothing is wrong, say 'no defect visible'. Avoid speculation."
        )
        
        # Generate with retry
        for attempt in range(3):
            try:
                outputs = self.llm.generate(
                    {"prompt": prompt, "multi_modal_data": {"image": img_url}},
                    sampling_params=self.sampling
                )
                if outputs and len(outputs) > 0:
                    text = outputs[0].outputs[0].text.strip()
                    if text:
                        return text
            except Exception as e:
                if attempt == 2:
                    raise e
        return None
    
    def _deterministic_fallback(self, result: AnomalyResult, class_name: str, 
                               img_shape: tuple):
        """Deterministic fallback that NEVER returns empty string."""
        H, W = img_shape[:2]
        explanations = []
        
        for i, (x, y, score) in enumerate(result.hotspots[:3], 1):
            # Determine severity
            if score > 3.0:
                severity = "high-severity"
                issue = "significant structural defect"
            elif score > 2.0:
                severity = "moderate"
                issue = "visible anomaly"
            else:
                severity = "low"
                issue = "subtle irregularity"
            
            # Determine location
            x_rel, y_rel = x / W, y / H
            if y_rel < 0.33:
                loc_v = "top"
            elif y_rel < 0.67:
                loc_v = "middle"
            else:
                loc_v = "bottom"
            
            if x_rel < 0.33:
                loc_h = "left"
            elif x_rel < 0.67:
                loc_h = "center"
            else:
                loc_h = "right"
            
            location = f"{loc_v}-{loc_h}" if loc_v != "middle" or loc_h != "center" else "center"
            
            # Build description
            explanations.append(
                f"Defect {i} at {location} region (pixel {x},{y}): "
                f"{severity} {issue} detected on {class_name} surface, "
                f"confidence score {score:.2f}. "
                f"Likely {'scratch/dent/damage' if score > 2.0 else 'surface irregularity'}."
            )
        
        summary = f"Total {len(result.hotspots)} anomalous region(s) identified. "
        if result.image_score > 3.0:
            summary += "Major defect detected, object should be rejected."
        elif result.image_score > 1.5:
            summary += "Moderate anomaly, requires inspection."
        else:
            summary += "Minor irregularity detected."
        
        return " ".join(explanations) + " " + summary


def detect_and_explain(winclip, image_path, normal_anchor, anomalous_anchor,
                       class_name, output_dir="outputs", use_llava=True):
    """Full pipeline: detect + explain + save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load image
    img = np.array(Image.open(image_path).convert('RGB'))
    
    # Detect
    result = winclip.detect(image_path, normal_anchor, anomalous_anchor)
    
    # Explain - ALWAYS produces output
    explainer = LLaVAExplainer(auto_start=use_llava)
    explanation = explainer.explain(img, result, class_name, heatmap=result.anomaly_map)
    
    # Sanity check: explanation must never be empty
    if not explanation or len(explanation.strip()) < 10:
        explanation = explainer._deterministic_fallback(result, class_name, img.shape)
    
    # Save heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    im = ax.imshow(result.anomaly_map, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Anomaly Heatmap (score={result.image_score:.2f})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save boxes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    for x, y, score in result.hotspots:
        rect = patches.Rectangle((x-16, y-16), 32, 32, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-20, f"{score:.2f}", color='red', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.axis('off')
    ax.set_title(f"Detected Regions ({len(result.hotspots)} hotspots)")
    plt.tight_layout()
    boxes_path = output_dir / f"{image_path.stem}_boxes.png"
    plt.savefig(boxes_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save report
    report = f"""ANOMALY DETECTION REPORT
{'='*60}
Image: {image_path.name}
Class: {class_name}
Anomaly Score: {result.image_score:.3f}

DETECTED REGIONS ({len(result.hotspots)}):
"""
    for i, (x, y, score) in enumerate(result.hotspots, 1):
        report += f"\n  Region {i}: Position ({x},{y}), Score {score:.3f}"
    
    report += f"\n\nEXPLANATION:\n{explanation}\n"
    
    report_path = output_dir / f"{image_path.stem}_report.txt"
    report_path.write_text(report)
    
    print(f"Saved: {heatmap_path.name}, {boxes_path.name}, {report_path.name}")
    
    return {
        'result': result,
        'explanation': explanation,
        'heatmap': str(heatmap_path),
        'boxes': str(boxes_path),
        'report': str(report_path)
    }


class MVTecBenchmark:
    def __init__(self, dataset_root: Path, device="cuda"):
        self.dataset_root = Path(dataset_root)
        self.winclip = WinCLIP(device=device)
    
    def get_class_data(self, class_name: str):
        class_dir = self.dataset_root / class_name
        train_good = sorted((class_dir / "train" / "good").glob("*.png"))
        test_dir = class_dir / "test"
        test_good = sorted((test_dir / "good").glob("*.png"))
        test_defect = []
        for defect_dir in test_dir.iterdir():
            if defect_dir.is_dir() and defect_dir.name != "good":
                test_defect.extend(sorted(defect_dir.glob("*.png")))
        return {"train": train_good, "test_good": test_good, "test_defect": test_defect}
    
    def get_ground_truth_mask(self, image_path: Path):
        parts = image_path.parts
        if 'test' not in parts:
            return None
        test_idx = parts.index('test')
        defect_type = parts[test_idx + 1]
        if defect_type == 'good':
            img = Image.open(image_path)
            return np.zeros((img.height, img.width), dtype=bool)
        gt_path = Path(*parts[:test_idx]) / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        if not gt_path.exists():
            return None
        mask = np.array(Image.open(gt_path).convert('L'))
        return (mask > 127).astype(bool)
    
    def evaluate_class(self, class_name: str):
        print(f"\n{'='*60}\nEvaluating: {class_name}\n{'='*60}")
        data = self.get_class_data(class_name)
        if not data["train"]:
            return {}
        
        normal_anchor, anomalous_anchor = self.winclip.build_text_anchors(class_name)
        self.winclip.build_normal_memory(data["train"], normal_anchor, anomalous_anchor,
                                         coreset_ratio=0.1, max_images=50)
        
        all_img_scores, all_img_labels = [], []
        all_pix_scores, all_pix_labels = [], []
        
        for img_path in tqdm(data["test_good"], desc="Good"):
            result = self.winclip.detect(img_path, normal_anchor, anomalous_anchor)
            all_img_scores.append(result.image_score)
            all_img_labels.append(0)
            gt_mask = self.get_ground_truth_mask(img_path)
            if gt_mask is not None:
                if gt_mask.shape != result.anomaly_map.shape:
                    gt_mask = cv2.resize(gt_mask.astype(np.uint8), 
                                        (result.anomaly_map.shape[1], result.anomaly_map.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                all_pix_scores.append(result.anomaly_map.flatten())
                all_pix_labels.append(gt_mask.flatten())
        
        for img_path in tqdm(data["test_defect"], desc="Defect"):
            result = self.winclip.detect(img_path, normal_anchor, anomalous_anchor)
            all_img_scores.append(result.image_score)
            all_img_labels.append(1)
            gt_mask = self.get_ground_truth_mask(img_path)
            if gt_mask is not None:
                if gt_mask.shape != result.anomaly_map.shape:
                    gt_mask = cv2.resize(gt_mask.astype(np.uint8),
                                        (result.anomaly_map.shape[1], result.anomaly_map.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                all_pix_scores.append(result.anomaly_map.flatten())
                all_pix_labels.append(gt_mask.flatten())
        
        metrics = {}
        if len(set(all_img_labels)) > 1:
            metrics['img_auroc'] = roc_auc_score(all_img_labels, all_img_scores) * 100
        else:
            metrics['img_auroc'] = 0.0
        
        if all_pix_scores:
            pix_scores = np.concatenate(all_pix_scores)
            pix_labels = np.concatenate(all_pix_labels)
            if len(set(pix_labels)) > 1:
                metrics['pix_auroc'] = roc_auc_score(pix_labels, pix_scores) * 100
                metrics['pix_ap'] = average_precision_score(pix_labels, pix_scores) * 100
            else:
                metrics['pix_auroc'] = metrics['pix_ap'] = 0.0
        
        print(f"Results: Img-AUROC={metrics.get('img_auroc',0):.1f}% "
              f"Pix-AUROC={metrics.get('pix_auroc',0):.1f}% "
              f"Pix-AP={metrics.get('pix_ap',0):.1f}%")
        return metrics
    
    def run_full_benchmark(self, classes=None):
        if classes is None:
            classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                      'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                      'transistor', 'wood', 'zipper']
        
        results = {}
        for class_name in classes:
            if not (self.dataset_root / class_name).exists():
                continue
            try:
                results[class_name] = self.evaluate_class(class_name)
            except Exception as e:
                print(f"Error on {class_name}: {e}")
                results[class_name] = {}
        
        print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
        print(f"{'Class':<15} {'Img-AUROC':<12} {'Pix-AUROC':<12} {'Pix-AP':<12}")
        print("-"*70)
        
        avg_img, avg_pix_auroc, avg_pix_ap = [], [], []
        for class_name, m in results.items():
            img_auroc = m.get('img_auroc', 0)
            pix_auroc = m.get('pix_auroc', 0)
            pix_ap = m.get('pix_ap', 0)
            print(f"{class_name:<15} {img_auroc:<12.1f} {pix_auroc:<12.1f} {pix_ap:<12.1f}")
            if img_auroc > 0: avg_img.append(img_auroc)
            if pix_auroc > 0: avg_pix_auroc.append(pix_auroc)
            if pix_ap > 0: avg_pix_ap.append(pix_ap)
        
        print("-"*70)
        print(f"{'MEAN':<15} {np.mean(avg_img) if avg_img else 0:<12.1f} "
              f"{np.mean(avg_pix_auroc) if avg_pix_auroc else 0:<12.1f} "
              f"{np.mean(avg_pix_ap) if avg_pix_ap else 0:<12.1f}")
        print("="*70)
        return results


if __name__ == "__main__":
    import sys
    mvtec_root = Path("../mvtec_anomaly_detection")
    if not mvtec_root.exists():
        print(f"MVTec not found at {mvtec_root}")
        sys.exit(1)
    
    print("Starting full MVTec benchmark...")
    benchmark = MVTecBenchmark(mvtec_root)
    results = benchmark.run_full_benchmark()
    print("\nBenchmark complete!")
