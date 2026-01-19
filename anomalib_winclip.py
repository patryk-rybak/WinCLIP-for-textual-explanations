# from anomalib.data import MVTecAD

# datamodule = MVTecAD(
#     root="./dataset",
#     category="wood",
#     train_batch_size=32,
#     eval_batch_size=2,
#     num_workers=8
# )

# datamodule.setup()

# test_loader = datamodule.test_dataloader()

# from anomalib.models.image import WinClip

# model = WinClip(class_name='wood')
# model.setup('wood')

# batch = next(iter(test_loader))
# res = model(batch.image)
# print(res)


import os
from pathlib import Path

import torch
from anomalib.data import ImageItem, MVTecAD
from anomalib.models.image import WinClip
from anomalib.visualization.image.item_visualizer import visualize_image_item


def save_anomaly_overlays(
    category="screw",
    indices=None,
    output_dir="./predictions",
    overlay_alpha=0.3,  # przezroczystość overlay
):
    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------
    datamodule = MVTecAD(
        root="./dataset",
        category=category,
        train_batch_size=32,
        eval_batch_size=10,
        num_workers=4
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # --------------------------------------------------
    # 2. Model WinCLIP (zero-shot)
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WinClip(class_name=category).to(device)
    model.setup(category)  # przygotowanie embeddings
    model.eval()

    # --------------------------------------------------
    # 3. Folder wyjściowy
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # 4. Iteracja po batchach
    # --------------------------------------------------
    img_idx = 0
    for batch in test_loader:
        # batch.image może nie istnieć w nowszych wersjach, używamy tensora bezpośrednio
        if hasattr(batch, "image"):
            images = batch.image.to(device)
        else:
            # batch jest tensorem lub ImageItem listą
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        batch_size = images.shape[0]

        with torch.no_grad():
            outputs = model(images)

        for i in range(batch_size):
            image = images[i].cpu()
            anomaly_map = outputs.anomaly_map[i].cpu()
            pred_label = outputs.pred_label[i].item()
            pred_score = outputs.pred_score[i].item() if hasattr(outputs, "pred_score") else None

            if hasattr(outputs, "pred_mask"):
                pred_mask = outputs.pred_mask[i].cpu()
            else:
                pred_mask = (anomaly_map > anomaly_map.mean()).to(dtype=torch.uint8)

            image_path = None
            if hasattr(batch, "image_path"):
                try:
                    image_path = batch.image_path[i]
                except Exception:
                    image_path = None

            item = ImageItem(
                image=image,
                image_path=image_path,
                anomaly_map=anomaly_map,
                pred_mask=pred_mask,
                pred_score=pred_score,
                pred_label=pred_label,
            )

            # vis = visualize_image_item(
            #     item,
            #     fields=["image", "anomaly_map", "pred_mask"],
            #     fields_config={
            #         "anomaly_map": {"colormap": True, "normalize": True},
            #         "pred_mask": {"mode": "contour", "color": (255, 0, 0), "alpha": overlay_alpha},
            #     },
            #     overlay_fields=[("image", ["pred_mask"])],
            #     overlay_fields_config={
            #         "pred_mask": {"mode": "contour", "color": (255, 0, 0), "alpha": overlay_alpha},
            #     },
            # )
            # KONKATYNACJA: poniżej powstają 2 obrazki (bazowy i overlay) sklejone obok siebie
            # vis = visualize_image_item(
            #     item,
            #     fields=["image"],
            #     overlay_fields=[("image", ["anomaly_map"])],
            #     overlay_fields_config={
            #         "anomaly_map": {"colormap": True, "normalize": True, "alpha": overlay_alpha},
            #     },
            # )

            # Bez konkatenacji: zwracamy tylko overlay anomaly_map na obrazie bazowym
            vis = visualize_image_item(
                item,
                fields=None,
                overlay_fields=[("image", ["anomaly_map"])],
                overlay_fields_config={
                    "anomaly_map": {"colormap": False, "normalize": True},
                },
                text_config={"enable": False},
            )

            breakpoint()

            label_txt = "ANOMALIA" if pred_label else "NORMALNY"
            if image_path:
                stem = Path(str(image_path)).stem
                out_name = f"{stem}_{label_txt}.png"
            else:
                out_name = f"image_{img_idx:03d}_{label_txt}.png"

            out_path = os.path.join(output_dir, out_name)
            vis.save(out_path)

            print(f"[{label_txt}] {out_name} | label={pred_label}")
            img_idx += 1

            break

    print(f"\nGotowe! Obrazy zapisano w: {Path(output_dir).absolute()}")


# --------------------------------------------------
# URUCHOMIENIE
# --------------------------------------------------
if __name__ == "__main__":
    save_anomaly_overlays(
        category="screw",
        indices=None,
        output_dir="./predictions",
        overlay_alpha=0.3
    )
