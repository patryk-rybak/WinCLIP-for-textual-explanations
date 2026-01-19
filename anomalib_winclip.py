import os
from pathlib import Path

import torch
from anomalib.data import ImageItem, MVTecAD
from anomalib.models.image import WinClip
from anomalib.visualization.image.item_visualizer import visualize_image_item

import torch.nn.functional as F
from torchvision.utils import save_image


def extract_max_anomaly_crop(
    image: torch.Tensor,          # (3, H, W)
    anomaly_map: torch.Tensor,    # (h, w)
    crop_size: int = 256,
    out_size: int = 1024,
):
    """
    Zwraca zoomowany crop obrazu wokół punktu maksymalnej anomalii.
    """
    _, H, W = image.shape
    h, w = anomaly_map.shape

    # 1. Argmax w anomaly_map
    flat_idx = torch.argmax(anomaly_map)
    y_map, x_map = divmod(flat_idx.item(), w)

    # 2. Skalowanie do rozdzielczości obrazu
    scale_y = H / h
    scale_x = W / w

    y_img = int(y_map * scale_y)
    x_img = int(x_map * scale_x)

    # 3. Wyznaczenie cropa
    half = crop_size // 2

    y1 = max(0, y_img - half)
    y2 = min(H, y_img + half)
    x1 = max(0, x_img - half)
    x2 = min(W, x_img + half)

    crop = image[:, y1:y2, x1:x2]

    # 4. Jeśli crop jest mniejszy (krawędzie obrazu) → resize
    crop = crop.unsqueeze(0)  # (1, 3, h, w)
    crop = F.interpolate(
        crop,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )

    return crop.squeeze(0)  # (3, out_size, out_size)


def save_anomaly_predicitons(
    category="screw",
    output_dir="./predictions",
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
        images = batch.image.to(device)

        batch_size = images.shape[0]

        with torch.no_grad():
            outputs = model(images)

        for i in range(batch_size):
            image = images[i].cpu()

            anomaly_map = outputs.anomaly_map[i].cpu()
            pred_label = outputs.pred_label[i].item()
            pred_score = outputs.pred_score[i].item()
            pred_mask = outputs.pred_mask[i].cpu()

            zoom = extract_max_anomaly_crop(
                image=image,
                anomaly_map=anomaly_map,
                crop_size=256,
                out_size=1024,
            )

            concat = torch.cat([image, zoom], dim=2)

            label_txt = "ANOMALIA" if pred_label else "NORMALNY"
            out_name = f"image_{img_idx:03d}_{label_txt}.png"
            out_path = os.path.join(output_dir, out_name)

            save_image(concat, out_path)

            SAVE_ADDITIONAL_ANOMALY_MASK = True

            if SAVE_ADDITIONAL_ANOMALY_MASK:

                item = ImageItem(
                    image=image,
                    image_path=None,
                    anomaly_map=anomaly_map,
                    pred_mask=pred_mask,
                    pred_score=pred_score,
                    pred_label=pred_label,
                )
                vis = visualize_image_item(
                    item,
                    fields=None,
                    overlay_fields=[("image", ["anomaly_map"])],
                    overlay_fields_config={
                        "anomaly_map": {"colormap": True, "normalize": True},
                    },
                    text_config={"enable": False},
                )

                out_name_anomaly = f"image_{img_idx:03d}_{label_txt}_anomaly_amp.png"
                out_path_anomaly = os.path.join(output_dir, out_name_anomaly)
                vis.save(out_path_anomaly)

            print(f"[{label_txt}] {out_name} | label={pred_label}")
            img_idx += 1


    print(f"\nGotowe! Obrazy zapisano w: {Path(output_dir).absolute()}")


# --------------------------------------------------
# URUCHOMIENIE
# --------------------------------------------------
if __name__ == "__main__":
    save_anomaly_predicitons(
        category="screw",
        output_dir="./predictions",
    )
