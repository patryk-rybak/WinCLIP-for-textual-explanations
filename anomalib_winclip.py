import os
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models.image import WinClip
from torchvision.utils import save_image
import torch.nn.functional as F


def extract_max_anomaly_crop(
    image: torch.Tensor,          # (3, H, W)
    anomaly_map: torch.Tensor,    # (h, w)
    crop_size: int = 256,
):
    """
    Zaznacza na obrazie czerwonym obramowaniem obszary wokół
    1–2 najsilniejszych, rozdzielnych punktów anomalii.
    """
    _, H, W = image.shape
    h, w = anomaly_map.shape

    # 1. Wybór 1–2 lokalnych maksimów (bardziej stabilnie niż top-2)
    if anomaly_map.numel() == 0:
        return image

    am = anomaly_map
    pooled = F.max_pool2d(am[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    peaks = (am == pooled)

    peak_vals = am[peaks]
    if peak_vals.numel() == 0:
        return image

    peak_indices = torch.nonzero(peaks, as_tuple=False)
    # sortuj piki po wartości (malejąco)
    sorted_idx = torch.argsort(peak_vals, descending=True)

    # próg względny dla drugiego piku
    rel_threshold = 0.4
    min_separation = max(1, int(0.1 * min(h, w)))

    selected = []
    for idx in sorted_idx:
        y_new, x_new = peak_indices[idx].tolist()
        val = peak_vals[idx]

        if not selected:
            selected.append((val, y_new, x_new))
            continue

        # zaakceptuj drugi pik tylko jeśli jest wystarczająco silny
        if val < selected[0][0] * rel_threshold:
            continue

        y_prev, x_prev = selected[0][1], selected[0][2]
        if abs(y_new - y_prev) < min_separation and abs(x_new - x_prev) < min_separation:
            continue

        selected.append((val, y_new, x_new))
        break

    # 2. Skalowanie do rozdzielczości obrazu i wyznaczenie ramek
    scale_y = H / h
    scale_x = W / w
    half = crop_size // 2

    # 4. Zaznaczenie czerwonej ramki na oryginalnym obrazie
    # (ramka o grubości 2 px)
    marked = image.clone()
    thickness = 2

    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)

    for _, y_map, x_map in selected:
        y_img = int(y_map * scale_y)
        x_img = int(x_map * scale_x)

        y1 = max(0, y_img - half)
        y2 = min(H, y_img + half)
        x1 = max(0, x_img - half)
        x2 = min(W, x_img + half)

        # upewnij się, że ramka mieści się w obrazie
        y1b = max(0, y1)
        y2b = min(H, y2)
        x1b = max(0, x1)
        x2b = min(W, x2)

        # górna i dolna krawędź
        marked[:, y1b:y1b + thickness, x1b:x2b] = red
        marked[:, y2b - thickness:y2b, x1b:x2b] = red
        # lewa i prawa krawędź
        marked[:, y1b:y2b, x1b:x1b + thickness] = red
        marked[:, y1b:y2b, x2b - thickness:x2b] = red

    return marked


def save_anomaly_predicitons(
    category="screw",
    output_dir=Path("./predictions"),
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
        image_paths = batch.image_path

        batch_size = images.shape[0]

        with torch.no_grad():
            outputs = model(images)

        for i in range(batch_size):
            image = images[i].cpu()

            anomaly_map = outputs.anomaly_map[i].cpu()
            pred_label = outputs.pred_label[i].item()

            marked = extract_max_anomaly_crop(
                image=image,
                anomaly_map=anomaly_map,
                crop_size=256,
            )

            label_txt = "ANOMALIA" if pred_label else "NORMALNY"
            # zapis do: predictions/<category>/<subdir>/...
            subdir = Path(image_paths[i]).parent.name
            out_dir = output_dir / subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"image_{img_idx:03d}_{label_txt}.png"
            out_path = out_dir / out_name

            save_image(marked, out_path)

            print(f"[{label_txt}] {out_name} | label={pred_label}")
            img_idx += 1


    print(f"\nGotowe! Obrazy zapisano w: {Path(output_dir).absolute()}")


# --------------------------------------------------
# URUCHOMIENIE
# --------------------------------------------------
if __name__ == "__main__":
    dataset_root = Path("./dataset")
    predictions_root = Path("./predictions")

    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        category = category_dir.name
        output_dir = predictions_root / category

        save_anomaly_predicitons(
            category=category,
            output_dir=output_dir,
        )
