from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

# Папка с кадрами из предыдущего этапа
INPUT_DIR = Path("frames")

# Результаты сегментации
OUTPUT_DIR = Path("results") / "segmentation"
MASK_DIR = OUTPUT_DIR / "masks"
COLOR_DIR = OUTPUT_DIR / "colored_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

for directory in (MASK_DIR, COLOR_DIR, OVERLAY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME
).to(device)
model.eval()

# Фиксированная палитра для визуализации
rng = np.random.default_rng(42)
palette = rng.integers(
    0, 256, size=(model.config.num_labels, 3),
    dtype=np.uint8
)
palette[0] = [0, 0, 0]

image_paths = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

if not image_paths:
    raise FileNotFoundError(
        f"No images found in {INPUT_DIR.resolve()}"
    )

csv_path = OUTPUT_DIR / "class_pixel_counts.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "class_id", "class_name", "pixels"])

    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = processor(
            images=image, return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            # Приводим предсказания к размеру исходного кадра
            prediction = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(height, width)]
            )[0]

        mask = prediction.cpu().numpy().astype(np.uint8)

        # 1. Маска с ID классов
        Image.fromarray(mask, mode="L").save(
            MASK_DIR / f"{image_path.stem}.png"
        )

        # 2. Цветная карта классов
        colored = palette[mask]
        Image.fromarray(colored, mode="RGB").save(
            COLOR_DIR / f"{image_path.stem}.png"
        )

        # 3. Наложение на исходный кадр
        overlay = (
            0.55 * np.asarray(image, dtype=np.float32)
            + 0.45 * colored.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        Image.fromarray(overlay).save(
            OVERLAY_DIR / f"{image_path.stem}.png"
        )

        # 4. Статистика классов
        class_ids, counts = np.unique(
            mask, return_counts=True
        )

        for class_id, pixel_count in zip(class_ids, counts):
            class_name = model.config.id2label.get(
                int(class_id), str(class_id)
            )
            writer.writerow([
                image_path.name,
                int(class_id),
                class_name,
                int(pixel_count),
            ])

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} processed"
        )

print(f"Done. Results saved to: {OUTPUT_DIR.resolve()}")