from pathlib import Path
import json

import torch
from torchvision.models.detection import (
fasterrcnn_resnet50_fpn_v2,
FasterRCNN_ResNet50_FPN_V2_Weights
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_DIR / "frames"
OUTPUT_DIR = PROJECT_DIR / "results" / "detections"

CONFIDENCE_THRESHOLD = 0.5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not FRAMES_DIR.is_dir():
        raise NotADirectoryError("Frames directory does not exist")

    frames = sorted(
        path for path in FRAMES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not frames:
        raise RuntimeError(f"No frames found in {FRAMES_DIR}")

    print(f"Found {len(frames)} frames")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
    ).to(device)
    model.eval()

    categories = weights.meta["categories"]

    transform = weights.transforms()

    with torch.inference_mode():
        for index, frame_path in enumerate(frames, start=1):
            from PIL import Image

            with Image.open(frame_path) as img:
                img = img.convert("RGB")

            width, height = img.size
            image_tensor = transform(img).to(device)
            prediction = model([image_tensor])[0]

            detections = []

            for box, label, score in zip(
                prediction["boxes"],
                prediction["labels"],
                prediction["scores"],
            ):
                confidence = float(score.item())

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(label.item())

                detections.append({
                    "class_id": class_id,
                    "class_name": categories[class_id],
                    "confidence": round(confidence, 2),
                    "bbox_xyxy": [
                        round(float(value), 2)
                        for value in box.tolist()
                    ]
                })

            result = {
                "frame": frame_path.name,
                "image_width": width,
                "image_height": height,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "detections": detections,
            }

            output_path = OUTPUT_DIR / f"{frame_path.stem}.json"

            with output_path.open("w") as f:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[{index}/{len(frames)}] {frame_path.name}: " f"objects found — {len(detections)}"
            )

    print(f"Successfully saved detections into {OUTPUT_DIR}")

if __name__ == "__main__":
    main()