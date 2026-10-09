
import json
import csv
import re
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIG
# ============================================================

FRAMES_DIR = Path("frames")
DETECTIONS_DIR = Path("results/detections")
SEGMENTATION_DIR = Path("results/segmentation")
OUTPUT_DIR = Path("results/visualization")

# Ожидаемая структура сегментации:
MASK_DIR = SEGMENTATION_DIR / "masks"
COLOR_MASK_DIR = SEGMENTATION_DIR / "colored_masks"

DETECTION_GIF = OUTPUT_DIR / "detection.gif"
SEGMENTATION_GIF = OUTPUT_DIR / "segmentation.gif"

# Длительность одного кадра GIF, миллисекунды.
FRAME_DURATION_MS = 100

# Прозрачность цветной маски при наложении.
MASK_ALPHA = 0.45

# Минимальный размер шрифта и толщина рамки.
FONT_SIZE = 16
BOX_WIDTH = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def natural_key(path):
    """Сортировка frame_2.jpg перед frame_10.jpg."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_font(size=FONT_SIZE):
    """Загрузка шрифта с поддержкой кириллицы, если доступен."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)

    return ImageFont.load_default()


FONT = load_font()
SMALL_FONT = load_font(13)


def get_class_color(class_id):
    """Один и тот же класс всегда получает один цвет."""
    rng = np.random.default_rng(int(class_id) + 2025)
    color = rng.integers(55, 256, size=3)
    return tuple(int(c) for c in color)


def find_frames():
    frames = sorted(
        (
            p for p in FRAMES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
        ),
        key=natural_key,
    )

    if not frames:
        raise FileNotFoundError(
            f"В папке {FRAMES_DIR.resolve()} нет кадров."
        )

    return frames


def load_detection_index():
    """
    Поддерживает отдельный JSON на кадр или несколько JSON
    с полем frame, указывающим имя исходного кадра.
    """
    if not DETECTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка {DETECTIONS_DIR.resolve()}"
        )

    index = {}

    for path in sorted(DETECTIONS_DIR.rglob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        # JSON может содержать один кадр или список кадров.
        records = data if isinstance(data, list) else [data]

        for record in records:
            if not isinstance(record, dict):
                continue

            frame_name = record.get("frame", path.stem + ".jpg")
            index[Path(frame_name).name] = record

    return index


def save_gif(images, output_path):
    if len(images) != 20:
        raise ValueError(
            f"{output_path.name}: ожидалось 20 кадров, "
            f"получено {len(images)}"
        )

    rgb_images = [image.convert("RGB") for image in images]

    rgb_images[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=rgb_images[1:],
        duration=100,  # 100 мс на кадр = 10 FPS
        loop=0,
        disposal=2,
        optimize=False,
    )

    # Проверяем фактическое число кадров в сохранённом GIF.
    with Image.open(output_path) as gif:
        frame_count = getattr(gif, "n_frames", 1)

    print(
        f"{output_path}: сохранено кадров = {frame_count}"
    )

    if frame_count != 20:
        raise RuntimeError(
            f"GIF содержит {frame_count} кадров вместо 20"
        )

# ============================================================
# TASK 5: OBJECT DETECTION VISUALIZATION
# ============================================================

def render_detection(frame_path, detection_record):
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    detections = (
        detection_record.get("detections", [])
        if detection_record
        else []
    )

    width, height = image.size

    for detection in detections:
        bbox = detection.get("bbox_xyxy")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # Ограничиваем рамку границами изображения.
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        class_id = int(detection.get("class_id", 0))
        class_name = str(
            detection.get("class_name", f"class_{class_id}")
        )
        confidence = float(detection.get("confidence", 0.0))

        color = get_class_color(class_id)

        # Bounding box.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=BOX_WIDTH,
        )

        # Подпись: класс и confidence.
        label = f"{class_name} {confidence:.2f}"

        text_bbox = draw.textbbox(
            (0, 0), label, font=FONT, stroke_width=0
        )
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_x = max(0, min(int(x1), width - text_width - 8))
        label_y = max(0, int(y1) - text_height - 8)

        # Контрастный фон подписи.
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 8,
                label_y + text_height + 8,
            ),
            fill=color,
        )

        draw.text(
            (label_x + 4, label_y + 3),
            label,
            font=FONT,
            fill=(0, 0, 0),
        )

    # Служебная надпись.
    draw.text(
        (8, 8),
        f"Detections: {len(detections)}",
        font=SMALL_FONT,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    return image


def build_detection_gif(frames, detection_index):
    rendered = []

    for i, frame_path in enumerate(frames, start=1):
        record = detection_index.get(frame_path.name)

        if record is None:
            print(
                f"[WARNING] Нет детекций для {frame_path.name}"
            )

        rendered.append(render_detection(frame_path, record))

        print(
            f"[Detection {i}/{len(frames)}] {frame_path.name}"
        )

    save_gif(rendered, DETECTION_GIF)


# ============================================================
# TASK 6: SEMANTIC SEGMENTATION VISUALIZATION
# ============================================================

def load_class_names():
    """
    Берём названия классов из CSV, созданного при сегментации.
    Если CSV отсутствует, используем ID классов.
    """
    csv_path = SEGMENTATION_DIR / "class_pixel_counts.csv"
    names = {}

    if csv_path.exists():
        with csv_path.open(
            "r", encoding="utf-8", newline=""
        ) as file:
            for row in csv.DictReader(file):
                try:
                    class_id = int(row["class_id"])
                    class_name = row["class_name"]
                    names[class_id] = class_name
                except (KeyError, ValueError):
                    continue

    return names


def find_segmentation_file(frame_path):
    """
    Сначала ищем числовую маску, затем цветную маску.
    Имена должны соответствовать исходным кадрам.
    """
    candidates = [
        MASK_DIR / f"{frame_path.stem}.png",
        COLOR_MASK_DIR / f"{frame_path.stem}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Допускаем маски, лежащие непосредственно в segmentation.
    direct = SEGMENTATION_DIR / f"{frame_path.stem}.png"
    if direct.exists():
        return direct

    return None


def render_segmentation(frame_path, class_names):
    image = Image.open(frame_path).convert("RGB")
    mask_path = find_segmentation_file(frame_path)

    if mask_path is None:
        raise FileNotFoundError(
            f"Маска для {frame_path.name} не найдена. "
            f"Проверь {MASK_DIR} и {COLOR_MASK_DIR}."
        )

    mask_image = Image.open(mask_path)

    # Палитровый PNG (P) с ID классов также является
    # индексированной маской, а не RGB-изображением.
    if mask_image.mode in ("L", "P", "I", "I;16", "1"):
        mask = np.asarray(mask_image)

        # Для семантических масок обычно достаточно 8 бит.
        # Если маска сохранена как 16-битная, значения ID
        # сохраняются при чтении через NumPy.
        if mask.ndim != 2:
            raise ValueError(
                f"Ожидалась двумерная маска: {mask_path}"
            )

        mask = Image.fromarray(mask.astype(np.int32)).resize(
            image.size, resample=Image.Resampling.NEAREST
        )
        mask = np.asarray(mask)

        # Игнорируемые/неизвестные классы можно исключить,
        # если они представлены ID 255.
        valid_mask = mask != 255
        colored = np.zeros(
            (image.height, image.width, 3),
            dtype=np.uint8,
        )

        class_ids, counts = np.unique(
            mask[valid_mask], return_counts=True
        )

        for class_id in class_ids:
            class_id = int(class_id)
            colored[mask == class_id] = get_class_color(class_id)

        base = np.asarray(image, dtype=np.float32)
        overlay = base.copy()

        overlay[valid_mask] = (
            base[valid_mask] * (1 - MASK_ALPHA)
            + colored[valid_mask] * MASK_ALPHA
        )

        result = Image.fromarray(
            np.clip(overlay, 0, 255).astype(np.uint8)
        )

        # Легенда: классы, присутствующие на кадре.
        legend_items = []
        for class_id, count in zip(class_ids, counts):
            name = class_names.get(
                int(class_id), f"class_{int(class_id)}"
            )
            percent = 100 * int(count) / mask.size
            legend_items.append(
                (int(class_id), name, percent)
            )

    else:
        # Если уже сохранена RGB-цветная маска, используем
        # её цвета напрямую: восстановить ID классов по RGB
        # без исходной палитры надёжно нельзя.
        color_mask = mask_image.convert("RGB").resize(
            image.size, Image.Resampling.NEAREST
        )
        result = Image.blend(image, color_mask, MASK_ALPHA)
        legend_items = []

    # Добавляем компактную легенду классов.
    if legend_items:
        draw = ImageDraw.Draw(result)
        max_items = 8
        visible_items = legend_items[:max_items]

        line_height = 20
        panel_width = min(300, result.width - 16)
        panel_height = len(visible_items) * line_height + 12

        panel = Image.new(
            "RGBA",
            (panel_width, panel_height),
            (0, 0, 0, 185),
        )
        panel_draw = ImageDraw.Draw(panel)

        for i, (class_id, name, percent) in enumerate(
            visible_items
        ):
            y = 6 + i * line_height
            panel_draw.rectangle(
                (7, y + 2, 19, y + 14),
                fill=get_class_color(class_id),
            )

            label = f"{name}: {percent:.1f}%"
            panel_draw.text(
                (25, y),
                label,
                font=SMALL_FONT,
                fill=(255, 255, 255),
            )

        # Размещаем легенду в левом верхнем углу.
        result_rgba = result.convert("RGBA")
        result_rgba.alpha_composite(panel, (8, 8))
        result = result_rgba.convert("RGB")

        if len(legend_items) > max_items:
            print(
                f"[INFO] {frame_path.name}: "
                f"в легенде показаны первые {max_items} "
                f"из {len(legend_items)} классов."
            )

    return result


def build_segmentation_gif(frames, class_names):
    rendered = []

    for i, frame_path in enumerate(frames, start=1):
        rendered.append(
            render_segmentation(frame_path, class_names)
        )

        print(
            f"[Segmentation {i}/{len(frames)}] "
            f"{frame_path.name}"
        )

    save_gif(rendered, SEGMENTATION_GIF)


# ============================================================
# MAIN
# ============================================================

def main():
    frames = find_frames()
    print(f"Найдено исходных кадров: {len(frames)}")

    detection_index = load_detection_index()
    print(
        f"Загружено записей детекции: {len(detection_index)}"
    )

    class_names = load_class_names()

    build_detection_gif(frames, detection_index)
    build_segmentation_gif(frames, class_names)

    print("\nВизуализация завершена.")
    print(f"Detection GIF:   {DETECTION_GIF.resolve()}")
    print(f"Segmentation GIF: {SEGMENTATION_GIF.resolve()}")


if __name__ == "__main__":
    main()