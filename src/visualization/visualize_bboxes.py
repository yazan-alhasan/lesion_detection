from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import parse_yolo_line, yolo_to_voc
from src.utils.paths import ensure_dir, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO boxes on processed PNG images.")
    parser.add_argument("--images_dir", default="data/processed/images/train")
    parser.add_argument("--labels_dir", default="data/processed/labels/train")
    parser.add_argument("--classes_file", default="data/processed/classes.txt")
    parser.add_argument("--output_dir", default="runs/visualization")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def load_classes(path) -> list[str]:
    path = project_path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    images_dir = project_path(args.images_dir)
    labels_dir = project_path(args.labels_dir)
    output_dir = ensure_dir(args.output_dir)
    classes = load_classes(args.classes_file)

    image_paths = sorted(images_dir.glob("*.png"))
    random.Random(args.random_seed).shuffle(image_paths)
    image_paths = image_paths[: args.num_samples]

    saved = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        for line in lines:
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            class_id, x_center, y_center, box_width, box_height = parsed
            xmin, ymin, xmax, ymax = yolo_to_voc(x_center, y_center, box_width, box_height, width, height)
            label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
            color = (0, 220, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                image,
                label,
                (xmin, max(18, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), image)
        saved += 1

    print(f"Saved {saved} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
