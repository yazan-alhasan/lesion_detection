from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import apply_clahe_to_mammogram
from src.utils.bbox_utils import parse_yolo_line, yolo_to_voc
from src.utils.paths import PROJECT_ROOT, ensure_dir, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CLAHE-enhanced copy of an existing YOLO dataset.")
    parser.add_argument("--input_dataset", default="data/dataset_mass_cropped")
    parser.add_argument("--output_dataset", default="data/dataset_mass_cropped_clahe")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid_size", type=int, default=8)
    parser.add_argument("--save_debug_samples", action="store_true")
    parser.add_argument("--num_debug_samples", type=int, default=20)
    return parser.parse_args()


def portable_dataset_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_class_name(dataset_root: Path) -> str:
    classes_file = dataset_root / "classes.txt"
    if classes_file.exists():
        classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if classes:
            return classes[0]
    return "mass"


def draw_yolo_boxes(image, label_path: Path, class_name: str):
    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    height, width = image.shape[:2]
    if not label_path.exists():
        return output
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        _, x_center, y_center, box_width, box_height = parsed
        xmin, ymin, xmax, ymax = yolo_to_voc(x_center, y_center, box_width, box_height, width, height)
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            output,
            class_name,
            (xmin, max(15, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def copy_metadata(input_root: Path, output_root: Path, class_name: str) -> None:
    for filename in ("classes.txt", "class_mapping.json"):
        source = input_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)
    dataset_yaml = {
        "path": portable_dataset_path(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: class_name},
    }
    (output_root / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = project_path(args.input_dataset)
    output_root = ensure_dir(args.output_dataset)
    class_name = read_class_name(input_root)

    stats = {
        "source_dataset": portable_dataset_path(input_root),
        "target_class_name": class_name,
        "clahe_enabled": True,
        "clahe_applied_after_crop": True,
        "clahe_clip_limit": float(args.clahe_clip_limit),
        "clahe_tile_grid_size": int(args.clahe_tile_grid_size),
        "num_clahe_failures": 0,
        "num_train_images": 0,
        "num_val_images": 0,
        "num_test_images": 0,
        "num_train_boxes": 0,
        "num_val_boxes": 0,
        "num_test_boxes": 0,
    }
    debug_saved = 0

    for split in ("train", "val", "test"):
        image_input = input_root / "images" / split
        label_input = input_root / "labels" / split
        image_output = ensure_dir(output_root / "images" / split)
        label_output = ensure_dir(output_root / "labels" / split)
        if not image_input.exists():
            continue

        for image_path in sorted(image_input.glob("*")):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                stats["num_clahe_failures"] += 1
                continue
            try:
                enhanced = apply_clahe_to_mammogram(
                    image,
                    clip_limit=args.clahe_clip_limit,
                    tile_grid_size=args.clahe_tile_grid_size,
                )
                assert enhanced.shape[:2] == image.shape[:2], "CLAHE changed image size, which should never happen."
            except Exception as exc:
                print(f"[WARNING] CLAHE failed for {image_path.name}, saving original image: {exc}")
                enhanced = image
                stats["num_clahe_failures"] += 1

            cv2.imwrite(str(image_output / f"{image_path.stem}.png"), enhanced)
            source_label = label_input / f"{image_path.stem}.txt"
            target_label = label_output / f"{image_path.stem}.txt"
            if source_label.exists():
                shutil.copy2(source_label, target_label)
                box_count = sum(1 for line in source_label.read_text(encoding="utf-8").splitlines() if parse_yolo_line(line))
            else:
                target_label.write_text("", encoding="utf-8")
                box_count = 0

            stats[f"num_{split}_images"] += 1
            stats[f"num_{split}_boxes"] += box_count

            if args.save_debug_samples and debug_saved < args.num_debug_samples:
                before = draw_yolo_boxes(image, source_label, class_name)
                after = draw_yolo_boxes(enhanced, source_label, class_name)
                if before.shape[:2] != after.shape[:2]:
                    after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
                debug_path = output_root / "debug_samples_clahe_compare" / split / f"{image_path.stem}.png"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path), cv2.hconcat([before, after]))
                debug_saved += 1

    stats["total_images"] = stats["num_train_images"] + stats["num_val_images"] + stats["num_test_images"]
    stats["total_boxes"] = stats["num_train_boxes"] + stats["num_val_boxes"] + stats["num_test_boxes"]
    copy_metadata(input_root, output_root, class_name)
    (output_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote CLAHE YOLO dataset to {output_root}")


if __name__ == "__main__":
    main()
