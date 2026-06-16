from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import has_valid_bbox, valid_yolo_box, voc_to_yolo
from src.utils.paths import ensure_dir, ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VinDr-Mammo boxes to single-class YOLO labels.")
    parser.add_argument("--finding_csv", required=True)
    parser.add_argument("--metadata_csv", default="data/processed/image_metadata.csv")
    parser.add_argument("--labels_output_dir", default="data/processed/labels/all")
    parser.add_argument("--classes_output", default="data/processed/classes.txt")
    parser.add_argument("--mapping_output", default="data/processed/class_mapping.json")
    parser.add_argument("--dataset_yaml", default="configs/dataset_single_class.yaml")
    parser.add_argument("--invalid_boxes_output", default="runs/invalid_boxes_single_class.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings = pd.read_csv(project_path(args.finding_csv))
    metadata = pd.read_csv(project_path(args.metadata_csv))
    labels_dir = ensure_dir(args.labels_output_dir)

    findings["image_id"] = findings["image_id"].astype(str)
    metadata["image_id"] = metadata["image_id"].astype(str)
    converted_ids = set(metadata["image_id"])
    findings = findings[findings["image_id"].isin(converted_ids)].copy()

    for image_id in converted_ids:
        (labels_dir / f"{image_id}.txt").write_text("", encoding="utf-8")

    meta_by_id = metadata.set_index("image_id").to_dict(orient="index")
    grouped_lines: dict[str, list[str]] = {image_id: [] for image_id in converted_ids}
    invalid_rows = []

    for _, row in findings.iterrows():
        image_id = str(row["image_id"])
        if not has_valid_bbox(row):
            continue

        meta = meta_by_id[image_id]
        xmin = float(row["xmin"]) * float(meta["scale_x"])
        ymin = float(row["ymin"]) * float(meta["scale_y"])
        xmax = float(row["xmax"]) * float(meta["scale_x"])
        ymax = float(row["ymax"]) * float(meta["scale_y"])

        yolo = voc_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            float(meta["processed_width"]),
            float(meta["processed_height"]),
        )
        if yolo is None or not valid_yolo_box(yolo):
            invalid = row.to_dict()
            invalid.update(
                {
                    "scaled_xmin": xmin,
                    "scaled_ymin": ymin,
                    "scaled_xmax": xmax,
                    "scaled_ymax": ymax,
                    "processed_width": meta["processed_width"],
                    "processed_height": meta["processed_height"],
                    "reason": "invalid_normalized_box",
                }
            )
            invalid_rows.append(invalid)
            continue

        grouped_lines[image_id].append(f"0 {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")

    for image_id, lines in grouped_lines.items():
        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    ensure_parent(args.classes_output).write_text("Lesion\n", encoding="utf-8")
    ensure_parent(args.mapping_output).write_text(json.dumps({"Lesion": 0}, indent=2) + "\n", encoding="utf-8")

    dataset = {
        "path": "data/processed",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Lesion"},
    }
    ensure_parent(args.dataset_yaml).write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    invalid_output = ensure_parent(args.invalid_boxes_output)
    pd.DataFrame(invalid_rows).to_csv(invalid_output, index=False)

    positive_images = sum(1 for lines in grouped_lines.values() if lines)
    total_boxes = sum(len(lines) for lines in grouped_lines.values())
    print(f"Wrote single-class labels for {len(converted_ids)} images to {labels_dir}")
    print(f"Positive images with at least one lesion: {positive_images}")
    print(f"Total lesion boxes written: {total_boxes}")
    print(f"Invalid boxes skipped: {len(invalid_rows)}")
    print(f"Wrote invalid box log to {invalid_output}")


if __name__ == "__main__":
    main()
