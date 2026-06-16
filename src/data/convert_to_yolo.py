from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import BBOX_COLUMNS, has_valid_bbox, voc_to_yolo
from src.utils.paths import ensure_dir, ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VinDr-Mammo boxes to YOLO labels.")
    parser.add_argument("--finding_csv", required=True)
    parser.add_argument("--metadata_csv", default="data/processed/image_metadata.csv")
    parser.add_argument("--labels_output_dir", default="data/processed/labels/all")
    parser.add_argument("--classes_output", default="data/processed/classes.txt")
    parser.add_argument("--mapping_output", default="data/processed/class_mapping.json")
    parser.add_argument("--dataset_yaml", default="configs/dataset.yaml")
    return parser.parse_args()


def parse_categories(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip().strip("'\"") for part in text.replace(";", ",").split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    findings = pd.read_csv(project_path(args.finding_csv))
    metadata = pd.read_csv(project_path(args.metadata_csv))
    labels_dir = ensure_dir(args.labels_output_dir)

    metadata["image_id"] = metadata["image_id"].astype(str)
    findings["image_id"] = findings["image_id"].astype(str)
    converted_ids = set(metadata["image_id"])
    findings = findings[findings["image_id"].isin(converted_ids)].copy()
    findings["category_list"] = findings["finding_categories"].apply(parse_categories)

    positive_rows = []
    for _, row in findings.iterrows():
        if not has_valid_bbox(row):
            continue
        categories = [cat for cat in row["category_list"] if cat.lower() != "no finding"]
        if not categories:
            continue
        row = row.copy()
        row["primary_category"] = categories[0]
        positive_rows.append(row)

    classes = sorted({row["primary_category"] for row in positive_rows})
    class_mapping = {name: idx for idx, name in enumerate(classes)}

    for image_id in converted_ids:
        (labels_dir / f"{image_id}.txt").write_text("", encoding="utf-8")

    meta_by_id = metadata.set_index("image_id").to_dict(orient="index")
    grouped_lines: dict[str, list[str]] = {image_id: [] for image_id in converted_ids}

    for row in positive_rows:
        image_id = str(row["image_id"])
        meta = meta_by_id[image_id]
        scaled = {
            "xmin": float(row["xmin"]) * float(meta["scale_x"]),
            "ymin": float(row["ymin"]) * float(meta["scale_y"]),
            "xmax": float(row["xmax"]) * float(meta["scale_x"]),
            "ymax": float(row["ymax"]) * float(meta["scale_y"]),
        }
        yolo = voc_to_yolo(
            scaled["xmin"],
            scaled["ymin"],
            scaled["xmax"],
            scaled["ymax"],
            float(meta["processed_width"]),
            float(meta["processed_height"]),
        )
        if yolo is None:
            continue
        class_id = class_mapping[row["primary_category"]]
        grouped_lines[image_id].append(
            f"{class_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}"
        )

    for image_id, lines in grouped_lines.items():
        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    classes_output = ensure_parent(args.classes_output)
    classes_output.write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")
    mapping_output = ensure_parent(args.mapping_output)
    mapping_output.write_text(json.dumps(class_mapping, indent=2) + "\n", encoding="utf-8")

    dataset_yaml = ensure_parent(args.dataset_yaml)
    dataset = {
        "path": "data/processed",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for name, idx in class_mapping.items()},
    }
    dataset_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    print(f"Wrote labels for {len(converted_ids)} images to {labels_dir}")
    print(f"Wrote {len(classes)} classes to {classes_output}")
    print(f"Wrote dataset YAML to {dataset_yaml}")


if __name__ == "__main__":
    main()
