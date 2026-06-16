from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import ensure_parent, project_path


METRIC_COLUMNS = [
    "Model",
    "Image Size",
    "Epochs",
    "Precision",
    "Recall",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "Training Time",
    "Inference Time",
    "Notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare detector metrics.")
    parser.add_argument("--yolo_metrics", default="runs/yolo/vindr_subset_yolov8n/results.csv")
    parser.add_argument("--faster_rcnn_metrics", default="runs/faster_rcnn/metrics.json")
    parser.add_argument("--retinanet_metrics", default=None)
    parser.add_argument("--output_csv", default="runs/comparison_results.csv")
    parser.add_argument("--output_md", default="runs/comparison_results.md")
    return parser.parse_args()


def read_yolo(path: Path) -> dict:
    row = {column: "" for column in METRIC_COLUMNS}
    row["Model"] = "YOLOv8"
    row["Notes"] = "Subset run"
    if not path.exists():
        row["Notes"] = f"Missing metrics file: {path}"
        return row
    df = pd.read_csv(path)
    if df.empty:
        row["Notes"] = "Empty YOLO metrics file"
        return row
    last = df.iloc[-1]
    row["Epochs"] = int(last.get("epoch", len(df)))
    row["Precision"] = last.get("metrics/precision(B)", "")
    row["Recall"] = last.get("metrics/recall(B)", "")
    row["mAP@0.5"] = last.get("metrics/mAP50(B)", "")
    row["mAP@0.5:0.95"] = last.get("metrics/mAP50-95(B)", "")
    row["Training Time"] = last.get("time", "")
    return row


def read_faster_rcnn(path: Path) -> dict:
    row = {column: "" for column in METRIC_COLUMNS}
    row["Model"] = "Faster R-CNN"
    if not path.exists():
        row["Notes"] = f"Missing metrics file: {path}"
        return row
    data = json.loads(path.read_text(encoding="utf-8"))
    row["Precision"] = data.get("precision", "")
    row["Recall"] = data.get("recall", "")
    row["mAP@0.5"] = data.get("map50", "")
    row["mAP@0.5:0.95"] = data.get("map50_95", "")
    row["Training Time"] = data.get("training_time", "")
    row["Inference Time"] = data.get("inference_time", "")
    row["Notes"] = data.get("notes", data.get("status", ""))
    return row


def read_retinanet(path: Path) -> dict:
    row = {column: "" for column in METRIC_COLUMNS}
    row["Model"] = "RetinaNet"
    if not path.exists():
        row["Notes"] = f"Missing metrics file: {path}"
        return row
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data
    if isinstance(data.get("history"), list) and data["history"]:
        metrics = max(data["history"], key=lambda item: float(item.get("map50", 0.0) or 0.0))
        row["Epochs"] = metrics.get("epoch", len(data["history"]))
        row["Training Time"] = data.get("training_time", "")
    row["Precision"] = metrics.get("precision", "")
    row["Recall"] = metrics.get("recall", "")
    row["mAP@0.5"] = metrics.get("map50", "")
    row["mAP@0.5:0.95"] = metrics.get("map50_95", "")
    row["Notes"] = data.get("notes", "torchvision RetinaNet")
    return row


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.fillna("").to_numpy().tolist()]
    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(str(header)), *(len(row[idx]) for row in rows)))
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def main() -> None:
    args = parse_args()
    rows = [
        read_yolo(project_path(args.yolo_metrics)),
        read_faster_rcnn(project_path(args.faster_rcnn_metrics)),
    ]
    if args.retinanet_metrics:
        rows.append(read_retinanet(project_path(args.retinanet_metrics)))
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    output_csv = ensure_parent(args.output_csv)
    output_md = ensure_parent(args.output_md)
    df.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(df) + "\n", encoding="utf-8")
    print(f"Saved comparison CSV to {output_csv}")
    print(f"Saved comparison Markdown to {output_md}")


if __name__ == "__main__":
    main()
