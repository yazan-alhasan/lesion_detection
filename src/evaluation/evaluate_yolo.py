from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.train_yolo import metrics_to_dict
from src.utils.paths import ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output", default="runs/yolo/evaluation_metrics.json")
    parser.add_argument("--device", default="auto", help="Use auto, cpu, or a CUDA device index such as 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc

    selected_device = None if args.device == "auto" else args.device
    model = YOLO(str(project_path(args.weights)))
    val_kwargs = {"data": str(project_path(args.data)), "imgsz": args.imgsz}
    if selected_device is not None:
        val_kwargs["device"] = selected_device
    metrics = model.val(**val_kwargs)
    summary = metrics_to_dict(metrics)
    output = ensure_parent(args.output)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved YOLO evaluation metrics to {output}")


if __name__ == "__main__":
    main()
