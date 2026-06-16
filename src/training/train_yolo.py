from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the processed VinDr-Mammo subset.")
    parser.add_argument("--data", default="configs/dataset_single_class.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--project", default="runs/yolo")
    parser.add_argument("--name", default="vindr_single_class_yolov8n_gpu_640")
    parser.add_argument("--device", default="0", help="Use auto, cpu, or a CUDA device index such as 0.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows for stability.")
    parser.add_argument("--single_cls", action="store_true", default=True, help="Train as single-class detection.")
    parser.add_argument("--exist_ok", action="store_true", help="Allow writing into an existing run name.")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision.")
    return parser.parse_args()


def metrics_to_dict(metrics) -> dict:
    box = getattr(metrics, "box", None)
    return {
        "precision": float(getattr(box, "mp", 0.0)) if box is not None else None,
        "recall": float(getattr(box, "mr", 0.0)) if box is not None else None,
        "map50": float(getattr(box, "map50", 0.0)) if box is not None else None,
        "map50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
    }


def print_device_status(selected_device: str) -> None:
    try:
        import torch
    except ImportError:
        print("Selected device:", selected_device)
        print("PyTorch is not installed.")
        return

    print("Selected device:", selected_device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("PyTorch CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU name: N/A")


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc

    selected_device = None if args.device == "auto" else args.device
    print_device_status(args.device)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(project_path(args.data)),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(project_path(args.project)),
        "name": args.name,
        "workers": args.workers,
        "single_cls": args.single_cls,
        "exist_ok": args.exist_ok,
        "amp": not args.no_amp,
    }
    if selected_device is not None:
        train_kwargs["device"] = selected_device

    try:
        model.train(**train_kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "bad allocation" in message or "out of memory" in message:
            raise SystemExit(
                "Training ran out of memory. Try a smaller command, for example:\n"
                "python src\\training\\train_yolo.py --data configs\\dataset_single_class.yaml "
                "--model yolov8n.pt --imgsz 640 --epochs 80 --batch 2 --project runs\\yolo "
                "--name vindr_single_class_yolov8n_gpu_640 --device 0 --workers 0 --single_cls\n"
                "Also close GPU-heavy apps such as browsers/game launchers if possible."
            ) from exc
        raise
    val_kwargs = {"data": str(project_path(args.data)), "imgsz": args.imgsz}
    if selected_device is not None:
        val_kwargs["device"] = selected_device
    metrics = model.val(**val_kwargs)
    summary = metrics_to_dict(metrics)

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / args.name))
    output = ensure_parent(save_dir / "metrics.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"mAP@0.5: {summary['map50']}")
    print(f"mAP@0.5:0.95: {summary['map50_95']}")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
