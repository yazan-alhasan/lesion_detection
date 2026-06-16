from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.yolo_detection_dataset import YoloDetectionDataset, detection_collate_fn, read_classes
from src.evaluation.evaluate_retinanet import evaluate_model
from src.utils.paths import ensure_dir, ensure_parent, project_path


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def config_value(args: argparse.Namespace, config: dict, name: str, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train torchvision RetinaNet on processed VinDr-Mammo labels.")
    parser.add_argument("--config", default="configs/train_retinanet.yaml")
    parser.add_argument("--images_root")
    parser.add_argument("--labels_root")
    parser.add_argument("--classes_file")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pretrained", action="store_true", default=None)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--score_thresh", type=float)
    parser.add_argument("--nms_thresh", type=float)
    parser.add_argument("--iou_threshold", type=float)
    parser.add_argument("--exist_ok", action="store_true")
    return parser.parse_args()


def resolve_device(selected: str) -> torch.device:
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    elif selected.isdigit():
        selected = f"cuda:{selected}"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def build_retinanet(num_classes: int, pretrained: bool, imgsz: int, score_thresh: float, nms_thresh: float):
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn(
        weights=weights,
        weights_backbone=None if pretrained else None,
        num_classes=91 if pretrained else num_classes,
        min_size=imgsz,
        max_size=imgsz,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )
    head = model.head.classification_head
    num_anchors = head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(256, num_anchors, num_classes)
    return model


def make_loader(images_root: Path, labels_root: Path, split: str, batch_size: int, workers: int, shuffle: bool):
    dataset = YoloDetectionDataset(images_root / split, labels_root / split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=detection_collate_fn,
    )


def train_one_epoch(model, loader, optimizer, device: torch.device, epoch: int, epochs: int) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets
        ]
        loss_dict = model(images, targets)
        loss = sum(value for value in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        progress.set_postfix(loss=f"{total_loss / batches:.4f}")
    return total_loss / max(1, batches)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, classes: list[str]) -> None:
    ensure_parent(path)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "classes": classes,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    images_root = project_path(config_value(args, config, "images_root", "data/processed/images"))
    labels_root = project_path(config_value(args, config, "labels_root", "data/processed/labels"))
    classes_file = config_value(args, config, "classes_file", "data/processed/classes.txt")
    imgsz = int(config_value(args, config, "imgsz", 640))
    epochs = int(config_value(args, config, "epochs", 40))
    batch = int(config_value(args, config, "batch", 2))
    lr = float(config_value(args, config, "lr", 0.0001))
    weight_decay = float(config_value(args, config, "weight_decay", 0.0001))
    project = project_path(config_value(args, config, "project", "runs/retinanet"))
    name = str(config_value(args, config, "name", "vindr_single_class_retinanet_resnet50"))
    device_name = str(config_value(args, config, "device", "auto"))
    workers = int(config_value(args, config, "workers", 0))
    pretrained = bool(config_value(args, config, "pretrained", True))
    score_thresh = float(config_value(args, config, "score_thresh", 0.05))
    nms_thresh = float(config_value(args, config, "nms_thresh", 0.5))
    iou_threshold = float(config_value(args, config, "iou_threshold", 0.5))

    run_dir = ensure_dir(project / name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.json"
    if any(run_dir.iterdir()) and not args.exist_ok:
        print(f"Run directory already exists: {run_dir}")
        print("Continuing because generated metrics/checkpoints can be overwritten with explicit paths.")

    classes = read_classes(classes_file)
    device = resolve_device(device_name)
    print(f"Selected device: {device}")
    print(f"Classes: {classes}")

    train_loader = make_loader(images_root, labels_root, "train", batch, workers, shuffle=True)
    val_loader = make_loader(images_root, labels_root, "val", batch, workers, shuffle=False)

    model = build_retinanet(len(classes), pretrained, imgsz, score_thresh, nms_thresh).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    start_epoch = 1
    best_map50 = -1.0
    history = []
    if args.resume:
        checkpoint = torch.load(project_path(args.resume), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_map50 = float(checkpoint.get("metrics", {}).get("map50", -1.0) or -1.0)

    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, epochs)
        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, iou_threshold=iou_threshold)
        metrics.update({"epoch": epoch, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0]})
        history.append(metrics)

        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, metrics, classes)
        if float(metrics.get("map50", 0.0) or 0.0) >= best_map50:
            best_map50 = float(metrics.get("map50", 0.0) or 0.0)
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, metrics, classes)

        metrics_path.write_text(json.dumps({"classes": classes, "history": history, "best_map50": best_map50}, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch}/{epochs} "
            f"loss={train_loss:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"map50={metrics['map50']:.4f}"
        )

    total_time = time.time() - started
    final = {"classes": classes, "history": history, "best_map50": best_map50, "training_time": total_time}
    metrics_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(f"Saved RetinaNet metrics to {metrics_path}")
    print(f"Saved checkpoints to {checkpoints_dir}")


if __name__ == "__main__":
    main()
