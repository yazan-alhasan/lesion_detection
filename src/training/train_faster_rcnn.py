from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.faster_rcnn_dataset import YOLOToFasterRCNNDataset, collate_fn, read_classes
from src.evaluation.detection_metrics import draw_prediction_image, evaluate_model, write_prediction_txt
from src.utils.paths import ensure_dir, project_path


METRIC_COLUMNS = [
    "epoch",
    "train_total_loss",
    "train_loss_classifier",
    "train_loss_box_reg",
    "train_loss_objectness",
    "train_loss_rpn_box_reg",
    "val_map",
    "val_map50",
    "val_map75",
    "val_precision",
    "val_recall",
    "val_f1",
    "tp",
    "fp",
    "fn",
    "num_ground_truth_boxes",
    "num_predictions",
    "learning_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN from an existing YOLO-format dataset.")
    parser.add_argument("--data_root", default="data/dataset_mass_cropped")
    parser.add_argument("--model", choices=("fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn"), default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--save_period", type=int, default=10)
    parser.add_argument("--output_dir", default="runs/faster_rcnn/mass_fasterrcnn_resnet50")
    parser.add_argument("--score_thresh", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--early_stop_metric", choices=("map50", "map", "recall", "f1"), default="map50")
    parser.add_argument("--predict_every", type=int, default=0)
    parser.add_argument("--predict_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--resume")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_name.isdigit():
        device_name = f"cuda:{device_name}"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def build_faster_rcnn_model(model_name: str, num_classes: int, pretrained: bool = True):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if model_name == "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None if not pretrained else None)
    elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None if not pretrained else None)
    else:
        raise ValueError(f"Unsupported Faster R-CNN model: {model_name}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def move_targets_to_device(targets, device: torch.device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(model, loader, optimizer, scaler, device: torch.device, epoch: int, epochs: int, use_amp: bool) -> dict:
    model.train()
    totals = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total_loss": 0.0,
    }
    batches = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        if scaler is not None and use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        batches += 1
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            totals[key] += float(loss_dict.get(key, torch.tensor(0.0)).detach().cpu())
        totals["total_loss"] += float(total_loss.detach().cpu())
        progress.set_postfix(loss=f"{totals['total_loss'] / batches:.4f}")

    return {key: value / max(1, batches) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def append_metrics(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def selected_prediction_indices(dataset, count: int, seed: int) -> list[int]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[: min(count, len(indices))]


def save_predictions(model, dataset, indices: list[int], device: torch.device, output_dir: Path, score_thresh: float) -> None:
    model.eval()
    images_dir = output_dir / "images"
    txt_dir = output_dir / "predictions_txt"
    with torch.no_grad():
        for index in indices:
            image, target = dataset[index]
            prediction = model([image.to(device)])[0]
            prediction_cpu = {key: value.detach().cpu() for key, value in prediction.items()}
            sample = dataset.samples[index]
            draw_prediction_image(sample.image_path, prediction_cpu, target, images_dir / f"{sample.image_path.stem}.png", score_thresh)
            write_prediction_txt(prediction_cpu, txt_dir / f"{sample.image_path.stem}.txt", score_thresh)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = project_path(args.data_root)
    output_dir = ensure_dir(args.output_dir)
    weights_dir = ensure_dir(output_dir / "weights")
    metrics_path = output_dir / "metrics.csv"
    classes_file = data_root / "classes.txt"
    classes = read_classes(classes_file)
    num_classes = len(classes) + 1
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    print(f"Data root: {data_root}")
    print(f"Classes: background + {classes}")
    print(f"Model classes: {num_classes}")
    print(f"Selected device: {device}")
    print(f"RAM cache: {args.cache}")
    print(f"AMP enabled: {use_amp}")

    train_dataset = YOLOToFasterRCNNDataset(data_root, "train", cache=args.cache)
    val_dataset = YOLOToFasterRCNNDataset(data_root, "val", cache=args.cache)
    test_dataset = YOLOToFasterRCNNDataset(data_root, "test", cache="none")
    train_loader = make_loader(train_dataset, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_dataset, 1, args.workers, shuffle=False)

    model = build_faster_rcnn_model(args.model, num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(project_path(args.resume), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))

    fixed_prediction_indices = selected_prediction_indices(test_dataset, args.predict_samples, args.seed)
    epochs_without_improvement = 0
    history = []
    started = time.time()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs, use_amp)
        scheduler.step()
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            score_thresh=args.score_thresh,
            iou_threshold=args.iou_threshold,
        )
        learning_rate = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_total_loss": train_losses["total_loss"],
            "train_loss_classifier": train_losses["loss_classifier"],
            "train_loss_box_reg": train_losses["loss_box_reg"],
            "train_loss_objectness": train_losses["loss_objectness"],
            "train_loss_rpn_box_reg": train_losses["loss_rpn_box_reg"],
            "val_map": val_metrics["map"],
            "val_map50": val_metrics["map50"],
            "val_map75": val_metrics["map75"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "tp": val_metrics["tp"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
            "num_ground_truth_boxes": val_metrics["num_ground_truth_boxes"],
            "num_predictions": val_metrics["num_predictions"],
            "learning_rate": learning_rate,
        }
        history.append(row)
        append_metrics(metrics_path, row)

        current_metric = float(val_metrics[args.early_stop_metric])
        improved = current_metric > best_metric + args.min_delta
        save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric, args)
        if improved:
            best_metric = current_metric
            epochs_without_improvement = 0
            save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric, args)
        else:
            epochs_without_improvement += 1

        if args.save_period > 0 and epoch % args.save_period == 0:
            save_checkpoint(weights_dir / f"epoch_{epoch}.pt", model, optimizer, scheduler, epoch, best_metric, args)

        if args.predict_every > 0 and epoch % args.predict_every == 0:
            save_predictions(
                model,
                test_dataset,
                fixed_prediction_indices,
                device,
                output_dir / "epoch_predictions" / f"epoch_{epoch}",
                args.score_thresh,
            )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_losses['total_loss']:.4f} "
            f"map50={val_metrics['map50']:.4f} "
            f"map={val_metrics['map']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"best_{args.early_stop_metric}={best_metric:.4f} "
            f"stale={epochs_without_improvement}/{args.patience}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            stopped_early = True
            break

    best_path = weights_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    save_predictions(model, test_dataset, list(range(len(test_dataset))), device, output_dir / "predictions" / "test", args.score_thresh)
    test_loader = make_loader(test_dataset, 1, args.workers, shuffle=False)
    test_metrics = evaluate_model(model, test_loader, device, score_thresh=args.score_thresh, iou_threshold=args.iou_threshold)

    summary = {
        "args": vars(args),
        "classes": ["background", *classes],
        "best_metric": best_metric,
        "early_stop_metric": args.early_stop_metric,
        "stopped_early": stopped_early,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "training_time_seconds": time.time() - started,
        "test_metrics": test_metrics,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoints to {weights_dir}")
    print(f"Saved test predictions to {output_dir / 'predictions' / 'test'}")


if __name__ == "__main__":
    main()
