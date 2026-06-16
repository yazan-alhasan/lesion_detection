from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.yolo_detection_dataset import YoloDetectionDataset, detection_collate_fn, read_classes
from src.utils.paths import ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained torchvision RetinaNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images_dir", default="data/processed/images/test")
    parser.add_argument("--labels_dir", default="data/processed/labels/test")
    parser.add_argument("--classes_file", default="data/processed/classes.txt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score_thresh", type=float, default=0.05)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/retinanet/evaluation_metrics.json")
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


def compute_ap(sorted_matches: list[int], total_gt: int) -> float:
    if total_gt <= 0 or not sorted_matches:
        return 0.0
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for matched in sorted_matches:
        if matched:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / total_gt)

    ap = 0.0
    for threshold in [i / 10 for i in range(11)]:
        candidates = [precision for precision, recall in zip(precisions, recalls) if recall >= threshold]
        ap += max(candidates) if candidates else 0.0
    return ap / 11.0


@torch.inference_mode()
def evaluate_model(model, loader, device: torch.device, iou_threshold: float = 0.5) -> dict:
    from torchvision.ops import box_iou

    model.eval()
    total_gt = 0
    total_predictions = 0
    total_tp = 0
    scored_matches: list[tuple[float, int]] = []

    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            target_boxes = target["boxes"].detach().cpu()
            target_labels = target["labels"].detach().cpu()
            matched_gt: set[int] = set()
            total_gt += int(target_boxes.shape[0])
            total_predictions += int(pred_boxes.shape[0])

            order = torch.argsort(pred_scores, descending=True)
            for pred_idx in order.tolist():
                score = float(pred_scores[pred_idx])
                same_class = torch.where(target_labels == pred_labels[pred_idx])[0]
                is_match = 0
                if same_class.numel() and target_boxes.numel():
                    ious = box_iou(pred_boxes[pred_idx].unsqueeze(0), target_boxes[same_class]).squeeze(0)
                    best_local = int(torch.argmax(ious).item())
                    best_iou = float(ious[best_local].item())
                    gt_idx = int(same_class[best_local].item())
                    if best_iou >= iou_threshold and gt_idx not in matched_gt:
                        matched_gt.add(gt_idx)
                        is_match = 1
                total_tp += is_match
                scored_matches.append((score, is_match))

    false_positives = total_predictions - total_tp
    false_negatives = total_gt - total_tp
    precision = total_tp / max(1, total_tp + false_positives)
    recall = total_tp / max(1, total_tp + false_negatives)
    sorted_matches = [match for _, match in sorted(scored_matches, key=lambda item: item[0], reverse=True)]
    map50 = compute_ap(sorted_matches, total_gt)
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": None,
        "true_positives": total_tp,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truth_boxes": total_gt,
        "predictions": total_predictions,
    }


def build_retinanet_for_eval(num_classes: int, imgsz: int, score_thresh: float, nms_thresh: float):
    from torchvision.models.detection import retinanet_resnet50_fpn

    return retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=imgsz,
        max_size=imgsz,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )


def main() -> None:
    args = parse_args()
    classes = read_classes(args.classes_file)
    device = resolve_device(args.device)
    dataset = YoloDetectionDataset(args.images_dir, args.labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
    )

    checkpoint = torch.load(project_path(args.checkpoint), map_location=device)
    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes and len(checkpoint_classes) != len(classes):
        raise ValueError(
            f"Checkpoint has {len(checkpoint_classes)} classes but {args.classes_file} has {len(classes)}."
        )
    model = build_retinanet_for_eval(len(classes), args.imgsz, args.score_thresh, args.nms_thresh).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(model, loader, device, iou_threshold=args.iou_threshold)
    metrics["classes"] = classes
    output = ensure_parent(args.output)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved RetinaNet evaluation metrics to {output}")


if __name__ == "__main__":
    main()
