from __future__ import annotations

from pathlib import Path

import cv2
import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-7)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def simple_precision_recall(
    predictions: list[dict],
    targets: list[dict],
    score_thresh: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict:
    tp = 0
    fp = 0
    fn = 0

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction["boxes"].detach().cpu()
        pred_scores = prediction["scores"].detach().cpu()
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]

        gt_boxes = target["boxes"].detach().cpu()
        matched_gt: set[int] = set()
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        for pred_idx in range(len(pred_boxes)):
            best_iou = 0.0
            best_gt = -1
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                current_iou = float(ious[pred_idx, gt_idx])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = gt_idx
            if best_gt >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    score_thresh: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict:
    model.eval()
    all_predictions = []
    all_targets = []
    map_metrics = {
        "map": 0.0,
        "map50": 0.0,
        "map75": 0.0,
        "ap_small": 0.0,
        "ap_medium": 0.0,
        "ap_large": 0.0,
        "ar_1": 0.0,
        "ar_10": 0.0,
        "ar_100": 0.0,
    }
    metric = None
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    except Exception as exc:
        print(f"[WARNING] torchmetrics mAP unavailable, using simple precision/recall only: {exc}")

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            predictions_cpu = [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
            targets_cpu = [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
            all_predictions.extend(predictions_cpu)
            all_targets.extend(targets_cpu)
            if metric is not None:
                metric.update(predictions_cpu, targets_cpu)

    if metric is not None:
        computed = metric.compute()
        map_metrics.update(
            {
                "map": float(computed.get("map", torch.tensor(0.0))),
                "map50": float(computed.get("map_50", torch.tensor(0.0))),
                "map75": float(computed.get("map_75", torch.tensor(0.0))),
                "ap_small": float(computed.get("map_small", torch.tensor(0.0))),
                "ap_medium": float(computed.get("map_medium", torch.tensor(0.0))),
                "ap_large": float(computed.get("map_large", torch.tensor(0.0))),
                "ar_1": float(computed.get("mar_1", torch.tensor(0.0))),
                "ar_10": float(computed.get("mar_10", torch.tensor(0.0))),
                "ar_100": float(computed.get("mar_100", torch.tensor(0.0))),
            }
        )

    simple = simple_precision_recall(all_predictions, all_targets, score_thresh, iou_threshold)
    gt_boxes = sum(int(len(target["boxes"])) for target in all_targets)
    pred_boxes = sum(int((prediction["scores"] >= score_thresh).sum()) for prediction in all_predictions)
    return {
        **map_metrics,
        **simple,
        "num_ground_truth_boxes": gt_boxes,
        "num_predictions": pred_boxes,
    }


def draw_prediction_image(image_path: str | Path, prediction: dict, target: dict, output_path: str | Path, score_thresh: float) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for box in target["boxes"].detach().cpu().numpy():
        xmin, ymin, xmax, ymax = [int(round(value)) for value in box]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(image, "gt", (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    for box, score in zip(boxes, scores):
        if float(score) < score_thresh:
            continue
        xmin, ymin, xmax, ymax = [int(round(value)) for value in box.tolist()]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"mass {float(score):.2f}",
            (xmin, max(15, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_prediction_txt(prediction: dict, output_path: str | Path, score_thresh: float) -> None:
    lines = []
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    for label, score, box in zip(labels, scores, boxes):
        if float(score) < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box.tolist()
        lines.append(f"{int(label)} {float(score):.6f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
