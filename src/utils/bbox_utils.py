from __future__ import annotations

import math
from typing import Iterable


BBOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")


def has_valid_bbox(row: dict | object) -> bool:
    values = []
    for column in BBOX_COLUMNS:
        value = row[column] if isinstance(row, dict) else getattr(row, column)
        values.append(value)
    if any(value is None for value in values):
        return False
    try:
        xmin, ymin, xmax, ymax = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    if any(math.isnan(value) for value in (xmin, ymin, xmax, ymax)):
        return False
    return xmax > xmin and ymax > ymin


def clip_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    xmin = max(0.0, min(float(width), float(xmin)))
    xmax = max(0.0, min(float(width), float(xmax)))
    ymin = max(0.0, min(float(height), float(ymin)))
    ymax = max(0.0, min(float(height), float(ymax)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def voc_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    clipped = clip_bbox(xmin, ymin, xmax, ymax, image_width, image_height)
    if clipped is None:
        return None
    xmin, ymin, xmax, ymax = clipped
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def yolo_to_voc(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    box_width = width * image_width
    box_height = height * image_height
    cx = x_center * image_width
    cy = y_center * image_height
    xmin = int(round(cx - box_width / 2.0))
    ymin = int(round(cy - box_height / 2.0))
    xmax = int(round(cx + box_width / 2.0))
    ymax = int(round(cy + box_height / 2.0))
    clipped = clip_bbox(xmin, ymin, xmax, ymax, image_width, image_height)
    if clipped is None:
        return 0, 0, 0, 0
    return tuple(int(round(value)) for value in clipped)


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = tuple(float(value) for value in parts[1:])
    except ValueError:
        return None
    return (class_id, *coords)


def valid_yolo_box(values: Iterable[float]) -> bool:
    try:
        x_center, y_center, width, height = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )
