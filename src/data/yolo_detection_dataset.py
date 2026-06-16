from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import torch
from torch.utils.data import Dataset

from src.utils.bbox_utils import parse_yolo_line, yolo_to_voc
from src.utils.paths import project_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    label_path: Path


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def read_classes(classes_file: str | Path) -> list[str]:
    path = project_path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Classes file is empty: {path}")
    return names


class YoloDetectionDataset(Dataset):
    """Read images and YOLO txt labels as torchvision detection samples."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        transforms: Callable | None = None,
        allow_empty: bool = True,
    ) -> None:
        self.images_dir = project_path(images_dir)
        self.labels_dir = project_path(labels_dir)
        self.transforms = transforms
        self.allow_empty = allow_empty

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No image/label pairs found in {self.images_dir} and {self.labels_dir}")

    def _collect_samples(self) -> list[DetectionSample]:
        samples = []
        for image_path in sorted(self.images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(DetectionSample(image_path=image_path, label_path=label_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {sample.image_path}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        for line in sample.label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            class_id, x_center, y_center, box_width, box_height = parsed
            xmin, ymin, xmax, ymax = yolo_to_voc(x_center, y_center, box_width, box_height, width, height)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels.append(int(class_id))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        elif self.allow_empty:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            raise ValueError(f"No valid labels found for {sample.image_path}")

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
            "image_path": str(sample.image_path),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target
