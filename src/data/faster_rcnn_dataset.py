from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.bbox_utils import parse_yolo_line
from src.utils.paths import project_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FasterRCNNSample:
    image_path: Path
    label_path: Path


def collate_fn(batch):
    return tuple(zip(*batch))


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> list[float] | None:
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    xmin = x_center - width / 2.0
    ymin = y_center - height / 2.0
    xmax = x_center + width / 2.0
    ymax = y_center + height / 2.0

    xmin = max(0.0, min(float(xmin), float(img_w - 1)))
    ymin = max(0.0, min(float(ymin), float(img_h - 1)))
    xmax = max(0.0, min(float(xmax), float(img_w - 1)))
    ymax = max(0.0, min(float(ymax), float(img_h - 1)))

    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def read_classes(classes_file: str | Path) -> list[str]:
    path = project_path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"Classes file is empty: {path}")
    return classes


class YOLOToFasterRCNNDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        cache: str = "none",
        skip_empty: bool = True,
    ) -> None:
        self.data_root = project_path(data_root)
        self.split = split
        self.images_dir = self.data_root / "images" / split
        self.labels_dir = self.data_root / "labels" / split
        self.cache = cache
        self.skip_empty = skip_empty
        self.cached_samples: list[tuple[torch.Tensor, dict]] | None = None

        if cache not in {"none", "ram"}:
            raise ValueError("cache must be 'none' or 'ram'.")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No valid Faster R-CNN samples found in {self.images_dir}")

        if self.cache == "ram":
            self.cached_samples = []
            for idx in tqdm(range(len(self.samples)), desc=f"Caching {split} in RAM", unit="image"):
                self.cached_samples.append(self.load_sample(idx))
            print(f"Cached {len(self.cached_samples)} {split} images in CPU RAM.")

    def _collect_samples(self) -> list[FasterRCNNSample]:
        samples = []
        skipped_empty = 0
        for image_path in sorted(self.images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            if self.skip_empty and not label_path.read_text(encoding="utf-8").strip():
                skipped_empty += 1
                continue
            samples.append(FasterRCNNSample(image_path=image_path, label_path=label_path))
        if skipped_empty:
            print(f"[WARNING] Skipped {skipped_empty} empty-label images from {self.split}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.cached_samples is not None:
            image, target = self.cached_samples[index]
            return image.clone(), {key: value.clone() if torch.is_tensor(value) else value for key, value in target.items()}
        return self.load_sample(index)

    def load_sample(self, index: int):
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

        img_h, img_w = image.shape[:2]
        boxes = []
        labels = []
        invalid = 0
        for line in sample.label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_line(line)
            if parsed is None:
                invalid += 1
                continue
            class_id, x_center, y_center, box_width, box_height = parsed
            box = yolo_to_xyxy(x_center, y_center, box_width, box_height, img_w, img_h)
            if box is None:
                invalid += 1
                continue
            boxes.append(box)
            labels.append(int(class_id) + 1)

        if invalid:
            print(f"[WARNING] Invalid YOLO boxes skipped for {sample.label_path}: {invalid}")
        if not boxes and self.skip_empty:
            raise ValueError(f"No valid boxes found for positive sample: {sample.image_path}")

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes_tensor)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
            "image_path": str(sample.image_path),
        }
        return image_tensor, target
