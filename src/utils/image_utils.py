from __future__ import annotations

import cv2
import numpy as np
import pydicom


def load_dicom_pixels(path: str):
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels, dataset


def normalize_to_uint8(pixels: np.ndarray) -> np.ndarray:
    pixels = pixels.astype(np.float32)
    min_value = float(np.nanmin(pixels))
    max_value = float(np.nanmax(pixels))
    if max_value <= min_value:
        return np.zeros(pixels.shape, dtype=np.uint8)
    pixels = (pixels - min_value) / (max_value - min_value)
    return np.clip(pixels * 255.0, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return clahe.apply(image)


def resize_long_side(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    if width >= height:
        new_width = target_size
        new_height = max(1, int(round(height * target_size / width)))
    else:
        new_height = target_size
        new_width = max(1, int(round(width * target_size / height)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, new_width / width, new_height / height
