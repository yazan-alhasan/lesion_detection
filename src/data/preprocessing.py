from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    image = image - float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if max_value > 0:
        image = image / max_value
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def apply_clahe_to_mammogram(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    image = ensure_uint8(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(gray)
