from __future__ import annotations

import cv2
import numpy as np


MLO_TOKEN = "MLO"


def is_mlo_view(view_position: object) -> bool:
    if view_position is None:
        return False
    return MLO_TOKEN in str(view_position).strip().upper()


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0].copy()
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype == np.uint8:
        return gray
    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not finite.any():
        return np.zeros(gray.shape, dtype=np.uint8)
    minimum = float(gray[finite].min())
    maximum = float(gray[finite].max())
    gray = np.nan_to_num(gray, nan=minimum, posinf=maximum, neginf=minimum)
    if maximum <= minimum:
        return np.zeros(gray.shape, dtype=np.uint8)
    return ((gray - minimum) * (255.0 / (maximum - minimum))).clip(0, 255).astype(np.uint8)


def bright_pixel_ratio(region: np.ndarray, percentile: float = 85) -> float:
    nonzero = region[region > 0]
    if nonzero.size == 0:
        return 0.0
    threshold = float(np.percentile(nonzero, percentile))
    return float(np.count_nonzero(region >= threshold)) / float(region.size)


def _candidate_score(region: np.ndarray, percentile: float) -> float:
    nonzero = region[region > 0]
    if nonzero.size == 0:
        return 0.0
    threshold = float(np.percentile(nonzero, percentile))
    bright = region[region >= threshold]
    return bright_pixel_ratio(region, percentile) * (float(bright.mean()) / 255.0)


def choose_pectoral_side(
    gray: np.ndarray,
    percentile: float = 90,
    max_width_ratio: float = 0.35,
    max_height_ratio: float = 0.50,
    laterality: object = None,
) -> str:
    height, width = gray.shape[:2]
    y2 = max(1, int(round(height * max_height_ratio)))
    side_width = max(1, int(round(width * max_width_ratio)))
    left_score = _candidate_score(gray[:y2, :side_width], percentile)
    right_score = _candidate_score(gray[:y2, width - side_width :], percentile)
    if np.isclose(left_score, right_score):
        hint = str(laterality or "").strip().upper()
        if hint in {"L", "LEFT"}:
            return "left"
        if hint in {"R", "RIGHT"}:
            return "right"
    return "left" if left_score >= right_score else "right"


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == label).astype(np.uint8) * 255)


def detect_pectoral_mask(
    image: np.ndarray,
    view_position: object,
    laterality: object = None,
    intensity_percentile: float = 90,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.20,
    max_width_ratio: float = 0.35,
    max_height_ratio: float = 0.50,
    max_y_ratio: float = 0.60,
) -> tuple[np.ndarray, dict]:
    gray = to_uint8_gray(image)
    empty = np.zeros_like(gray, dtype=np.uint8)
    if not is_mlo_view(view_position):
        return empty, {"applied": False, "reason": "not_mlo_view", "mask_area": 0, "mask_area_ratio": 0.0}
    if not 0 <= intensity_percentile <= 100:
        raise ValueError("intensity_percentile must be between 0 and 100")
    if not 0 < max_width_ratio <= 0.5 or not 0 < max_height_ratio <= 1:
        raise ValueError("candidate region ratios are outside their valid ranges")
    if not 0 <= min_area_ratio < max_area_ratio <= 1:
        raise ValueError("mask area ratios are outside their valid ranges")
    if not 0 < max_y_ratio <= 1:
        raise ValueError("max_y_ratio must be between 0 and 1")

    height, width = gray.shape
    side = choose_pectoral_side(
        gray, intensity_percentile, max_width_ratio, max_height_ratio, laterality
    )
    side_width = max(1, int(round(width * max_width_ratio)))
    y2 = max(1, int(round(height * max_height_ratio)))
    x1, x2 = (0, side_width) if side == "left" else (width - side_width, width)
    candidate = gray[:y2, x1:x2]
    nonzero = candidate[candidate > 0]
    base = {"side": side, "mask_area": 0, "mask_area_ratio": 0.0, "mask_max_y_ratio": 0.0}
    if nonzero.size == 0:
        return empty, {"applied": False, "reason": "empty_candidate_region", **base}

    threshold = float(np.percentile(nonzero, intensity_percentile))
    candidate_mask = ((candidate >= threshold).astype(np.uint8) * 255)
    kernel = np.ones((7, 7), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = largest_component(candidate_mask)
    full_mask = empty.copy()
    full_mask[:y2, x1:x2] = candidate_mask
    full_mask = cv2.dilate(full_mask, np.ones((9, 9), np.uint8), iterations=1)

    area = int(np.count_nonzero(full_mask))
    ratio = area / float(height * width)
    mask_ys = np.where(full_mask > 0)[0]
    mask_max_y_ratio = float(mask_ys.max()) / float(height) if mask_ys.size else 0.0
    base.update({
        "mask_area": area,
        "mask_area_ratio": ratio,
        "mask_max_y_ratio": mask_max_y_ratio,
    })
    if ratio < min_area_ratio:
        return empty, {"applied": False, "reason": "mask_too_small", **base}
    if ratio > max_area_ratio:
        return empty, {"applied": False, "reason": "mask_too_large", **base}
    if mask_ys.size == 0:
        return empty, {"applied": False, "reason": "empty_candidate_region", **base}
    if mask_max_y_ratio > max_y_ratio:
        return empty, {"applied": False, "reason": "mask_extends_too_low", **base}
    return full_mask, {"applied": True, "reason": "success", **base}


def apply_pectoral_attenuation(
    gray: np.ndarray, mask: np.ndarray, attenuation_factor: float = 0.5
) -> np.ndarray:
    if not 0 <= attenuation_factor <= 1:
        raise ValueError("attenuation_factor must be between 0 and 1")
    output = to_uint8_gray(gray)
    region = mask > 0
    output[region] = (
        output[region].astype(np.float32) * attenuation_factor
    ).clip(0, 255).astype(np.uint8)
    return output


def suppress_pectoral_muscle(
    image: np.ndarray,
    view_position: object,
    laterality: object = None,
    method: str = "threshold_triangle",
    suppress_mode: str = "attenuate",
    attenuation_factor: float = 0.5,
    intensity_percentile: float = 90,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.20,
    max_width_ratio: float = 0.35,
    max_height_ratio: float = 0.50,
    max_y_ratio: float = 0.60,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    del debug  # Debug artifacts are written by the dataset pipeline.
    if method != "threshold_triangle":
        raise ValueError(f"Unknown pectoral method: {method}")
    if suppress_mode not in {"blackout", "inpaint", "attenuate"}:
        raise ValueError(f"Unknown suppress_mode: {suppress_mode}")

    gray = to_uint8_gray(image)
    mask, info = detect_pectoral_mask(
        gray,
        view_position,
        laterality=laterality,
        intensity_percentile=intensity_percentile,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio,
        max_y_ratio=max_y_ratio,
    )
    info.update(
        view_position="" if view_position is None else str(view_position),
        laterality="" if laterality is None else str(laterality),
        method=method,
        suppress_mode=suppress_mode,
        attenuation_factor=float(attenuation_factor),
    )
    if not info["applied"]:
        return gray, mask, info

    if suppress_mode == "blackout":
        suppressed = gray.copy()
        suppressed[mask > 0] = 0
    elif suppress_mode == "attenuate":
        suppressed = apply_pectoral_attenuation(gray, mask, attenuation_factor)
    else:
        try:
            suppressed = cv2.inpaint(gray, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        except cv2.error:
            suppressed = gray.copy()
            info["applied"] = False
            info["reason"] = "inpaint_failed"

    if suppressed.shape[:2] != gray.shape[:2]:
        raise AssertionError("Pectoral suppression changed image size.")
    return suppressed, mask, info
