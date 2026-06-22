from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import apply_clahe_to_mammogram
from src.data.pectoral_suppression import is_mlo_view, suppress_pectoral_muscle
from src.utils.bbox_utils import BBOX_COLUMNS, has_valid_bbox, valid_yolo_box, voc_to_yolo
from src.utils.dicom_index import build_dicom_index
from src.utils.image_utils import load_dicom_pixels, normalize_to_uint8, resize_long_side
from src.utils.paths import PROJECT_ROOT, ensure_dir, ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a selected-class VinDr-Mammo YOLO dataset with optional breast cropping."
    )
    parser.add_argument("--annotations", "--finding_csv", dest="annotations", required=True)
    parser.add_argument(
        "--images_root",
        "--input_images_path",
        "--images_input_path",
        dest="images_root",
        default="data/raw/images",
        help="Folder containing DICOM files. Can be any absolute path or a project-relative path.",
    )
    parser.add_argument("--output_dir", default="data/processed_selected")
    parser.add_argument(
        "--target_finding",
        help="Single category mode, for example: Mass. The YOLO class name defaults to this value in lowercase.",
    )
    parser.add_argument(
        "--target_findings",
        nargs="+",
        help="Multi-category mode, for example: Mass 'Suspicious Calcification'. All selected boxes become one class.",
    )
    parser.add_argument(
        "--target_class_name",
        help="Output class name. Defaults to the single target name, or 'lesion' for multi-category mode.",
    )
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--apply_clahe", action="store_true")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid_size", type=int, default=8)
    parser.add_argument("--suppress_pectoral", action="store_true")
    parser.add_argument("--pectoral_method", choices=("threshold_triangle",), default="threshold_triangle")
    parser.add_argument("--pectoral_intensity_percentile", type=float, default=90)
    parser.add_argument("--pectoral_min_area_ratio", type=float, default=0.005)
    parser.add_argument("--pectoral_max_area_ratio", type=float, default=0.20)
    parser.add_argument("--pectoral_max_width_ratio", type=float, default=0.35)
    parser.add_argument("--pectoral_max_height_ratio", type=float, default=0.50)
    parser.add_argument("--pectoral_max_y_ratio", type=float, default=0.60)
    parser.add_argument("--pectoral_suppress_mode", choices=("inpaint", "blackout", "attenuate"), default="attenuate")
    parser.add_argument("--pectoral_attenuation_factor", type=float, default=0.5)
    parser.add_argument("--pectoral_order", choices=("before_clahe", "after_clahe"), default="before_clahe")
    parser.add_argument("--pectoral_debug", action="store_true")
    parser.add_argument("--crop_breast", action="store_true", default=True)
    parser.add_argument("--no_crop_breast", action="store_false", dest="crop_breast")
    parser.add_argument("--crop_padding", type=int, default=10)
    parser.add_argument(
        "--train_ratio",
        "--train_percent",
        dest="train_ratio",
        type=float,
        default=0.8,
        help="Train split share. Accepts ratios like 0.8 or percentages like 80.",
    )
    parser.add_argument(
        "--val_ratio",
        "--val_percent",
        dest="val_ratio",
        type=float,
        default=0.1,
        help="Validation split share. Accepts ratios like 0.1 or percentages like 10.",
    )
    parser.add_argument(
        "--test_ratio",
        "--test_percent",
        dest="test_ratio",
        type=float,
        default=0.1,
        help="Test split share. Accepts ratios like 0.1 or percentages like 10.",
    )
    parser.add_argument(
        "--split_by",
        choices=("exam", "study_id"),
        default="exam",
        help="Group train/val/test by actual DICOM exam folder or by annotation study_id.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--save_debug_samples", action="store_true")
    parser.add_argument("--num_debug_samples", type=int, default=50)
    parser.add_argument("--metadata_output", default=None)
    parser.add_argument("--invalid_boxes_output", default=None)
    parser.add_argument("--missing_images_output", default=None)
    return parser.parse_args()


def parse_categories(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip().strip("'\"") for part in text.replace(";", ",").split(",") if part.strip()]


def normalize_category(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def row_matches_targets(row: pd.Series, targets: set[str]) -> bool:
    categories = row.get("category_list", [])
    normalized_categories = {normalize_category(category) for category in categories}
    if normalized_categories & targets:
        return True

    raw_text = normalize_category(row.get("finding_categories", ""))
    return any(target in raw_text for target in targets)


def crop_breast_region(image: np.ndarray, padding: int = 10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_height, image_width = gray.shape[:2]
    if num_labels <= 1:
        return image, (0, 0, image_width, image_height)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))
    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + width + padding)
    y2 = min(image_height, y + height + padding)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def crop_box_including_targets(
    image: np.ndarray,
    scaled_boxes: list[tuple[float, float, float, float]],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    if not scaled_boxes:
        image_height, image_width = image.shape[:2]
        return image, (0, 0, image_width, image_height), False

    image_height, image_width = image.shape[:2]
    try:
        _, breast_crop = crop_breast_region(image, padding)
    except Exception:
        return image, (0, 0, image_width, image_height), False

    min_x = min(box[0] for box in scaled_boxes)
    min_y = min(box[1] for box in scaled_boxes)
    max_x = max(box[2] for box in scaled_boxes)
    max_y = max(box[3] for box in scaled_boxes)

    x1 = int(max(0, min(breast_crop[0], min_x) - padding))
    y1 = int(max(0, min(breast_crop[1], min_y) - padding))
    x2 = int(min(image_width, max(breast_crop[2], max_x) + padding))
    y2 = int(min(image_height, max(breast_crop[3], max_y) + padding))

    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, image_width, image_height), False

    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2), True


def split_by_group(
    images: pd.DataFrame,
    group_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict[str, set[str]]:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    val_ratio = val_ratio / total
    test_ratio = test_ratio / total
    temp_ratio = val_ratio + test_ratio
    groups = images[group_column].astype(str).drop_duplicates()

    if len(groups) < 3 or temp_ratio == 0:
        return {"train": set(groups), "val": set(), "test": set()}

    train_groups, temp_groups = train_test_split(groups, test_size=temp_ratio, random_state=random_seed)
    if len(temp_groups) < 2 or test_ratio == 0:
        val_groups = temp_groups
        test_groups = []
    elif val_ratio == 0:
        val_groups = []
        test_groups = temp_groups
    else:
        relative_test_ratio = test_ratio / temp_ratio
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=relative_test_ratio,
            random_state=random_seed,
        )

    return {
        "train": set(str(value) for value in train_groups),
        "val": set(str(value) for value in val_groups),
        "test": set(str(value) for value in test_groups),
    }


def split_for_group(group_id: str, split_groups: dict[str, set[str]]) -> str:
    group_id = str(group_id)
    for split_name, groups in split_groups.items():
        if group_id in groups:
            return split_name
    return "train"


def normalize_split_shares(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    shares = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(value < 0 for value in shares):
        raise ValueError("Split shares must be non-negative.")
    total = sum(shares)
    if total <= 0:
        raise ValueError("At least one split share must be positive.")
    return tuple(value / total for value in shares)


def draw_debug_sample(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    output_path: Path,
    class_name: str,
) -> None:
    if image.ndim == 2:
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug_image = image.copy()

    for xmin, ymin, xmax, ymax in boxes:
        pt1 = (int(round(xmin)), int(round(ymin)))
        pt2 = (int(round(xmax)), int(round(ymax)))
        cv2.rectangle(debug_image, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            class_name,
            (pt1[0], max(15, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug_image)


def draw_debug_comparison(
    before_image: np.ndarray,
    after_image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    output_path: Path,
    class_name: str,
) -> None:
    before_debug = draw_boxes_to_image(before_image, boxes, class_name)
    after_debug = draw_boxes_to_image(after_image, boxes, class_name)
    if before_debug.shape[:2] != after_debug.shape[:2]:
        after_debug = cv2.resize(after_debug, (before_debug.shape[1], before_debug.shape[0]), interpolation=cv2.INTER_AREA)
    comparison = np.concatenate([before_debug, after_debug], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)


def draw_boxes_to_image(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    class_name: str,
) -> np.ndarray:
    if image.ndim == 2:
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug_image = image.copy()

    for xmin, ymin, xmax, ymax in boxes:
        pt1 = (int(round(xmin)), int(round(ymin)))
        pt2 = (int(round(xmax)), int(round(ymax)))
        cv2.rectangle(debug_image, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            class_name,
            (pt1[0], max(15, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return debug_image


def add_debug_metadata(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for index, line in enumerate(lines):
        cv2.putText(
            output, line, (8, 20 + index * 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def dataset_yaml_path(output_dir: Path) -> str:
    try:
        return output_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return output_dir.as_posix()


def choose_targets(args: argparse.Namespace) -> tuple[list[str], str, str]:
    if bool(args.target_finding) == bool(args.target_findings):
        raise ValueError("Use exactly one mode: --target_finding for one category, or --target_findings for many.")

    if args.target_finding:
        targets = [args.target_finding]
        class_name = args.target_class_name or normalize_category(args.target_finding)
        mode = "single_category"
    else:
        targets = list(args.target_findings)
        if len(targets) < 2:
            raise ValueError("--target_findings is for multi-category mode. Use --target_finding for one category.")
        class_name = args.target_class_name or "lesion"
        mode = "multi_category_as_lesion"

    return targets, class_name, mode


def main() -> None:
    args = parse_args()
    targets, class_name, mode = choose_targets(args)
    train_share, val_share, test_share = normalize_split_shares(args.train_ratio, args.val_ratio, args.test_ratio)
    target_set = {normalize_category(target) for target in targets}
    output_dir = ensure_dir(args.output_dir)
    metadata_output = ensure_parent(args.metadata_output or output_dir / "image_metadata.csv")
    invalid_boxes_output = ensure_parent(args.invalid_boxes_output or output_dir / "invalid_boxes.csv")
    missing_images_output = ensure_parent(args.missing_images_output or output_dir / "missing_images.csv")

    annotations = pd.read_csv(project_path(args.annotations))
    required_columns = {"study_id", "image_id", "finding_categories", *BBOX_COLUMNS}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Annotation CSV is missing required columns: {sorted(missing_columns)}")

    annotations = annotations.copy()
    annotations["image_id"] = annotations["image_id"].astype(str)
    annotations["study_id"] = annotations["study_id"].astype(str)
    annotations["category_list"] = annotations["finding_categories"].apply(parse_categories)

    target_rows = annotations[
        annotations.apply(lambda row: row_matches_targets(row, target_set) and has_valid_bbox(row), axis=1)
    ].copy()
    selected_columns = [
        column
        for column in ("study_id", "series_id", "image_id", "view_position", "laterality")
        if column in target_rows.columns
    ]
    selected_images = (
        target_rows[selected_columns]
        .drop_duplicates()
        .sort_values(["study_id", "image_id"], kind="stable")
        .reset_index(drop=True)
    )

    for split_name in ("train", "val", "test"):
        ensure_dir(output_dir / "images" / split_name)
        ensure_dir(output_dir / "labels" / split_name)

    print(f"Selected target categories: {', '.join(targets)}")
    print(f"Output class name: {class_name}")
    print(f"Split shares: train={train_share:.3f}, val={val_share:.3f}, test={test_share:.3f}")
    print(f"Split grouping: {args.split_by}")
    print(f"Selected image IDs before conversion: {len(selected_images)}")
    print("Indexing DICOM files...")
    dicom_index = build_dicom_index(args.images_root)
    images_root = project_path(args.images_root)
    missing_rows = []
    available_rows = []
    for _, image_row in selected_images.iterrows():
        image_id = str(image_row["image_id"])
        source = dicom_index.get(image_id)
        if source is None:
            expected_exam_folder = images_root / str(image_row.get("study_id", ""))
            folder_file_count = None
            if expected_exam_folder.exists() and expected_exam_folder.is_dir():
                folder_file_count = sum(1 for item in expected_exam_folder.iterdir() if item.is_file())
            missing_rows.append(
                {
                    "study_id": image_row.get("study_id", ""),
                    "series_id": image_row.get("series_id", ""),
                    "image_id": image_id,
                    "images_root": str(images_root),
                    "expected_exam_folder": str(expected_exam_folder),
                    "expected_exam_folder_exists": expected_exam_folder.exists(),
                    "expected_exam_folder_file_count": folder_file_count,
                    "reason": "image_id_not_found_in_recursive_dicom_index",
                }
            )
            continue

        available = image_row.to_dict()
        available["source_dicom_path"] = str(source)
        available["exam_id"] = source.parent.name
        available["split_group"] = source.parent.name if args.split_by == "exam" else str(image_row.get("study_id", ""))
        available_rows.append(available)

    pd.DataFrame(missing_rows).to_csv(missing_images_output, index=False)
    available_images = pd.DataFrame(available_rows)
    print(f"Pre-conversion missing image check: {len(available_images)} available, {len(missing_rows)} missing")
    print(f"Saved missing image report to {missing_images_output}")
    if available_images.empty:
        raise SystemExit("No selected DICOM images were found. Check --input_images_path and the missing image report.")

    split_groups = split_by_group(available_images, "split_group", train_share, val_share, test_share, args.random_seed)

    grouped = target_rows.groupby("image_id", sort=False)
    metadata_rows = []
    invalid_rows = []
    converted_by_split = {"train": 0, "val": 0, "test": 0}
    boxes_by_split = {"train": 0, "val": 0, "test": 0}
    failed_conversions = 0
    crop_fallbacks = 0
    clahe_failures = 0
    pectoral_rows = []
    num_mlo_images = 0
    num_cc_images = 0
    num_pectoral_suppressed = 0
    num_pectoral_skipped_not_mlo = 0
    num_pectoral_failed = 0
    pectoral_failure_reasons = {
        "mask_too_small": 0,
        "mask_too_large": 0,
        "mask_extends_too_low": 0,
    }
    pectoral_area_ratios = []
    debug_candidates = []

    for _, image_row in tqdm(available_images.iterrows(), total=len(available_images), desc="Preparing selected images"):
        image_id = str(image_row["image_id"])
        study_id = str(image_row["study_id"])
        source = Path(str(image_row["source_dicom_path"]))
        exam_id = str(image_row["exam_id"])
        split_group = str(image_row["split_group"])
        view_position = image_row.get("view_position")
        laterality = image_row.get("laterality")

        rows = grouped.get_group(image_id)
        try:
            pixels, _ = load_dicom_pixels(str(source))
            original_height, original_width = pixels.shape[:2]
            image = normalize_to_uint8(pixels)
            image, scale_x, scale_y = resize_long_side(image, args.image_size)
        except Exception as exc:
            print(f"[WARNING] Failed to convert image_id={image_id}: {exc}")
            failed_conversions += 1
            continue

        scaled_boxes = []
        for _, box_row in rows.iterrows():
            scaled_boxes.append(
                (
                    float(box_row["xmin"]) * scale_x,
                    float(box_row["ymin"]) * scale_y,
                    float(box_row["xmax"]) * scale_x,
                    float(box_row["ymax"]) * scale_y,
                )
            )

        if args.crop_breast:
            processed_image, crop_box, cropped = crop_box_including_targets(image, scaled_boxes, args.crop_padding)
            if not cropped:
                crop_fallbacks += 1
                print(f"[WARNING] Breast crop failed for image_id={image_id}; using original processed image.")
        else:
            processed_height, processed_width = image.shape[:2]
            processed_image = image
            crop_box = (0, 0, processed_width, processed_height)

        crop_x1, crop_y1, _, _ = crop_box
        processed_height, processed_width = processed_image.shape[:2]
        adjusted_boxes = []
        label_lines = []

        for original_box, box_row in zip(scaled_boxes, rows.to_dict(orient="records")):
            xmin = max(0.0, min(processed_width, original_box[0] - crop_x1))
            ymin = max(0.0, min(processed_height, original_box[1] - crop_y1))
            xmax = max(0.0, min(processed_width, original_box[2] - crop_x1))
            ymax = max(0.0, min(processed_height, original_box[3] - crop_y1))
            yolo = voc_to_yolo(xmin, ymin, xmax, ymax, processed_width, processed_height)
            if yolo is None or not valid_yolo_box(yolo):
                invalid = dict(box_row)
                invalid.update(
                    {
                        "scaled_xmin": original_box[0],
                        "scaled_ymin": original_box[1],
                        "scaled_xmax": original_box[2],
                        "scaled_ymax": original_box[3],
                        "crop_x1": crop_box[0],
                        "crop_y1": crop_box[1],
                        "crop_x2": crop_box[2],
                        "crop_y2": crop_box[3],
                        "processed_width": processed_width,
                        "processed_height": processed_height,
                        "reason": "invalid_box_after_crop",
                    }
                )
                invalid_rows.append(invalid)
                continue
            adjusted_boxes.append((xmin, ymin, xmax, ymax))
            label_lines.append(f"0 {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")

        if not label_lines:
            print(f"[WARNING] No valid selected boxes remained for image_id={image_id}; skipping image.")
            continue

        cropped_image = processed_image.copy()
        pectoral_mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
        normalized_view = "" if view_position is None or pd.isna(view_position) else str(view_position).upper()
        if is_mlo_view(view_position):
            num_mlo_images += 1
        elif "CC" in normalized_view:
            num_cc_images += 1

        def apply_pectoral(input_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
            empty_mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
            if not args.suppress_pectoral:
                return input_image, empty_mask, {"applied": False, "reason": "suppression_disabled"}
            if view_position is None or pd.isna(view_position) or not str(view_position).strip():
                print(f"[WARNING] Missing view_position for image_id={image_id}; skipping pectoral suppression.")
                return input_image, empty_mask, {"applied": False, "reason": "missing_view_position"}
            try:
                return suppress_pectoral_muscle(
                    input_image,
                    view_position=view_position,
                    laterality=laterality,
                    method=args.pectoral_method,
                    suppress_mode=args.pectoral_suppress_mode,
                    attenuation_factor=args.pectoral_attenuation_factor,
                    intensity_percentile=args.pectoral_intensity_percentile,
                    min_area_ratio=args.pectoral_min_area_ratio,
                    max_area_ratio=args.pectoral_max_area_ratio,
                    max_width_ratio=args.pectoral_max_width_ratio,
                    max_height_ratio=args.pectoral_max_height_ratio,
                    max_y_ratio=args.pectoral_max_y_ratio,
                    debug=args.pectoral_debug,
                )
            except Exception as exc:
                print(f"[WARNING] Pectoral suppression failed for image_id={image_id}: {exc}")
                return input_image, empty_mask, {"applied": False, "reason": f"suppression_failed: {exc}"}

        def apply_clahe(input_image: np.ndarray) -> np.ndarray:
            nonlocal clahe_failures
            if not args.apply_clahe:
                return input_image
            try:
                output = apply_clahe_to_mammogram(
                    input_image,
                    clip_limit=args.clahe_clip_limit,
                    tile_grid_size=args.clahe_tile_grid_size,
                )
                assert output.shape[:2] == input_image.shape[:2], "CLAHE changed image size."
                return output
            except Exception as exc:
                clahe_failures += 1
                print(f"[WARNING] CLAHE failed for image_id={image_id}: {exc}")
                return input_image

        if args.pectoral_order == "before_clahe":
            after_pectoral_image, pectoral_mask, pectoral_info = apply_pectoral(cropped_image)
            after_clahe_image = apply_clahe(after_pectoral_image)
            final_image = after_clahe_image
        else:
            after_clahe_image = apply_clahe(cropped_image)
            after_pectoral_image, pectoral_mask, pectoral_info = apply_pectoral(after_clahe_image)
            final_image = after_pectoral_image

        if pectoral_info.get("applied"):
            num_pectoral_suppressed += 1
            pectoral_area_ratios.append(float(pectoral_info.get("mask_area_ratio", 0.0)))
            for xmin, ymin, xmax, ymax in adjusted_boxes:
                box_mask = pectoral_mask[int(ymin):int(np.ceil(ymax)), int(xmin):int(np.ceil(xmax))]
                box_area = max(1, box_mask.size)
                if np.count_nonzero(box_mask) / box_area > 0.20:
                    print(f"[WARNING] Pectoral mask overlaps target box for image_id={image_id}.")
                    break
        elif pectoral_info.get("reason") == "not_mlo_view":
            num_pectoral_skipped_not_mlo += 1
        elif args.suppress_pectoral and is_mlo_view(view_position):
            num_pectoral_failed += 1
            reason = pectoral_info.get("reason")
            if reason in pectoral_failure_reasons:
                pectoral_failure_reasons[reason] += 1

        pectoral_info = {
            "image_id": image_id,
            "view_position": "" if pd.isna(view_position) else str(view_position),
            "laterality": "" if pd.isna(laterality) else str(laterality),
            "applied": bool(pectoral_info.get("applied", False)),
            "reason": pectoral_info.get("reason", "unknown"),
            "side": pectoral_info.get("side", ""),
            "mask_area": int(pectoral_info.get("mask_area", 0)),
            "mask_area_ratio": float(pectoral_info.get("mask_area_ratio", 0.0)),
            "mask_max_y_ratio": float(pectoral_info.get("mask_max_y_ratio", 0.0)),
            "suppress_mode": args.pectoral_suppress_mode,
            "attenuation_factor": float(args.pectoral_attenuation_factor),
            "intensity_percentile": float(args.pectoral_intensity_percentile),
            "pectoral_order": args.pectoral_order,
        }

        if final_image.shape[:2] != cropped_image.shape[:2]:
            raise AssertionError("Post-crop preprocessing changed image size.")

        split_name = split_for_group(split_group, split_groups)
        pectoral_info["split"] = split_name
        pectoral_rows.append(pectoral_info)
        image_output = output_dir / "images" / split_name / f"{image_id}.png"
        label_output = output_dir / "labels" / split_name / f"{image_id}.txt"
        cv2.imwrite(str(image_output), final_image)
        label_output.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        converted_by_split[split_name] += 1
        boxes_by_split[split_name] += len(label_lines)
        metadata_rows.append(
            {
                "study_id": study_id,
                "series_id": image_row.get("series_id", ""),
                "image_id": image_id,
                "split": split_name,
                "exam_id": exam_id,
                "split_by": args.split_by,
                "split_group": split_group,
                "source_dicom_path": str(source),
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": processed_width,
                "processed_height": processed_height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "crop_x1": crop_box[0],
                "crop_y1": crop_box[1],
                "crop_x2": crop_box[2],
                "crop_y2": crop_box[3],
                "view_position": pectoral_info["view_position"],
                "laterality": pectoral_info["laterality"],
                "pectoral_applied": pectoral_info["applied"],
                "clahe_applied": bool(args.apply_clahe),
                "processed_image_path": str(image_output),
            }
        )
        debug_candidates.append({
            "split": split_name,
            "image_id": image_id,
            "cropped": cropped_image.copy(),
            "mask": pectoral_mask.copy(),
            "after_pectoral": after_pectoral_image.copy(),
            "after_clahe": after_clahe_image.copy(),
            "final": final_image.copy(),
            "boxes": adjusted_boxes,
            "pectoral_info": pectoral_info.copy(),
        })

    (output_dir / "classes.txt").write_text(f"{class_name}\n", encoding="utf-8")
    (output_dir / "class_mapping.json").write_text(json.dumps({class_name: 0}, indent=2) + "\n", encoding="utf-8")
    dataset_yaml = {
        "path": dataset_yaml_path(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: class_name},
    }
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    pd.DataFrame(metadata_rows).to_csv(metadata_output, index=False)
    pd.DataFrame(invalid_rows).to_csv(invalid_boxes_output, index=False)
    pectoral_columns = [
        "image_id", "split", "view_position", "laterality", "applied", "reason", "side",
        "mask_area", "mask_area_ratio", "mask_max_y_ratio", "suppress_mode",
        "attenuation_factor", "intensity_percentile", "pectoral_order",
    ]
    pd.DataFrame(pectoral_rows, columns=pectoral_columns).to_csv(output_dir / "pectoral_log.csv", index=False)

    if args.save_debug_samples and debug_candidates:
        rng = random.Random(args.random_seed)
        samples = rng.sample(debug_candidates, min(args.num_debug_samples, len(debug_candidates)))
        debug_dir = "debug_samples_clahe" if args.apply_clahe else "debug_samples"
        for sample in samples:
            split_name = sample["split"]
            image_id = sample["image_id"]
            boxes = sample["boxes"]
            info = sample["pectoral_info"]
            draw_debug_sample(sample["final"], boxes, output_dir / debug_dir / split_name / f"{image_id}.png", class_name)
            if args.apply_clahe:
                draw_debug_comparison(
                    sample["cropped"],
                    sample["after_clahe"],
                    boxes,
                    output_dir / "debug_samples_clahe_compare" / split_name / f"{image_id}.png",
                    class_name,
                )
            if args.suppress_pectoral:
                metadata_lines = [
                    f"image_id={image_id} view={info['view_position']}",
                    f"mode={info['suppress_mode']} factor={info['attenuation_factor']}",
                    f"area={info['mask_area_ratio']:.4f} reason={info['reason']}",
                    f"order={info['pectoral_order']}",
                ]
                debug_path = output_dir / "debug_pectoral" / split_name / image_id
                debug_path.mkdir(parents=True, exist_ok=True)
                artifacts = {
                    "cropped_before_pectoral.png": sample["cropped"],
                    "pectoral_mask.png": sample["mask"],
                    "after_pectoral.png": sample["after_pectoral"],
                    "after_clahe.png": sample["after_clahe"],
                    "final_with_boxes.png": draw_boxes_to_image(sample["final"], boxes, class_name),
                }
                for filename, artifact in artifacts.items():
                    cv2.imwrite(str(debug_path / filename), add_debug_metadata(artifact, metadata_lines))

    total_images = sum(converted_by_split.values())
    total_boxes = sum(boxes_by_split.values())
    stats = {
        "target_findings": targets,
        "target_class_name": class_name,
        "selection_mode": mode,
        "convert_selected_only": True,
        "include_negative_images": False,
        "cropping_enabled": bool(args.crop_breast),
        "crop_method": "otsu_largest_component",
        "crop_padding": args.crop_padding,
        "clahe_enabled": bool(args.apply_clahe),
        "clahe_applied_after_crop": bool(args.apply_clahe),
        "clahe_clip_limit": float(args.clahe_clip_limit),
        "clahe_tile_grid_size": int(args.clahe_tile_grid_size),
        "num_clahe_failures": int(clahe_failures),
        "suppress_pectoral": bool(args.suppress_pectoral),
        "pectoral_method": args.pectoral_method,
        "pectoral_suppress_mode": args.pectoral_suppress_mode,
        "pectoral_attenuation_factor": float(args.pectoral_attenuation_factor),
        "pectoral_intensity_percentile": float(args.pectoral_intensity_percentile),
        "pectoral_min_area_ratio": float(args.pectoral_min_area_ratio),
        "pectoral_max_area_ratio": float(args.pectoral_max_area_ratio),
        "pectoral_max_width_ratio": float(args.pectoral_max_width_ratio),
        "pectoral_max_height_ratio": float(args.pectoral_max_height_ratio),
        "pectoral_max_y_ratio": float(args.pectoral_max_y_ratio),
        "pectoral_order": args.pectoral_order,
        "pectoral_applied_to_mlo_only": True,
        "num_mlo_images": int(num_mlo_images),
        "num_cc_images": int(num_cc_images),
        "num_pectoral_suppressed": int(num_pectoral_suppressed),
        "num_pectoral_skipped_not_mlo": int(num_pectoral_skipped_not_mlo),
        "num_pectoral_failed": int(num_pectoral_failed),
        "num_pectoral_mask_too_small": int(pectoral_failure_reasons["mask_too_small"]),
        "num_pectoral_mask_too_large": int(pectoral_failure_reasons["mask_too_large"]),
        "num_pectoral_mask_extends_too_low": int(pectoral_failure_reasons["mask_extends_too_low"]),
        "avg_pectoral_mask_area_ratio": (
            float(np.mean(pectoral_area_ratios)) if pectoral_area_ratios else 0.0
        ),
        "split_method": f"{args.split_by}_grouped",
        "split_by": args.split_by,
        "train_share": float(train_share),
        "val_share": float(val_share),
        "test_share": float(test_share),
        "num_selected_image_ids_before_conversion": int(len(selected_images)),
        "num_converted_images": int(total_images),
        "num_skipped_missing_dicoms": int(len(missing_rows)),
        "num_failed_conversions": int(failed_conversions),
        "num_crop_fallbacks": int(crop_fallbacks),
        "num_excluded_non_target_images": int(annotations["image_id"].nunique() - len(selected_images)),
        "num_train_images": int(converted_by_split["train"]),
        "num_val_images": int(converted_by_split["val"]),
        "num_test_images": int(converted_by_split["test"]),
        "num_train_boxes": int(boxes_by_split["train"]),
        "num_val_boxes": int(boxes_by_split["val"]),
        "num_test_boxes": int(boxes_by_split["test"]),
        "total_images": int(total_images),
        "total_boxes": int(total_boxes),
        "num_invalid_boxes": int(len(invalid_rows)),
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Wrote selected YOLO dataset to {output_dir}")
    print(f"Wrote dataset YAML to {output_dir / 'dataset.yaml'}")
    if args.save_debug_samples:
        print(f"Wrote debug samples to {output_dir / 'debug_samples'}")


if __name__ == "__main__":
    main()
