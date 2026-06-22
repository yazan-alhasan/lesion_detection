from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import BBOX_COLUMNS, has_valid_bbox
from src.utils.dicom_index import build_dicom_index
from src.utils.paths import ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether selected DICOM images exist locally.")
    parser.add_argument("--subset_csv", default=None, help="CSV containing at least image_id, optionally study_id.")
    parser.add_argument("--annotations", "--finding_csv", dest="annotations", default=None)
    parser.add_argument(
        "--target_finding",
        help="When --annotations is used, check only images containing this one category.",
    )
    parser.add_argument(
        "--target_findings",
        nargs="+",
        help="When --annotations is used, check only images containing any of these categories.",
    )
    parser.add_argument(
        "--images_root",
        "--input_images_path",
        "--images_input_path",
        dest="images_root",
        default="data/raw/images",
        help="Folder containing DICOM files. The check searches this folder recursively.",
    )
    parser.add_argument("--missing_output", default="data/subset/missing_images.csv")
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


def load_selected_rows(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.subset_csv) == bool(args.annotations):
        raise ValueError("Use exactly one input source: --subset_csv or --annotations.")

    if args.subset_csv:
        selected = pd.read_csv(project_path(args.subset_csv))
    else:
        if bool(args.target_finding) == bool(args.target_findings):
            raise ValueError("With --annotations, use exactly one mode: --target_finding or --target_findings.")
        annotations = pd.read_csv(project_path(args.annotations))
        required_columns = {"study_id", "image_id", "finding_categories", *BBOX_COLUMNS}
        missing_columns = required_columns - set(annotations.columns)
        if missing_columns:
            raise ValueError(f"Annotation CSV is missing required columns: {sorted(missing_columns)}")

        targets = [args.target_finding] if args.target_finding else list(args.target_findings)
        target_set = {normalize_category(target) for target in targets}
        annotations = annotations.copy()
        annotations["category_list"] = annotations["finding_categories"].apply(parse_categories)
        selected_columns = [column for column in ("study_id", "series_id", "image_id") if column in annotations.columns]
        selected = annotations[
            annotations.apply(lambda row: row_matches_targets(row, target_set) and has_valid_bbox(row), axis=1)
        ][selected_columns].drop_duplicates()

    if "image_id" not in selected.columns:
        raise ValueError("Selected CSV must contain an image_id column.")

    selected = selected.copy()
    selected["image_id"] = selected["image_id"].astype(str)
    if "study_id" in selected.columns:
        selected["study_id"] = selected["study_id"].astype(str)
    else:
        selected["study_id"] = ""
    return selected.drop_duplicates("image_id").sort_values(["study_id", "image_id"], kind="stable")


def main() -> None:
    args = parse_args()
    selected = load_selected_rows(args)
    dicom_index = build_dicom_index(args.images_root)

    missing_rows = []
    available = 0
    for _, row in selected.iterrows():
        image_id = str(row["image_id"])
        path = dicom_index.get(image_id)
        if path is not None:
            available += 1
        else:
            expected_exam_folder = project_path(args.images_root) / str(row.get("study_id", ""))
            folder_file_count = None
            if expected_exam_folder.exists() and expected_exam_folder.is_dir():
                folder_file_count = sum(1 for item in expected_exam_folder.iterdir() if item.is_file())
            missing = row.to_dict()
            missing["images_root"] = str(project_path(args.images_root))
            missing["expected_exam_folder"] = str(expected_exam_folder)
            missing["expected_exam_folder_exists"] = expected_exam_folder.exists()
            missing["expected_exam_folder_file_count"] = folder_file_count
            missing["reason"] = "image_id_not_found_in_recursive_dicom_index"
            missing_rows.append(missing)

    output = ensure_parent(args.missing_output)
    pd.DataFrame(missing_rows).to_csv(output, index=False)
    print(f"Selected images: {len(selected)}")
    print(f"Indexed DICOM files: {len(set(dicom_index.values()))}")
    print(f"Available DICOM files: {available}")
    print(f"Missing DICOM files: {len(missing_rows)}")
    print(f"Saved missing image report to {output}")


if __name__ == "__main__":
    main()
