from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import BBOX_COLUMNS
from src.utils.paths import dicom_path, ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible VinDr-Mammo subset.")
    parser.add_argument("--finding_csv", required=True)
    parser.add_argument("--output_csv", default="data/subset/subset_image_ids.csv")
    parser.add_argument("--max_per_class", type=int, default=50)
    parser.add_argument("--num_negative", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument(
        "--require_images",
        action="store_true",
        default=True,
        help="Only select images whose DICOM file exists locally. This is the default.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Allow subset rows even when the DICOM file is not present locally.",
    )
    parser.add_argument(
        "--all_available",
        action="store_true",
        help="Select every locally available annotated image, ignoring class and negative caps.",
    )
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


def has_bbox(df: pd.DataFrame) -> pd.Series:
    if not set(BBOX_COLUMNS).issubset(df.columns):
        return pd.Series(False, index=df.index)
    coords = df[list(BBOX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return coords.notna().all(axis=1) & (coords["xmax"] > coords["xmin"]) & (coords["ymax"] > coords["ymin"])


def image_metadata(row: pd.Series, category: str, is_negative: bool) -> dict:
    return {
        "study_id": row.get("study_id"),
        "image_id": row.get("image_id"),
        "split": row.get("split", "training"),
        "selected_category": category,
        "is_negative": is_negative,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(project_path(args.finding_csv))
    if "image_id" not in df.columns or "study_id" not in df.columns:
        raise ValueError("finding_annotations.csv must contain image_id and study_id columns.")

    df = df.copy()
    df["has_bbox"] = has_bbox(df)
    df["category_list"] = df["finding_categories"].apply(parse_categories)
    require_images = args.require_images and not args.allow_missing
    if require_images:
        exists = df.apply(
            lambda row: dicom_path(args.images_root, row["study_id"], row["image_id"]).exists(),
            axis=1,
        )
        before = df["image_id"].nunique()
        df = df[exists].copy()
        after = df["image_id"].nunique()
        print(f"Filtered to locally available DICOM images: {after}/{before}")

    selected: dict[str, dict] = {}
    positive = df[df["has_bbox"]].copy()
    positive = positive.sample(frac=1.0, random_state=args.random_seed)
    categories = sorted({cat for cats in positive["category_list"] for cat in cats if cat.lower() != "no finding"})

    if args.all_available:
        for _, row in positive.drop_duplicates("image_id").iterrows():
            categories_for_row = [cat for cat in row["category_list"] if cat.lower() != "no finding"]
            selected_category = categories_for_row[0] if categories_for_row else "Lesion"
            selected.setdefault(str(row["image_id"]), image_metadata(row, selected_category, False))

        negative_mask = df["category_list"].apply(lambda cats: any(cat.lower() == "no finding" for cat in cats))
        for _, row in df[negative_mask].drop_duplicates("image_id").iterrows():
            selected.setdefault(str(row["image_id"]), image_metadata(row, "No Finding", True))
    else:
        for category in categories:
            category_rows = positive[positive["category_list"].apply(lambda cats: category in cats)]
            for _, row in category_rows.drop_duplicates("image_id").head(args.max_per_class).iterrows():
                selected.setdefault(str(row["image_id"]), image_metadata(row, category, False))

        if args.num_negative > 0:
            negative_mask = df["category_list"].apply(lambda cats: any(cat.lower() == "no finding" for cat in cats))
            negatives = (
                df[negative_mask]
                .drop_duplicates("image_id")
                .sample(frac=1.0, random_state=args.random_seed)
                .head(args.num_negative)
            )
            for _, row in negatives.iterrows():
                selected.setdefault(str(row["image_id"]), image_metadata(row, "No Finding", True))

    subset = pd.DataFrame(selected.values())
    if not subset.empty:
        subset = subset.sort_values(["split", "study_id", "image_id"], kind="stable")

    output = ensure_parent(args.output_csv)
    subset.to_csv(output, index=False)
    print(f"Selected {len(subset)} images from {len(categories)} lesion categories.")
    if args.all_available:
        print("All available annotated images were selected.")
    if require_images:
        print("Subset contains only locally available DICOM images.")
    else:
        print("Subset may contain missing local DICOM images.")
    print(f"Saved subset CSV to {output}")


if __name__ == "__main__":
    main()
