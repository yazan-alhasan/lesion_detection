from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import ensure_dir, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split processed PNG and YOLO labels by study.")
    parser.add_argument("--metadata_csv", default="data/processed/image_metadata.csv")
    parser.add_argument("--subset_csv", default="data/subset/subset_image_ids.csv")
    parser.add_argument("--images_all", default="data/processed/images/all")
    parser.add_argument("--labels_all", default="data/processed/labels/all")
    parser.add_argument("--output_root", default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--use_existing_test_split",
        action="store_true",
        help="Keep rows marked as test in the subset CSV instead of creating a new test split.",
    )
    return parser.parse_args()


def study_split(df: pd.DataFrame, val_ratio: float, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    studies = df["study_id"].astype(str).drop_duplicates()
    if len(studies) < 2:
        return df.copy(), df.iloc[0:0].copy()
    train_studies, val_studies = train_test_split(
        studies,
        test_size=val_ratio,
        random_state=random_seed,
    )
    train = df[df["study_id"].astype(str).isin(set(train_studies))]
    val = df[df["study_id"].astype(str).isin(set(val_studies))]
    return train, val


def three_way_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    studies = df["study_id"].astype(str).drop_duplicates()
    if len(studies) < 3:
        splits = {"train": df.copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}
        return splits

    temp_ratio = val_ratio + test_ratio
    train_studies, temp_studies = train_test_split(studies, test_size=temp_ratio, random_state=random_seed)
    if len(temp_studies) < 2:
        val_studies = temp_studies
        test_studies = []
    else:
        relative_test_ratio = test_ratio / temp_ratio
        val_studies, test_studies = train_test_split(
            temp_studies,
            test_size=relative_test_ratio,
            random_state=random_seed,
        )
    return {
        "train": df[df["study_id"].astype(str).isin(set(train_studies))],
        "val": df[df["study_id"].astype(str).isin(set(val_studies))],
        "test": df[df["study_id"].astype(str).isin(set(test_studies))],
    }


def reset_split_dir(path: Path) -> Path:
    path = ensure_dir(path)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    cache_path = path.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()
    return path


def copy_split(rows: pd.DataFrame, split_name: str, images_all, labels_all, output_root) -> int:
    image_out = reset_split_dir(output_root / "images" / split_name)
    label_out = reset_split_dir(output_root / "labels" / split_name)
    copied = 0
    for _, row in rows.iterrows():
        image_id = str(row["image_id"])
        image_src = images_all / f"{image_id}.png"
        label_src = labels_all / f"{image_id}.txt"
        if not image_src.exists() or not label_src.exists():
            continue
        shutil.copy2(image_src, image_out / image_src.name)
        shutil.copy2(label_src, label_out / label_src.name)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(project_path(args.metadata_csv))
    subset = pd.read_csv(project_path(args.subset_csv))
    metadata["image_id"] = metadata["image_id"].astype(str)
    subset["image_id"] = subset["image_id"].astype(str)
    df = metadata.merge(subset[["image_id", "split"]], on="image_id", how="left", suffixes=("", "_subset"))
    if "split_subset" in df.columns:
        df["split"] = df["split_subset"].fillna(df.get("split", "training"))
        df = df.drop(columns=["split_subset"])
    df["split"] = df["split"].fillna("training").astype(str).str.lower()

    has_test = args.use_existing_test_split and (df["split"] == "test").any()
    if has_test:
        train_source = df[df["split"] != "test"]
        test = df[df["split"] == "test"]
        train, val = study_split(train_source, args.val_ratio, args.random_seed)
        split_frames = {"train": train, "val": val, "test": test}
    else:
        split_frames = three_way_split(df, args.val_ratio, args.test_ratio, args.random_seed)

    images_all = project_path(args.images_all)
    labels_all = project_path(args.labels_all)
    output_root = ensure_dir(args.output_root)

    rows = []
    for split_name, frame in split_frames.items():
        copied = copy_split(frame, split_name, images_all, labels_all, output_root)
        rows.append({"split": split_name, "images": copied})
        print(f"{split_name}: {copied} images")

    pd.DataFrame(rows).to_csv(output_root / "split_summary.csv", index=False)
    print(f"Saved split summary to {output_root / 'split_summary.csv'}")


if __name__ == "__main__":
    main()
