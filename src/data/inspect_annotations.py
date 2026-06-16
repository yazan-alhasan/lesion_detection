from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.bbox_utils import BBOX_COLUMNS
from src.utils.paths import ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect VinDr-Mammo finding annotations.")
    parser.add_argument("--finding_csv", required=True)
    parser.add_argument("--output", default="runs/annotation_summary.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finding_csv = project_path(args.finding_csv)
    output = ensure_parent(args.output)

    df = pd.read_csv(finding_csv)
    bbox_missing = df[list(BBOX_COLUMNS)].isna().any(axis=1).sum() if set(BBOX_COLUMNS).issubset(df.columns) else 0
    no_finding_count = (
        df["finding_categories"].astype(str).str.contains("No Finding", case=False, na=False).sum()
        if "finding_categories" in df.columns
        else 0
    )

    lines = [
        f"Annotation file: {finding_csv}",
        "",
        "Columns:",
        *[f"- {column}" for column in df.columns],
        "",
        f"Rows: {len(df)}",
        f"Unique studies: {df['study_id'].nunique() if 'study_id' in df.columns else 'N/A'}",
        f"Unique images: {df['image_id'].nunique() if 'image_id' in df.columns else 'N/A'}",
        "",
        "Finding category counts:",
    ]
    if "finding_categories" in df.columns:
        lines.extend(df["finding_categories"].fillna("NA").value_counts().to_string().splitlines())
    else:
        lines.append("finding_categories column not found")

    lines.extend(["", "Split counts:"])
    if "split" in df.columns:
        lines.extend(df["split"].fillna("NA").value_counts().to_string().splitlines())
    else:
        lines.append("split column not found")

    lines.extend(
        [
            "",
            f"Rows with missing bounding boxes: {bbox_missing}",
            f"No Finding rows: {no_finding_count}",
        ]
    )

    text = "\n".join(lines)
    print(text)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved summary to {output}")


if __name__ == "__main__":
    main()
