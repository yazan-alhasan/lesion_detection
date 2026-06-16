from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import dicom_path, ensure_parent, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check which selected DICOM images exist locally.")
    parser.add_argument("--subset_csv", default="data/subset/subset_image_ids.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--missing_output", default="data/subset/missing_images.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(project_path(args.subset_csv))
    missing_rows = []
    available = 0

    for _, row in subset.iterrows():
        path = dicom_path(args.images_root, row["study_id"], row["image_id"])
        if path.exists():
            available += 1
        else:
            missing = row.to_dict()
            missing["expected_path"] = str(path)
            missing_rows.append(missing)

    output = ensure_parent(args.missing_output)
    pd.DataFrame(missing_rows).to_csv(output, index=False)
    print(f"Selected images: {len(subset)}")
    print(f"Available DICOM files: {available}")
    print(f"Missing DICOM files: {len(missing_rows)}")
    print(f"Saved missing image report to {output}")


if __name__ == "__main__":
    main()
