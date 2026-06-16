from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.image_utils import apply_clahe, load_dicom_pixels, normalize_to_uint8, resize_long_side
from src.utils.paths import dicom_path, ensure_dir, ensure_parent, png_name, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert selected VinDr-Mammo DICOM files to PNG.")
    parser.add_argument("--subset_csv", default="data/subset/subset_image_ids.csv")
    parser.add_argument("--images_root", default="data/raw/images")
    parser.add_argument("--output_dir", default="data/processed/images/all")
    parser.add_argument("--metadata_output", default="data/processed/image_metadata.csv")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--apply_clahe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = pd.read_csv(project_path(args.subset_csv))
    output_dir = ensure_dir(args.output_dir)
    metadata_output = ensure_parent(args.metadata_output)
    rows = []
    missing = 0
    failed = 0

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Converting DICOM"):
        source = dicom_path(args.images_root, row["study_id"], row["image_id"])
        if not source.exists():
            missing += 1
            continue

        try:
            pixels, _ = load_dicom_pixels(str(source))
            original_height, original_width = pixels.shape[:2]
            image = normalize_to_uint8(pixels)
            if args.apply_clahe:
                image = apply_clahe(image)
            image, scale_x, scale_y = resize_long_side(image, args.image_size)
        except Exception as exc:
            failed += 1
            print(f"Failed to convert {source}: {exc}")
            continue

        output_path = output_dir / png_name(str(row["image_id"]))
        cv2.imwrite(str(output_path), image)
        processed_height, processed_width = image.shape[:2]
        rows.append(
            {
                "study_id": row["study_id"],
                "image_id": row["image_id"],
                "split": row.get("split", "training"),
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": processed_width,
                "processed_height": processed_height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "processed_image_path": str(output_path),
            }
        )

    pd.DataFrame(rows).to_csv(metadata_output, index=False)
    print(f"Converted images: {len(rows)}")
    print(f"Missing images skipped: {missing}")
    print(f"Failed conversions: {failed}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
