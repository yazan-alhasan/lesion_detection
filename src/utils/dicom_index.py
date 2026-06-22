from __future__ import annotations

from pathlib import Path

from src.utils.paths import project_path


DICOM_EXTENSIONS = {".dicom", ".dcm", ""}


def is_dicom_candidate(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS


def build_dicom_index(images_root: str | Path) -> dict[str, Path]:
    root = project_path(images_root)
    index: dict[str, Path] = {}
    if not root.exists():
        return index

    for path in root.rglob("*"):
        if not is_dicom_candidate(path):
            continue
        index.setdefault(path.stem, path)
        index.setdefault(path.name, path)
    return index


def find_dicom_path(
    images_root: str | Path,
    image_id: str,
    study_id: str | None = None,
    dicom_index: dict[str, Path] | None = None,
) -> Path | None:
    root = project_path(images_root)
    image_id = str(image_id)
    candidates = []

    if study_id is not None:
        study_dir = root / str(study_id)
        candidates.extend(
            [
                study_dir / f"{image_id}.dicom",
                study_dir / f"{image_id}.dcm",
                study_dir / image_id,
            ]
        )

    candidates.extend(
        [
            root / f"{image_id}.dicom",
            root / f"{image_id}.dcm",
            root / image_id,
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    if dicom_index is None:
        dicom_index = build_dicom_index(root)
    return dicom_index.get(image_id)
