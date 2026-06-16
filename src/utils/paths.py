from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_parent(path: str | Path) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dicom_path(images_root: str | Path, study_id: str, image_id: str) -> Path:
    return project_path(images_root) / str(study_id) / f"{image_id}.dicom"


def png_name(image_id: str) -> str:
    return f"{image_id}.png"
