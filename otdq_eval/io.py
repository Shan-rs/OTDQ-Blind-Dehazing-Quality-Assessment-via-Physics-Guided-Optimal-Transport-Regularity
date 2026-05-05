"""Image and manifest I/O utilities."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

RESIZE_METHODS = {
    "NEAREST": Image.NEAREST,
    "BILINEAR": Image.BILINEAR,
    "BICUBIC": Image.BICUBIC,
    "LANCZOS": Image.LANCZOS,
}


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 array."""
    return np.asarray(Image.open(path).convert("RGB"))


def align_to_hazy(hazy: np.ndarray, dehazed: np.ndarray, method: str = "LANCZOS") -> np.ndarray:
    """Resize a dehazed image to the hazy image size if needed."""
    if hazy.shape[:2] == dehazed.shape[:2]:
        return dehazed
    height, width = hazy.shape[:2]
    resample = RESIZE_METHODS.get(method.upper(), Image.LANCZOS)
    return np.asarray(Image.fromarray(dehazed).resize((width, height), resample))


def image_files(directory: str | Path) -> List[Path]:
    path = Path(directory)
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _numeric_prefix(stem: str) -> str:
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else stem


def match_image_pairs(hazy_dir: str | Path, dehazed_dir: str | Path) -> List[Dict[str, str]]:
    """Match hazy/dehazed images by exact stem, then by leading numeric prefix."""
    hazy = {path.stem: path for path in image_files(hazy_dir)}
    dehazed = {path.stem: path for path in image_files(dehazed_dir)}

    common = sorted(set(hazy) & set(dehazed))
    if common:
        return [
            {"id": stem, "hazy_path": str(hazy[stem]), "dehazed_path": str(dehazed[stem])}
            for stem in common
        ]

    hazy_prefix = {_numeric_prefix(stem): (stem, path) for stem, path in hazy.items()}
    dehazed_prefix = {_numeric_prefix(stem): (stem, path) for stem, path in dehazed.items()}
    common_prefixes = sorted(set(hazy_prefix) & set(dehazed_prefix))
    return [
        {
            "id": prefix,
            "hazy_path": str(hazy_prefix[prefix][1]),
            "dehazed_path": str(dehazed_prefix[prefix][1]),
        }
        for prefix in common_prefixes
    ]


def resolve_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return Path(base_dir) / path


def resolve_dataset_path(root: str | Path, value: str, fallback_subdir: Optional[str] = None) -> Path:
    """Resolve a dataset-relative path, with optional fallback to a named subfolder."""
    root_path = Path(root)
    rel = Path(str(value).replace("\\", "/"))
    direct = root_path / rel
    if direct.exists() or fallback_subdir is None:
        return direct
    return root_path / fallback_subdir / rel.name


def read_manifest(
    manifest: str | Path,
    *,
    base_dir: str | Path | None = None,
    id_col: str = "id",
    hazy_col: str = "hazy_path",
    dehazed_col: str = "dehazed_path",
) -> List[Dict[str, str]]:
    """Read a pair manifest CSV into dictionaries with absolute-ish paths."""
    manifest_path = Path(manifest)
    root = Path(base_dir) if base_dir is not None else manifest_path.parent
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {hazy_col, dehazed_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            item = dict(row)
            item["id"] = str(row.get(id_col) or row.get("image_id") or row.get("name") or index)
            item["hazy_path"] = str(resolve_path(row[hazy_col], root))
            item["dehazed_path"] = str(resolve_path(row[dehazed_col], root))
            rows.append(item)
    return rows


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], fieldnames: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
