"""Batch OTDQ scoring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from .config import otdq_kwargs
from .io import align_to_hazy, load_rgb
from .metric import compute_otdq


SCORE_FIELDS = [
    "otdq",
    "s_vis",
    "s_str",
    "s_arti",
    "tr",
    "sf",
    "sdc",
    "sdc_factor",
    "patch_size",
    "ssim_structure",
    "n_pixels",
    "n_textured",
    "n_flat",
    "halo_score",
    "dark_score",
    "patch_score",
    "patchiness",
]


def score_pair(
    hazy_path: str | Path,
    dehazed_path: str | Path,
    *,
    resize_method: str = "LANCZOS",
    config: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    hazy = load_rgb(hazy_path)
    dehazed = align_to_hazy(hazy, load_rgb(dehazed_path), resize_method)
    scores = compute_otdq(hazy, dehazed, **otdq_kwargs(dict(config or {})))
    row: Dict[str, object] = {}
    for key in SCORE_FIELDS:
        value = scores.get(key, np.nan)
        if isinstance(value, (np.integer, int)):
            row[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            row[key] = float(value)
        else:
            row[key] = value
    return row


def score_records(
    records: Iterable[Mapping[str, object]],
    *,
    resize_method: str = "LANCZOS",
    config: Mapping[str, object] | None = None,
    keep_input_columns: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, record in enumerate(records, 1):
        hazy_path = record["hazy_path"]
        dehazed_path = record["dehazed_path"]
        row: Dict[str, object] = dict(record) if keep_input_columns else {}
        try:
            row.update(score_pair(hazy_path, dehazed_path, resize_method=resize_method, config=config))
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            for key in SCORE_FIELDS:
                row[key] = np.nan
            row["status"] = "error"
            row["error"] = repr(exc)
        if "id" not in row:
            row["id"] = str(index)
        rows.append(row)
    return rows
