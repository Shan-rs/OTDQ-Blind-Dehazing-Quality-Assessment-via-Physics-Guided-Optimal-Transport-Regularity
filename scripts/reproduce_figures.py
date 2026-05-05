#!/usr/bin/env python3
"""Reproduce an OTDQ diagnostic figure from a score CSV and image paths."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from otdq_eval.config import merged_otdq_config, otdq_kwargs
from otdq_eval.diagnostics import diagnostic_maps
from otdq_eval.io import align_to_hazy, load_rgb, resolve_dataset_path


def select_rows(df: pd.DataFrame, group_col: str, target_col: str, score_col: str):
    candidates = []
    for group_name, group in df.groupby(group_col, sort=True):
        group = group.dropna(subset=[target_col, score_col])
        if len(group) < 2:
            continue
        high = group.loc[group[target_col].idxmax()]
        low = group.loc[group[target_col].idxmin()]
        candidates.append(
            {
                "group": group_name,
                "high_index": int(high.name),
                "low_index": int(low.name),
                "target_gap": float(high[target_col] - low[target_col]),
                "score_gap": float(high[score_col] - low[score_col]),
            }
        )
    if not candidates:
        raise ValueError("No group with at least two valid rows was found.")
    table = pd.DataFrame(candidates)
    agreeing = table[table["score_gap"] > 0].copy()
    pool = agreeing if not agreeing.empty else table
    selected = pool.sort_values(["target_gap", "score_gap", "group"], ascending=[False, False, True]).iloc[0]
    return df.loc[selected["high_index"]], df.loc[selected["low_index"]], table


def resolve_image(row, path_col: str | None, root: Path | None, name_col: str, fallback: str):
    if path_col and path_col in row and pd.notna(row[path_col]):
        path = Path(str(row[path_col]))
        if path.is_absolute() or root is None:
            return path
        return root / path
    if root is None:
        raise ValueError(f"Need --image-root when {path_col or name_col} is relative.")
    return resolve_dataset_path(root, row[name_col], fallback_subdir=fallback)


def robust_scale(arrays):
    values = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in arrays if np.isfinite(arr).any()])
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, 99.0)), 0.15)


def draw(output: Path, hazy, high, low, high_scores, low_scores, high_maps, low_maps, target_col):
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.8), constrained_layout=True)
    titles = ["Hazy input", "Dehazed output", "Transport error", "Structure error", "Artifact map"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11)
    scales = {
        "transport": robust_scale([high_maps["transport"], low_maps["transport"]]),
        "structure": robust_scale([high_maps["structure"], low_maps["structure"]]),
        "artifact": robust_scale([high_maps["artifact"], low_maps["artifact"]]),
    }
    rows = [
        ("High target", high, high_scores, high_maps),
        ("Low target", low, low_scores, low_maps),
    ]
    for row_idx, (label, image, scores, maps) in enumerate(rows):
        for ax in axes[row_idx]:
            ax.set_axis_off()
        axes[row_idx, 0].imshow(hazy)
        axes[row_idx, 0].set_ylabel(label, fontsize=11)
        axes[row_idx, 1].imshow(image)
        axes[row_idx, 2].imshow(maps["transport"], cmap="inferno", vmin=0, vmax=scales["transport"])
        axes[row_idx, 3].imshow(maps["structure"], cmap="inferno", vmin=0, vmax=scales["structure"])
        axes[row_idx, 4].imshow(maps["artifact"], cmap="inferno", vmin=0, vmax=scales["artifact"])
        axes[row_idx, 1].set_xlabel(
            f"OTDQ {scores['otdq']:.3f} | S_vis {scores['s_vis']:.3f} | "
            f"S_str {scores['s_str']:.3f} | S_arti {scores['s_arti']:.3f}",
            fontsize=8,
        )
    fig.suptitle(f"OTDQ diagnostic visualization ({target_col} selected pair)", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce a diagnostic OTDQ figure.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/fig_otdq_diagnostics.png"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "otdq_default.json")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--group-col", default="Haze_name")
    parser.add_argument("--target-col", default="MOS_Mean")
    parser.add_argument("--score-col", default="otdq")
    parser.add_argument("--hazy-path-col", default=None)
    parser.add_argument("--dehazed-path-col", default=None)
    parser.add_argument("--hazy-name-col", default="Haze_name")
    parser.add_argument("--dehazed-name-col", default=None)
    parser.add_argument("--resize-method", default=None)
    parser.add_argument("--selection-output", type=Path, default=None)
    args = parser.parse_args()

    config = merged_otdq_config(args.config, resize_method=args.resize_method)
    df = pd.read_csv(args.csv)
    dehazed_name_col = args.dehazed_name_col or ("Dehaze_name" if "Dehaze_name" in df.columns else "name")
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    high_row, low_row, candidates = select_rows(df, args.group_col, args.target_col, args.score_col)

    hazy_path = resolve_image(high_row, args.hazy_path_col, args.image_root, args.hazy_name_col, "Haze")
    high_path = resolve_image(high_row, args.dehazed_path_col, args.image_root, dehazed_name_col, "Dehaze")
    low_path = resolve_image(low_row, args.dehazed_path_col, args.image_root, dehazed_name_col, "Dehaze")
    hazy = load_rgb(hazy_path)
    high = align_to_hazy(hazy, load_rgb(high_path), config["resize_method"])
    low = align_to_hazy(hazy, load_rgb(low_path), config["resize_method"])

    params = otdq_kwargs(config)
    high_scores, high_maps = diagnostic_maps(hazy, high, **params)
    low_scores, low_maps = diagnostic_maps(hazy, low, **params)
    draw(args.output, hazy, high, low, high_scores, low_scores, high_maps, low_maps, args.target_col)

    selection_output = args.selection_output or args.output.with_name("diagnostic_selection.csv")
    selected = pd.DataFrame(
        [
            {"quality_label": "high", "hazy_path": str(hazy_path), "dehazed_path": str(high_path), **high_scores},
            {"quality_label": "low", "hazy_path": str(hazy_path), "dehazed_path": str(low_path), **low_scores},
        ]
    )
    selected.to_csv(selection_output, index=False)
    candidates.to_csv(args.output.with_name("diagnostic_candidates.csv"), index=False)
    print(f"[SAVE] {args.output}")
    print(f"[SAVE] {selection_output}")


if __name__ == "__main__":
    main()
