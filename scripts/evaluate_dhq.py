#!/usr/bin/env python3
"""Evaluate OTDQ and objective metrics on a DHQ-style MOS CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from otdq_eval.config import merged_otdq_config
from otdq_eval.io import resolve_dataset_path
from otdq_eval.scoring import score_records
from otdq_eval.stats import DEFAULT_LOWER_IS_BETTER, correlation_table, pairwise_accuracy, write_table

DEFAULT_METRICS = [
    "niqe",
    "brisque",
    "fade",
    "musiq",
    "liqe",
    "clipiqa+",
    "maniqa",
    "hyperiqa",
    "nima",
    "entropy",
    "visibility",
    "dark_channel",
    "saturation",
    "contrast",
    "otdq",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DHQ MOS consistency evaluation.")
    parser.add_argument("--dhq-csv", required=True, type=Path, help="CSV with Haze_name, name/Dehaze_name, MOS_Mean.")
    parser.add_argument("--dhq-root", type=Path, default=None, help="Root containing Haze/ and Dehaze/ images.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dhq"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dhq_eval.json")
    parser.add_argument("--compute-otdq", action="store_true", help="Recompute OTDQ from image files.")
    parser.add_argument("--haze-col", default="Haze_name")
    parser.add_argument("--dehaze-col", default=None, help="Defaults to Dehaze_name when present, otherwise name.")
    parser.add_argument("--mos-col", default="MOS_Mean")
    parser.add_argument("--group-col", default="Haze_name")
    parser.add_argument("--metric-cols", nargs="*", default=None)
    parser.add_argument("--lower-is-better", nargs="*", default=sorted(DEFAULT_LOWER_IS_BETTER))
    parser.add_argument("--digits", type=int, default=4)
    return parser


def build_image_records(df: pd.DataFrame, args: argparse.Namespace):
    if args.dhq_root is None:
        raise ValueError("--dhq-root is required when computing OTDQ.")
    dehaze_col = args.dehaze_col or ("Dehaze_name" if "Dehaze_name" in df.columns else "name")
    records = []
    for idx, row in df.iterrows():
        hazy_path = resolve_dataset_path(args.dhq_root, row[args.haze_col], fallback_subdir="Haze")
        dehazed_path = resolve_dataset_path(args.dhq_root, row[dehaze_col], fallback_subdir="Dehaze")
        records.append(
            {
                "source_index": int(idx),
                "id": str(row.get(dehaze_col, idx)),
                "hazy_path": str(hazy_path),
                "dehazed_path": str(dehazed_path),
            }
        )
    return records


def main() -> None:
    args = build_parser().parse_args()
    config = merged_otdq_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dhq_csv)
    dehaze_col = args.dehaze_col or ("Dehaze_name" if "Dehaze_name" in df.columns else "name")
    required = {args.haze_col, dehaze_col, args.mos_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.dhq_csv} missing required columns: {sorted(missing)}")
    df = df[df[dehaze_col].astype(str).str.lower() != "average"].copy()
    df[args.mos_col] = pd.to_numeric(df[args.mos_col], errors="coerce")
    df = df.dropna(subset=[args.haze_col, dehaze_col, args.mos_col]).reset_index(drop=True)

    if args.compute_otdq or "otdq" not in df.columns:
        records = build_image_records(df, args)
        scored = pd.DataFrame(score_records(records, resize_method=config["resize_method"], config=config))
        score_cols = [col for col in scored.columns if col not in {"id", "source_index", "hazy_path", "dehazed_path"}]
        df = df.drop(columns=[col for col in score_cols if col in df.columns], errors="ignore")
        df = df.merge(scored[["source_index", *score_cols]], left_index=True, right_on="source_index", how="left")

    scores_path = args.output_dir / "dhq_scores_with_otdq.csv"
    df.to_csv(scores_path, index=False)
    print(f"[SAVE] {scores_path}")

    metrics = args.metric_cols or [metric for metric in DEFAULT_METRICS if metric in df.columns]
    corr = correlation_table(df, metrics, args.mos_col, lower_is_better=args.lower_is_better)
    pair = pairwise_accuracy(
        df,
        metrics=metrics,
        references=[args.mos_col],
        group_col=args.group_col,
        lower_is_better=args.lower_is_better,
    )
    pair = pair.rename(columns={"Reference": "MOS_Reference"})

    for stem, table in [
        ("table1_dhq_mos_correlations", corr),
        ("table2_dhq_pairwise_preference", pair),
    ]:
        write_table(table, args.output_dir / f"{stem}.csv", args.digits)
        write_table(table, args.output_dir / f"{stem}.md", args.digits)
        print(f"[SAVE] {args.output_dir / f'{stem}.csv'}")
        print(f"[SAVE] {args.output_dir / f'{stem}.md'}")


if __name__ == "__main__":
    main()
