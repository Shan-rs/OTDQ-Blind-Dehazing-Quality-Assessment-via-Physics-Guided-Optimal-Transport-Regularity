#!/usr/bin/env python3
"""Compute within-group pairwise ranking accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from otdq_eval.stats import DEFAULT_LOWER_IS_BETTER, pairwise_accuracy, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise ranking accuracy for objective metrics.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/pairwise_ranking.csv"))
    parser.add_argument("--group-col", required=True, help="Compare pairs only within this group.")
    parser.add_argument("--metric-cols", nargs="+", required=True)
    parser.add_argument("--reference-cols", nargs="+", required=True)
    parser.add_argument("--lower-is-better", nargs="*", default=sorted(DEFAULT_LOWER_IS_BETTER))
    parser.add_argument("--thresholds", type=float, nargs="*", default=[5.0, 10.0, 15.0])
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    table = pairwise_accuracy(
        df,
        metrics=args.metric_cols,
        references=args.reference_cols,
        group_col=args.group_col,
        lower_is_better=args.lower_is_better,
        thresholds=args.thresholds,
    )
    write_table(table, args.output, args.digits)
    if args.output.suffix.lower() != ".md":
        write_table(table, args.output.with_suffix(".md"), args.digits)
    print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
