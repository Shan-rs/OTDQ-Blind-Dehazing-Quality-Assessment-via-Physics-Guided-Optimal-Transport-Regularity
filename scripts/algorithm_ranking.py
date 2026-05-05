#!/usr/bin/env python3
"""Aggregate sample scores to algorithm-level rankings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from otdq_eval.stats import (
    DEFAULT_LOWER_IS_BETTER,
    add_refq,
    algorithm_mean_scores,
    ranking_consistency,
    write_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm-level ranking consistency.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--algorithm-col", default="algorithm")
    parser.add_argument("--metric-cols", nargs="+", required=True)
    parser.add_argument("--reference-cols", nargs="+", required=True)
    parser.add_argument("--lower-is-better", nargs="*", default=sorted(DEFAULT_LOWER_IS_BETTER))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    df = add_refq(pd.read_csv(args.csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = algorithm_mean_scores(
        df,
        metrics=args.metric_cols,
        references=args.reference_cols,
        dataset_col=args.dataset_col,
        algorithm_col=args.algorithm_col,
        lower_is_better=args.lower_is_better,
    )
    ranks = ranking_consistency(
        means,
        metrics=args.metric_cols,
        references=args.reference_cols,
        dataset_col=args.dataset_col,
        algorithm_col=args.algorithm_col,
        top_k=args.top_k,
    )
    for stem, table in [
        ("algorithm_mean_scores", means),
        ("algorithm_level_ranking", ranks),
    ]:
        write_table(table, args.output_dir / f"{stem}.csv", args.digits)
        write_table(table, args.output_dir / f"{stem}.md", args.digits)
        print(f"[SAVE] {args.output_dir / f'{stem}.csv'}")
        print(f"[SAVE] {args.output_dir / f'{stem}.md'}")


if __name__ == "__main__":
    main()
