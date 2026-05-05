#!/usr/bin/env python3
"""Compute OTDQ and consistency tables for paired dehazing benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from otdq_eval.config import merged_otdq_config
from otdq_eval.io import read_manifest
from otdq_eval.scoring import score_records
from otdq_eval.stats import (
    DEFAULT_LOWER_IS_BETTER,
    add_refq,
    algorithm_mean_scores,
    pairwise_accuracy,
    ranking_consistency,
    safe_corr,
    write_table,
)


def sample_reference_table(df: pd.DataFrame, metrics, references, lower_is_better):
    from otdq_eval.stats import normalize_metric_name, quality_values

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        q = quality_values(df, metric, lower_is_better)
        for reference in references:
            if reference not in df.columns:
                continue
            ref = quality_values(df, reference, lower_is_better)
            valid = q.notna() & ref.notna()
            rows.append(
                {
                    "Reference": normalize_metric_name(reference),
                    "Metric": normalize_metric_name(metric),
                    "N": int(valid.sum()),
                    "SRCC": safe_corr(q[valid], ref[valid], "srcc"),
                    "KRCC": safe_corr(q[valid], ref[valid], "krcc"),
                    "PLCC": safe_corr(q[valid], ref[valid], "plcc"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired benchmark OTDQ evaluation.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with dataset, algorithm, image_id, paths, and optional FR metrics.")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paired"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paired_benchmark.json")
    parser.add_argument("--id-col", default="image_id")
    parser.add_argument("--hazy-col", default="hazy_path")
    parser.add_argument("--dehazed-col", default="dehazed_path")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--algorithm-col", default="algorithm")
    parser.add_argument("--metric-cols", nargs="*", default=["otdq"])
    parser.add_argument("--reference-cols", nargs="*", default=["PSNR", "SSIM", "LPIPS_Q", "RefQ"])
    parser.add_argument("--lower-is-better", nargs="*", default=sorted(DEFAULT_LOWER_IS_BETTER))
    parser.add_argument("--skip-scoring", action="store_true", help="Use existing metric columns in the manifest.")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    config = merged_otdq_config(args.config)
    records = read_manifest(
        args.manifest,
        base_dir=args.base_dir,
        id_col=args.id_col,
        hazy_col=args.hazy_col,
        dehazed_col=args.dehazed_col,
    )
    input_df = pd.DataFrame(records)
    if args.skip_scoring:
        scored = input_df.copy()
    else:
        scored = pd.DataFrame(score_records(records, resize_method=config["resize_method"], config=config))

    scored = add_refq(scored)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "paired_scores_with_otdq.csv"
    scored.to_csv(scores_path, index=False)
    print(f"[SAVE] {scores_path}")

    references = [ref for ref in args.reference_cols if ref in scored.columns]
    metrics = [metric for metric in args.metric_cols if metric in scored.columns]
    if not references or not metrics:
        print("[INFO] No reference/metric overlap found; only score CSV was written.")
        return

    group_col = args.id_col if args.id_col in scored.columns else "id"
    sample = sample_reference_table(scored, metrics, references, args.lower_is_better)
    pair = pairwise_accuracy(
        scored,
        metrics=metrics,
        references=references,
        group_col=group_col,
        lower_is_better=args.lower_is_better,
    )
    for stem, table in [
        ("table1_sample_level_correlations", sample),
        ("table2_pairwise_ranking_accuracy", pair),
    ]:
        write_table(table, args.output_dir / f"{stem}.csv", args.digits)
        write_table(table, args.output_dir / f"{stem}.md", args.digits)
        print(f"[SAVE] {args.output_dir / f'{stem}.csv'}")

    if args.algorithm_col in scored.columns:
        means = algorithm_mean_scores(
            scored,
            metrics=metrics,
            references=references,
            dataset_col=args.dataset_col,
            algorithm_col=args.algorithm_col,
            lower_is_better=args.lower_is_better,
        )
        ranks = ranking_consistency(
            means,
            metrics=metrics,
            references=references,
            dataset_col=args.dataset_col,
            algorithm_col=args.algorithm_col,
        )
        for stem, table in [
            ("table3_algorithm_mean_scores", means),
            ("table4_algorithm_level_ranking", ranks),
        ]:
            write_table(table, args.output_dir / f"{stem}.csv", args.digits)
            write_table(table, args.output_dir / f"{stem}.md", args.digits)
            print(f"[SAVE] {args.output_dir / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
