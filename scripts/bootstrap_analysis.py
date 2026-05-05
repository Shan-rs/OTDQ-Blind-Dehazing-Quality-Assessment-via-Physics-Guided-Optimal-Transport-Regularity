#!/usr/bin/env python3
"""Group bootstrap confidence intervals for MOS or paired-reference evaluations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from otdq_eval.stats import (
    DEFAULT_LOWER_IS_BETTER,
    pairwise_accuracy,
    plcc_rmse_after_logistic,
    quality_values,
    safe_corr,
    write_table,
)


STAT_ORDER = ["SRCC", "KRCC", "PLCC", "RMSE", "Weighted_Acc", "Pairwise_Acc"]


def metric_stats(df: pd.DataFrame, metric: str, target_col: str, group_col: str, lower_is_better) -> Dict[str, float]:
    q = quality_values(df, metric, lower_is_better)
    target = quality_values(df, target_col, lower_is_better)
    valid = q.notna() & target.notna()
    plcc, rmse = plcc_rmse_after_logistic(q[valid], target[valid])
    pair = pairwise_accuracy(
        df,
        metrics=[metric],
        references=[target_col],
        group_col=group_col,
        lower_is_better=lower_is_better,
        thresholds=[],
    )
    pair_row = pair.iloc[0] if not pair.empty else {}
    return {
        "SRCC": safe_corr(q[valid], target[valid], "srcc"),
        "KRCC": safe_corr(q[valid], target[valid], "krcc"),
        "PLCC": plcc,
        "RMSE": rmse,
        "Weighted_Acc": float(pair_row.get("Weighted_Acc", np.nan)),
        "Pairwise_Acc": float(pair_row.get("Pairwise_Acc", np.nan)),
    }


def bootstrap_sample(groups: Dict[str, pd.DataFrame], draw: np.ndarray) -> pd.DataFrame:
    parts = []
    keys = list(groups)
    for repeat_id, group_index in enumerate(draw):
        part = groups[keys[int(group_index)]].copy()
        part["_bootstrap_group"] = f"{repeat_id:06d}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "Bootstrap_Mean": float(np.mean(arr)) if arr.size else np.nan,
        "CI95_Low": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
        "CI95_High": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
        "N_Boot": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-bootstrap confidence intervals.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bootstrap"))
    parser.add_argument("--group-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--metric-cols", nargs="+", required=True)
    parser.add_argument("--lower-is-better", nargs="*", default=sorted(DEFAULT_LOWER_IS_BETTER))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--competitor", default="auto", help="'auto' compares OTDQ with the best non-OTDQ metric per statistic.")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.group_col, args.target_col]).copy()
    groups = {str(name): group.copy() for name, group in df.groupby(args.group_col, sort=True)}
    if not groups:
        raise ValueError("No non-empty bootstrap groups were found.")

    point_rows = []
    point_stats: Dict[tuple[str, str], float] = {}
    for metric in args.metric_cols:
        stats = metric_stats(df, metric, args.target_col, args.group_col, args.lower_is_better)
        for stat, value in stats.items():
            point_stats[(metric, stat)] = value

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(groups), size=(args.n_boot, len(groups)), endpoint=False)
    boot_values: Dict[tuple[str, str], List[float]] = {
        (metric, stat): []
        for metric in args.metric_cols
        for stat in STAT_ORDER
    }
    for idx, draw in enumerate(draws, 1):
        boot_df = bootstrap_sample(groups, draw)
        for metric in args.metric_cols:
            stats = metric_stats(boot_df, metric, args.target_col, "_bootstrap_group", args.lower_is_better)
            for stat, value in stats.items():
                boot_values[(metric, stat)].append(value)
        if idx == 1 or idx % max(args.n_boot // 10, 1) == 0 or idx == args.n_boot:
            print(f"[BOOT] {idx}/{args.n_boot}")

    ci_rows = []
    for metric in args.metric_cols:
        for stat in STAT_ORDER:
            ci_rows.append(
                {
                    "Metric": metric,
                    "Statistic": stat,
                    "Point": point_stats[(metric, stat)],
                    **summarize(boot_values[(metric, stat)]),
                    "Requested_Boot": args.n_boot,
                }
            )
    ci_df = pd.DataFrame(ci_rows)

    diff_rows = []
    if "otdq" in {m.lower() for m in args.metric_cols}:
        otdq_metric = next(m for m in args.metric_cols if m.lower() == "otdq")
        for stat in STAT_ORDER:
            if args.competitor == "auto":
                competitors = [m for m in args.metric_cols if m != otdq_metric]
                if not competitors:
                    continue
                reverse = stat != "RMSE"
                competitor = sorted(
                    competitors,
                    key=lambda m: point_stats[(m, stat)] if np.isfinite(point_stats[(m, stat)]) else (-np.inf if reverse else np.inf),
                    reverse=reverse,
                )[0]
            else:
                competitor = args.competitor
                if competitor not in args.metric_cols:
                    continue
            if stat == "RMSE":
                point_adv = point_stats[(competitor, stat)] - point_stats[(otdq_metric, stat)]
                diffs = np.asarray(boot_values[(competitor, stat)]) - np.asarray(boot_values[(otdq_metric, stat)])
                orientation = "competitor - OTDQ"
            else:
                point_adv = point_stats[(otdq_metric, stat)] - point_stats[(competitor, stat)]
                diffs = np.asarray(boot_values[(otdq_metric, stat)]) - np.asarray(boot_values[(competitor, stat)])
                orientation = "OTDQ - competitor"
            diffs = diffs[np.isfinite(diffs)]
            diff_rows.append(
                {
                    "Statistic": stat,
                    "Competitor": competitor,
                    "OTDQ_Point": point_stats[(otdq_metric, stat)],
                    "Competitor_Point": point_stats[(competitor, stat)],
                    "OTDQ_Advantage_Point": point_adv,
                    "Advantage_Mean": float(np.mean(diffs)) if diffs.size else np.nan,
                    "CI95_Low": float(np.percentile(diffs, 2.5)) if diffs.size else np.nan,
                    "CI95_High": float(np.percentile(diffs, 97.5)) if diffs.size else np.nan,
                    "P_OTDQ_Better": float(np.mean(diffs > 0)) if diffs.size else np.nan,
                    "N_Boot": int(diffs.size),
                    "Diff_Orientation": orientation,
                }
            )
    diff_df = pd.DataFrame(diff_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, table in [
        ("bootstrap_ci", ci_df),
        ("bootstrap_paired_diff", diff_df),
    ]:
        write_table(table, args.output_dir / f"{stem}.csv", args.digits)
        write_table(table, args.output_dir / f"{stem}.md", args.digits)
        print(f"[SAVE] {args.output_dir / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
