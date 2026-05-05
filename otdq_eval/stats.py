"""Statistics used by the OTDQ-Eval scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


DEFAULT_LOWER_IS_BETTER = {
    "niqe",
    "brisque",
    "fade",
    "dark_channel",
    "lpips",
}


def normalize_metric_name(metric: str) -> str:
    name = str(metric)
    if name.startswith("NR_"):
        name = name[3:]
    if name.startswith("Average_NR_"):
        name = name[len("Average_NR_") :]
    if name.startswith("Average_"):
        name = name[len("Average_") :]
    return name


def metric_key(metric: str) -> str:
    return normalize_metric_name(metric).lower()


def quality_values(
    df: pd.DataFrame,
    column: str,
    lower_is_better: Iterable[str] = DEFAULT_LOWER_IS_BETTER,
) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    lower = {metric_key(item) for item in lower_is_better}
    if metric_key(column) in lower:
        values = -values
    return values


def safe_corr(x: Iterable[float], y: Iterable[float], kind: str) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3:
        return np.nan
    if np.nanstd(x_arr) < 1e-12 or np.nanstd(y_arr) < 1e-12:
        return np.nan
    if kind == "srcc":
        return float(spearmanr(x_arr, y_arr).correlation)
    if kind == "krcc":
        return float(kendalltau(x_arr, y_arr).correlation)
    if kind == "plcc":
        return float(pearsonr(x_arr, y_arr).statistic)
    raise ValueError(f"Unknown correlation kind: {kind}")


def logistic_func(x, beta1, beta2, beta3, beta4, beta5):
    z = np.clip(beta2 * (x - beta3), -500, 500)
    return beta1 * (0.5 - 1 / (1 + np.exp(z))) + beta4 * x + beta5


def plcc_rmse_after_logistic(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 5:
        return np.nan, np.nan
    if np.nanstd(x_arr) < 1e-12 or np.nanstd(y_arr) < 1e-12:
        return np.nan, np.nan

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    y_range = float(np.max(y_arr) - np.min(y_arr))
    x_range = float(np.max(x_arr) - np.min(x_arr) + 1e-9)
    initial_guess = [
        y_range / 2,
        5.0,
        x_mean,
        y_range / x_range,
        y_mean - (y_range / x_range) * x_mean,
    ]
    lower = [-np.inf, 0.1, float(np.min(x_arr)), -np.inf, -np.inf]
    upper = [np.inf, 50.0, float(np.max(x_arr)), np.inf, np.inf]

    try:
        popt, _ = curve_fit(
            logistic_func,
            x_arr,
            y_arr,
            p0=initial_guess,
            bounds=(lower, upper),
            maxfev=20000,
        )
        mapped = logistic_func(x_arr, *popt)
    except Exception:
        slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
        mapped = slope * x_arr + intercept

    return safe_corr(mapped, y_arr, "plcc"), float(np.sqrt(np.mean((mapped - y_arr) ** 2)))


def correlation_table(
    df: pd.DataFrame,
    metrics: Sequence[str],
    target_col: str,
    *,
    lower_is_better: Iterable[str] = DEFAULT_LOWER_IS_BETTER,
) -> pd.DataFrame:
    rows = []
    target = pd.to_numeric(df[target_col], errors="coerce")
    for metric in metrics:
        q = quality_values(df, metric, lower_is_better)
        valid = q.notna() & target.notna()
        plcc, rmse = plcc_rmse_after_logistic(q[valid], target[valid])
        rows.append(
            {
                "Metric": normalize_metric_name(metric),
                "N": int(valid.sum()),
                "SRCC": safe_corr(q[valid], target[valid], "srcc"),
                "KRCC": safe_corr(q[valid], target[valid], "krcc"),
                "PLCC": plcc,
                "RMSE": rmse,
            }
        )
    return pd.DataFrame(rows)


def pairwise_accuracy(
    df: pd.DataFrame,
    metrics: Sequence[str],
    references: Sequence[str],
    group_col: str,
    *,
    lower_is_better: Iterable[str] = DEFAULT_LOWER_IS_BETTER,
    thresholds: Sequence[float] = (5.0, 10.0, 15.0),
) -> pd.DataFrame:
    rows = []
    for reference in references:
        ref_values = quality_values(df, reference, lower_is_better)
        for metric in metrics:
            metric_values = quality_values(df, metric, lower_is_better)
            totals = {f"Acc_{threshold:g}": [0.0, 0.0] for threshold in thresholds}
            weighted_correct = 0.0
            weighted_total = 0.0
            all_correct = 0.0
            all_total = 0

            tmp = df[[group_col]].copy()
            tmp["reference"] = ref_values
            tmp["metric"] = metric_values
            tmp = tmp.dropna(subset=["reference", "metric"])
            for _, group in tmp.groupby(group_col, sort=True):
                records = group[["reference", "metric"]].to_dict("records")
                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        ref_diff = records[i]["reference"] - records[j]["reference"]
                        weight = abs(float(ref_diff))
                        if weight < 1e-12:
                            continue
                        pred_diff = records[i]["metric"] - records[j]["metric"]
                        if abs(pred_diff) < 1e-12:
                            correct = 0.5
                        elif ref_diff * pred_diff > 0:
                            correct = 1.0
                        else:
                            correct = 0.0
                        all_correct += correct
                        all_total += 1
                        weighted_correct += correct * weight
                        weighted_total += weight
                        for threshold in thresholds:
                            if weight >= threshold:
                                key = f"Acc_{threshold:g}"
                                totals[key][0] += correct
                                totals[key][1] += 1.0

            row: Dict[str, object] = {
                "Reference": normalize_metric_name(reference),
                "Metric": normalize_metric_name(metric),
                "Pairwise_Acc": all_correct / all_total if all_total else np.nan,
                "Weighted_Acc": weighted_correct / weighted_total if weighted_total else np.nan,
                "NumPairs": int(all_total),
            }
            for key, (correct, total) in totals.items():
                row[key] = correct / total if total else np.nan
                row[f"NumPairs_{key[4:]}"] = int(total)
            rows.append(row)
    return pd.DataFrame(rows)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return values * np.nan
    return (values - values.mean()) / std


def add_refq(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"PSNR", "SSIM", "LPIPS"}.issubset(out.columns):
        out["LPIPS_Q"] = -pd.to_numeric(out["LPIPS"], errors="coerce")
        out["RefQ"] = (zscore(out["PSNR"]) + zscore(out["SSIM"]) + zscore(out["LPIPS_Q"])) / 3.0
    return out


def algorithm_mean_scores(
    df: pd.DataFrame,
    metrics: Sequence[str],
    references: Sequence[str],
    *,
    dataset_col: str = "dataset",
    algorithm_col: str = "algorithm",
    lower_is_better: Iterable[str] = DEFAULT_LOWER_IS_BETTER,
) -> pd.DataFrame:
    rows = []
    group_cols = [col for col in [dataset_col, algorithm_col] if col in df.columns]
    if algorithm_col not in group_cols:
        raise ValueError(f"Missing algorithm column: {algorithm_col}")
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["NumImages"] = int(len(group))
        for column in [*references, *metrics]:
            if column in group.columns:
                row[normalize_metric_name(column)] = quality_values(group, column, lower_is_better).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rank_items(df: pd.DataFrame, score_col: str, algorithm_col: str = "algorithm") -> List[str]:
    sub = df.dropna(subset=[score_col]).sort_values([score_col, algorithm_col], ascending=[False, True])
    return sub[algorithm_col].astype(str).tolist()


def ranking_consistency(
    mean_df: pd.DataFrame,
    metrics: Sequence[str],
    references: Sequence[str],
    *,
    dataset_col: str = "dataset",
    algorithm_col: str = "algorithm",
    top_k: int = 3,
) -> pd.DataFrame:
    rows = []
    group_cols = [dataset_col] if dataset_col in mean_df.columns else [None]
    groups = mean_df.groupby(dataset_col, sort=True) if dataset_col in mean_df.columns else [(None, mean_df)]
    for dataset, group in groups:
        for reference in references:
            ref_col = normalize_metric_name(reference)
            if ref_col not in group.columns:
                continue
            for metric in metrics:
                metric_col = normalize_metric_name(metric)
                if metric_col not in group.columns:
                    continue
                sub = group.dropna(subset=[ref_col, metric_col])
                ref_order = rank_items(sub, ref_col, algorithm_col)
                pred_order = rank_items(sub, metric_col, algorithm_col)
                ref_top = set(ref_order[:top_k])
                pred_top = set(pred_order[:top_k])
                row = {
                    "Reference_Ranking": ref_col,
                    "Metric": metric_col,
                    "NumMethods": int(len(sub)),
                    "Rank_SRCC": safe_corr(sub[metric_col], sub[ref_col], "srcc"),
                    "Rank_KRCC": safe_corr(sub[metric_col], sub[ref_col], "krcc"),
                    f"Top{top_k}_Overlap": len(ref_top & pred_top) / float(top_k) if len(sub) >= top_k else np.nan,
                    f"Reference_Top{top_k}": ";".join(ref_order[:top_k]),
                    f"Metric_Top{top_k}": ";".join(pred_order[:top_k]),
                }
                if dataset_col in mean_df.columns:
                    row[dataset_col] = dataset
                rows.append(row)
    return pd.DataFrame(rows)


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: round(float(x), digits) if np.isfinite(x) else x)
    return out


def write_table(df: pd.DataFrame, path: str | Path, digits: int = 4) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounded = round_numeric(df, digits)
    if out_path.suffix.lower() == ".md":
        out_path.write_text(rounded.to_markdown(index=False), encoding="utf-8")
    else:
        rounded.to_csv(out_path, index=False)
