"""Diagnostic maps for OTDQ figure reproduction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from .metric import _as_float01, _patch_stats, compute_otdq


def transport_irregularity_map(hazy: np.ndarray, dehazed: np.ndarray, n_q: int, var_thresh: float):
    short_edge = min(hazy.shape[:2])
    patch_size = max(16, min(64, short_edge // 8))
    height, width = hazy.shape[:2]
    out = np.zeros((height, width), dtype=np.float64)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            hp = hazy[i : i + patch_size, j : j + patch_size, :]
            dp = dehazed[i : i + patch_size, j : j + patch_size, :]
            if hp.shape[0] < 4 or hp.shape[1] < 4:
                continue
            patch_r2s = []
            for channel in range(3):
                if float(np.var(hp[:, :, channel])) < var_thresh:
                    continue
                r2, _ = _patch_stats(hp[:, :, channel], dp[:, :, channel], n_q)
                patch_r2s.append(r2)
            out[i : i + patch_size, j : j + patch_size] = 1.0 - float(np.mean(patch_r2s) if patch_r2s else 1.0)
    return np.clip(out, 0.0, 1.0), patch_size


def structure_inconsistency_map(
    hazy: np.ndarray,
    dehazed: np.ndarray,
    flat_alpha: float,
    window_size: int,
    texture_pct: float,
) -> np.ndarray:
    gray_h = np.mean(hazy, axis=2)
    gray_d = np.mean(dehazed, axis=2)
    gx_h, gy_h = np.gradient(gray_h)
    gx_d, gy_d = np.gradient(gray_d)
    norm_h = np.sqrt(gx_h ** 2 + gy_h ** 2)
    norm_d = np.sqrt(gx_d ** 2 + gy_d ** 2)
    threshold = max(float(np.percentile(norm_h, np.clip(texture_pct, 0.0, 100.0))), 0.005)
    textured = norm_h > threshold

    dot = gx_h * gx_d + gy_h * gy_d
    cos_sim = np.clip(dot / ((norm_h + 1e-8) * (norm_d + 1e-8)), -1.0, 1.0)
    direction_failure = 1.0 - np.exp(3.0 * (cos_sim - 1.0))

    ratio = (norm_d + 1e-8) / (norm_h + 1e-8)
    log_ratio = np.log(np.clip(ratio, 0.1, 10.0))
    size = max(int(window_size), 3)
    local_mean = uniform_filter(log_ratio, size=size)
    local_sq_mean = uniform_filter(log_ratio ** 2, size=size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    lcec_failure = 1.0 - np.exp(-2.0 * local_var)

    nz = norm_h[norm_h > 1e-6]
    g_ref = float(np.median(nz)) if nz.size else 1e-3
    flat_failure = 1.0 - np.exp(-max(float(flat_alpha), 0.0) * (norm_d / (g_ref + 1e-8)) ** 2)
    out = np.where(textured, np.maximum(direction_failure, lcec_failure), flat_failure)
    return np.clip(out, 0.0, 1.0)


def artifact_failure_maps(hazy: np.ndarray, dehazed: np.ndarray, halo_pct: float, patch_window: int) -> Dict[str, np.ndarray]:
    gray_h = np.mean(hazy, axis=2)
    gray_d = np.mean(dehazed, axis=2)
    gy_h, gx_h = np.gradient(gray_h)
    grad_h = np.sqrt(gx_h ** 2 + gy_h ** 2)
    gy_d, gx_d = np.gradient(gray_d)
    grad_d = np.sqrt(gx_d ** 2 + gy_d ** 2)

    edge_thresh = np.percentile(grad_h, float(np.clip(halo_pct, 0.0, 100.0)))
    edge_mask = grad_h > edge_thresh
    halo_region = binary_dilation(edge_mask, structure=np.ones((7, 7), dtype=bool)) & ~edge_mask
    halo_map = np.zeros_like(gray_h, dtype=np.float64)
    if int(np.sum(halo_region)) > 100:
        denom = float(np.mean(grad_h[halo_region]) + 1e-6)
        halo_map[halo_region] = np.maximum(0.0, grad_d[halo_region] / denom - 2.0) / 3.0
        halo_map = uniform_filter(halo_map, size=3)

    dark_map = np.zeros_like(gray_h, dtype=np.float64)
    bright_mask = gray_h > 0.7
    dark_map[bright_mask] = np.maximum(0.0, gray_h[bright_mask] - gray_d[bright_mask] - 0.3) / 0.4

    diff = np.abs(gray_d - gray_h)
    local_mu = uniform_filter(diff, size=max(int(patch_window), 3))
    deviation = np.abs(local_mu - float(np.median(local_mu)))
    scale = max(float(np.percentile(deviation, 99.0)), 1e-8)
    patch_map = deviation / scale

    return {
        "halo": np.clip(halo_map, 0.0, 1.0),
        "dark": np.clip(dark_map, 0.0, 1.0),
        "patch": np.clip(patch_map, 0.0, 1.0),
        "artifact": np.clip(np.maximum.reduce([halo_map, dark_map, patch_map]), 0.0, 1.0),
    }


def diagnostic_maps(hazy_u8: np.ndarray, dehazed_u8: np.ndarray, **params):
    hazy = _as_float01(hazy_u8, "hazy")
    dehazed = _as_float01(dehazed_u8, "dehazed")
    scores = compute_otdq(hazy, dehazed, **params, verbose=False)
    tr_map, patch_size = transport_irregularity_map(
        hazy,
        dehazed,
        int(params.get("n_q", 1000)),
        float(params.get("var_thresh", 1e-4)),
    )
    str_map = structure_inconsistency_map(
        hazy,
        dehazed,
        float(params.get("flat_alpha", 5.0)),
        int(params.get("window_size", 15)),
        float(params.get("texture_pct", 40.0)),
    )
    arti = artifact_failure_maps(
        hazy,
        dehazed,
        float(params.get("halo_pct", 90.0)),
        int(params.get("patch_window", 32)),
    )
    scores["diagnostic_patch_size"] = patch_size
    return scores, {
        "transport": tr_map,
        "structure": str_map,
        "artifact": arti["artifact"],
        "halo": arti["halo"],
        "dark": arti["dark"],
        "patch": arti["patch"],
    }
