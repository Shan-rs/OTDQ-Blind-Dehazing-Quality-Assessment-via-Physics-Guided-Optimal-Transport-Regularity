"""
Optimal Transport Dehazing Quality Index (OTDQ)
================================================================================
A continuous, mathematically rigorous no-reference IQA for image dehazing.

Orthogonal Dimensions:
  S_vis : Visibility & Contrast Recovery (OT mapping regularity)
  S_str : Structural Spatial Fidelity (gradient direction + local LCEC)
  S_arti: Artifact Suppression (halo, over-darkening, patchiness)


Final Fusion (power-weighted geometric mean):
  OTDQ = S_vis^1.00 * S_str^0.50 * S_arti^0.50
"""

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from typing import Dict, Tuple


OTDQ_WEIGHTS = dict(s_vis=1.00, s_str=0.50, s_arti=0.50)


def _as_float01(img: np.ndarray, name: str) -> np.ndarray:
    """Validate an RGB image and convert it to float64 in [0, 1]."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{name} contains NaN or Inf values")

    if img.dtype == np.uint8:
        img_f = img.astype(np.float64) / 255.0
    else:
        img_f = img.astype(np.float64)
        if img_f.size and img_f.max() > 2.0:
            raise ValueError(
                f"{name} appears to be a float image outside [0, 1]. "
                "Pass uint8 [0,255] or normalize float inputs before calling."
            )

    return np.clip(img_f, 0.0, 1.0)

# ============================================================
# 1. Atmospheric Light
# ============================================================

def estimate_atmospheric_light(hazy: np.ndarray, min_size: int = 15,
                                dark_channel: np.ndarray = None) -> np.ndarray:
    min_size = max(int(min_size), 1)
    gray = dark_channel if dark_channel is not None else hazy.min(axis=2)
    def _qt(img_r, gray_r):
        h, w = gray_r.shape
        if h <= min_size or w <= min_size: return img_r
        mh, mw = h // 2, w // 2
        slices = [(slice(None, mh), slice(None, mw)), (slice(None, mh), slice(mw, None)),
                  (slice(mh, None), slice(None, mw)), (slice(mh, None), slice(mw, None))]
        best = max(range(4), key=lambda i: gray_r[slices[i]].mean())
        return _qt(img_r[slices[best]], gray_r[slices[best]])

    region = _qt(hazy, gray)
    flat_gray = np.mean(region, axis=2).ravel()
    n_top = max(int(flat_gray.size * 0.1), 1)
    idx = np.argpartition(flat_gray, -n_top)[-n_top:]
    A = np.median(region.reshape(-1, 3)[idx], axis=0)
    return np.clip(A, 0.05, 1.0)

# ============================================================
# 2. S_vis: Visibility & Contrast Recovery
# ============================================================

def _patch_stats(hy_ch, dx_ch, n_q=1000):
    hy_s, dx_s = np.sort(hy_ch.ravel()), np.sort(dx_ch.ravel())
    n = len(hy_s)
    n_q = max(int(n_q), 2)
    idx = np.linspace(0, n - 1, min(n_q, n)).astype(int)
    y, x = hy_s[idx].astype(np.float64), dx_s[idx].astype(np.float64)

    ym, xm = y.mean(), x.mean()
    ss_yy = np.sum((y - ym) ** 2)
    if ss_yy < 1e-12: return 0.0, 1.0
    a = np.sum((y - ym) * (x - xm)) / ss_yy
    b = xm - a * ym
    ss_res = np.sum((x - (a * y + b)) ** 2)
    ss_tot = np.sum((x - xm) ** 2)
    r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0, 1))
    return r2, float(a)

def compute_visibility_consistency(hazy, dehazed, hazy_dc, A, n_q=1000, var_thresh=1e-4):
    """
    S_vis = TR (Transport Regularity) * SF (Slope Factor) * SDC_factor
    """
    short_edge = min(hazy.shape[:2])
    patch_size = max(16, min(64, short_edge // 8))
    H, W = hazy.shape[:2]
    var_thresh = max(float(var_thresh), 0.0)

    r2_acc, sl_acc, wt_acc = [[], [], []], [[], [], []], [[], [], []]
    sdc_tr_list, fog_w = [], []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            hp = hazy[i:i+patch_size, j:j+patch_size, :]
            dp = dehazed[i:i+patch_size, j:j+patch_size, :]
            if hp.shape[0] < 4 or hp.shape[1] < 4: continue

            dark_ch = hazy_dc[i:i+patch_size, j:j+patch_size]
            diff_to_A = np.linalg.norm(hp - A[np.newaxis, np.newaxis, :], axis=2)
            fw = float(np.mean(dark_ch * np.exp(-5.0 * diff_to_A)))

            patch_r2s = []
            for c in range(3):
                var_c = float(np.var(hp[:,:,c]))
                if var_c < var_thresh: continue
                r2, slope = _patch_stats(hp[:,:,c], dp[:,:,c], n_q)
                r2_acc[c].append(r2); sl_acc[c].append(slope); wt_acc[c].append(np.sqrt(var_c))
                patch_r2s.append(r2)

            if patch_r2s and fw >= 1e-4:
                sdc_tr_list.append(float(np.mean(patch_r2s)))
                fog_w.append(fw)

    r2_ch, sl_ch = [], []
    for c in range(3):
        if not r2_acc[c]:
            r2_c, a_c = _patch_stats(hazy[:,:,c], dehazed[:,:,c], n_q)
        else:
            w = np.array(wt_acc[c])
            w_sum = w.sum()
            w = w / w_sum if w_sum > 1e-12 else np.ones_like(w) / len(w)
            r2_c, a_c = float(np.dot(w, r2_acc[c])), float(np.dot(w, sl_acc[c]))
        r2_ch.append(r2_c); sl_ch.append(a_c)

    tr = float(np.mean(r2_ch))

    ms = float(np.mean(sl_ch))
    sf = float(np.exp(-0.2 * max(0, 1.0 - ms)**2 - 0.05 * max(0, ms - 2.5)**2))

    if not sdc_tr_list: sdc = 1.0
    else:
        fw_arr = np.array(fog_w); fw_arr /= fw_arr.sum()
        sdc = float(np.dot(fw_arr, sdc_tr_list))

    sdc_factor = 0.5 + 0.5 * sdc
    S_vis = float(np.clip(tr * sf * sdc_factor, 1e-6, 1.0))
    return S_vis, dict(tr=tr, sf=sf, sdc=sdc, sdc_factor=sdc_factor,
                       patch_size=patch_size)

# ============================================================
# 3. S_str: Structural Spatial Fidelity
# ============================================================

def compute_structural_fidelity(hazy, dehazed,
                                 flat_alpha: float = 5.0,
                                 window_size: int = 15,
                                 texture_pct: float = 40.0):
    """
    S_str: 梯度方向一致性 + 局部透射率平滑度 + 平坦区幻觉惩罚

    物理基础（ASM）：J = (I−A)/t + A
      ∇J = ∇I / t   ← 梯度方向完全保留，幅值仅被 1/t 缩放（尺度不变）

    · 有纹理区：两项子分量的几何均值
        ① 方向余弦（全图矩阵化，再掩码取有纹理像素）
           score_dir = exp(3 × (cos θ − 1))
        ② 局部 LCEC（透射率平滑先验）：norm_d/norm_h ≈ 1/t，
           在 window_size×window_size 局部窗口内局部方差应小。
           使用局部方差（而非全局方差）：真实 3D 场景的透射率因景深
           变化在全局天然高方差；分块/GAN伪影的局部方差更大。
           score_lcec = exp(−2 × local_var(log(ratio)))

    · 平坦区（|∇I| ≤ 阈值）：幻觉惩罚
        好的去雾：平坦区 ∇J 也接近 0；GAN 幻觉：∇J 显著
        score_flat = exp(−flat_alpha × (norm_d / g_ref)²)

    阈值设计：相对（texture_pct 百分位） + 绝对保底（0.005）
      防止大片天空/平坦雾区的传感器噪声梯度被误分为"有纹理"，
      导致随机方向拉低 cos_score。
    """
    hg = np.mean(hazy,    axis=2)
    dg = np.mean(dehazed, axis=2)
    flat_alpha = max(float(flat_alpha), 0.0)
    window_size = max(int(window_size), 3)

    gx_h, gy_h = np.gradient(hg)
    gx_d, gy_d = np.gradient(dg)

    norm_h = np.sqrt(gx_h ** 2 + gy_h ** 2)
    norm_d = np.sqrt(gx_d ** 2 + gy_d ** 2)

    # 自适应阈值 + 绝对噪声底线：防止大片平坦区噪声被误判为纹理
    texture_pct = float(np.clip(texture_pct, 0.0, 100.0))
    base_thresh     = float(np.percentile(norm_h, texture_pct))
    grad_thresh_abs = max(base_thresh, 0.005)   # 0.005 ≈ [0,1] 归一化图像的噪声底

    textured = norm_h > grad_thresh_abs
    flat     = ~textured

    nz = norm_h[norm_h > 1e-6]
    g_ref = float(np.median(nz)) if nz.size > 0 else 1e-3

    n_tex = int(np.sum(textured))
    n_flt = int(np.sum(flat))

    # ── 有纹理区：方向余弦 + 局部 LCEC ──────────────────────────────────────
    if n_tex > 100:
        # 子分量 1：梯度方向余弦（全图矩阵化，再掩码）
        dot         = gx_h * gx_d + gy_h * gy_d
        cos_sim_map = np.clip(dot / ((norm_h + 1e-8) * (norm_d + 1e-8)), -1.0, 1.0)
        cos_score   = float(np.mean(np.exp(3.0 * (cos_sim_map[textured] - 1.0))))

        # 子分量 2：局部 LCEC（透射率平滑先验）
        # Var(X) = E[X²] − E[X]²，用 uniform_filter 快速计算局部均值
        ratio_map     = (norm_d + 1e-8) / (norm_h + 1e-8)
        log_ratio_map = np.log(np.clip(ratio_map, 0.1, 10.0))
        local_mean    = uniform_filter(log_ratio_map,      size=window_size)
        local_sq_mean = uniform_filter(log_ratio_map ** 2, size=window_size)
        local_var     = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        lcec_score    = float(np.mean(np.exp(-2.0 * local_var[textured])))

        texture_score = float(np.sqrt(max(cos_score, 1e-8) * max(lcec_score, 1e-8)))
    else:
        texture_score = 1.0

    # ── 平坦区：幻觉惩罚 ─────────────────────────────────────────────────────
    if n_flt > 100:
        norm_d_rel = norm_d[flat] / (g_ref + 1e-8)
        flat_score = float(np.mean(np.exp(-flat_alpha * norm_d_rel ** 2)))
    else:
        flat_score = 1.0

    n_tot = n_tex + n_flt
    S_str = (n_tex * texture_score + n_flt * flat_score) / max(n_tot, 1)

    return float(np.clip(S_str, 1e-6, 1.0)), dict(
        ssim_structure=S_str,
        n_pixels=n_tot,
        n_textured=n_tex,
        n_flat=n_flt,
        # Backward-compatible alias for DEMO_MOS.py/old CSV consumers.
        n_patches=n_tot,
    )

# ============================================================
# 4. S_arti: Artifact Detection
# ============================================================

def compute_artifact_score(hazy: np.ndarray, dehazed: np.ndarray,
                           halo_pct: float = 90.0,
                           patch_window: int = 32) -> Tuple[float, Dict]:
    """
    S_arti: 检测三类去雾视觉伪影，各子项独立返回，方便消融。

    (a) halo_score  — 强边缘旁的光晕（邻域梯度相对原图放大 >2× 则惩罚）
    (b) dark_score  — 亮区过度变暗（原图亮度 >0.7 的区域均值下降 >0.3 则惩罚）
    (c) patch_score — 去雾分块性（|J−I| 的局部均值图标准差，衡量处理不一致）

    S_arti = (halo_score × dark_score × patch_score)^(1/3)
    """
    gray_h = np.mean(hazy,    axis=2)
    gray_d = np.mean(dehazed, axis=2)

    # ── (a) Halo detection ───────────────────────────────────────────────────
    gy_h, gx_h = np.gradient(gray_h)
    grad_h = np.sqrt(gx_h ** 2 + gy_h ** 2)
    halo_pct = float(np.clip(halo_pct, 0.0, 100.0))
    edge_thresh = np.percentile(grad_h, halo_pct)    # high-gradient pixels = strong edges
    edge_mask = grad_h > edge_thresh

    dilated     = binary_dilation(edge_mask, structure=np.ones((7, 7), dtype=bool))
    halo_region = dilated & ~edge_mask               # 边缘膨胀带 \ 边缘本身

    if np.sum(halo_region) > 100:
        gy_d, gx_d = np.gradient(gray_d)
        grad_d = np.sqrt(gx_d ** 2 + gy_d ** 2)
        ratio = (np.mean(grad_d[halo_region]) /
                 (np.mean(grad_h[halo_region]) + 1e-6))
        halo_score = float(np.exp(-max(0.0, ratio - 2.0)))
    else:
        halo_score = 1.0

    # ── (b) Over-darkening ───────────────────────────────────────────────────
    bright_mask = gray_h > 0.7
    if np.sum(bright_mask) > 100:
        darkening  = float(np.mean(gray_h[bright_mask] - gray_d[bright_mask]))
        dark_score = float(np.exp(-3.0 * max(0.0, darkening - 0.3)))
    else:
        dark_score = 1.0

    # ── (c) Patchiness ───────────────────────────────────────────────────────
    diff      = np.abs(gray_d - gray_h)
    patch_window = max(int(patch_window), 3)
    local_mu  = uniform_filter(diff, size=patch_window)
    patchiness = float(np.std(local_mu))
    patch_score = float(np.exp(-10.0 * patchiness))

    S_arti = float((halo_score * dark_score * patch_score) ** (1.0 / 3.0))

    return float(np.clip(S_arti, 1e-6, 1.0)), dict(
        halo_score=halo_score,
        dark_score=dark_score,
        patch_score=patch_score,
        patchiness=patchiness,
    )


# ============================================================
# 5. Final OTDQ Fusion
# ============================================================

def compute_otdq(hazy: np.ndarray, dehazed: np.ndarray,
                 *,
                 min_size: int = 15,
                 n_q: int = 1000,
                 var_thresh: float = 1e-4,
                 flat_alpha: float = 5.0,
                 window_size: int = 15,
                 texture_pct: float = 40.0,
                 halo_pct: float = 90.0,
                 patch_window: int = 32,
                 verbose: bool = False,
                 **kwargs) -> Dict[str, float]:
    """
    Compute OTDQ: Optical Transmission-based Dehazing Quality metric.

    Architecture
    ------------
    Three orthogonal dimensions, each ∈ [0, 1]:
      S_vis  : Visibility / contrast recovery  (ASM linear mapping regularity)
               S_vis = TR × SF × SDC_factor
      S_str  : Structural spatial fidelity     (gradient direction + local LCEC)
      S_arti : Artifact penalty                (halo / over-darkening / patchiness)

    Fusion — power-weighted geometric mean
    ------
    OTDQ = S_vis^1.0 × S_str^0.50 × S_arti^0.50

    Equivalent to log-linear weighting:
        log(OTDQ) = log(S_vis) + 0.50·log(S_str) + 0.50·log(S_arti)

    Exponents reflect per-component MOS correlation (DHQ dataset, N=1750):
        S_vis  global SROCC≈0.53, per-content SROCC≈0.59 → primary backbone (exp=1.00)
        S_str  global SROCC≈0.57, per-content SROCC≈0.39 → hallucination guard (exp=0.50)
        S_arti global SROCC≈0.42, per-content SROCC≈0.45 → artifact guard (exp=0.50)

    Parameters
    ----------
    hazy, dehazed:
        RGB images with identical shape (H, W, 3). Inputs may be uint8 [0,255]
        or float arrays already normalized to [0,1].
    min_size:
        Minimum quadtree region size for atmospheric light estimation.
    n_q:
        Number of quantiles used by the OT/QQ regression in S_vis.
    var_thresh:
        Per-channel patch variance threshold below which patches are skipped.
    flat_alpha, window_size, texture_pct:
        Structural fidelity parameters for flat-region hallucination, local
        LCEC window size, and the haze-gradient texture threshold percentile.
    halo_pct, patch_window:
        Artifact detector parameters for strong-edge halo selection and
        patchiness local averaging.

    Unsupported keyword arguments raise TypeError so stale CLI/script options do
    not silently appear to work.
    """
    if "grad_pct" in kwargs:
        halo_pct = kwargs.pop("grad_pct")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported OTDQ parameter(s): {unknown}")

    hazy = _as_float01(hazy, "hazy")
    dehazed = _as_float01(dehazed, "dehazed")
    if hazy.shape != dehazed.shape:
        raise ValueError(
            f"hazy and dehazed must have the same shape, got "
            f"{hazy.shape} and {dehazed.shape}"
        )

    hazy_dc = np.min(hazy, axis=2)
    A = estimate_atmospheric_light(hazy, min_size=min_size, dark_channel=hazy_dc)

    S_vis,  d_vis  = compute_visibility_consistency(
        hazy, dehazed, hazy_dc, A, n_q=n_q, var_thresh=var_thresh)
    S_str,  d_str  = compute_structural_fidelity(
        hazy, dehazed, flat_alpha=flat_alpha, window_size=window_size,
        texture_pct=texture_pct)
    S_arti, d_arti = compute_artifact_score(
        hazy, dehazed, halo_pct=halo_pct, patch_window=patch_window)

    # ── Power-weighted geometric mean fusion ─────────────────────────────────
    otdq = float(np.clip(
        (S_vis ** OTDQ_WEIGHTS["s_vis"]) *
        (S_str ** OTDQ_WEIGHTS["s_str"]) *
        (S_arti ** OTDQ_WEIGHTS["s_arti"]),
        1e-6, 1.0
    ))

    if verbose:
        print(f"  A = [{A[0]:.3f}, {A[1]:.3f}, {A[2]:.3f}]")
        print(f"  S_vis   = {S_vis:.4f}  (TR={d_vis['tr']:.4f}  SF={d_vis['sf']:.4f}  SDC={d_vis['sdc_factor']:.4f}  patch={d_vis['patch_size']})")
        print(f"  S_str   = {S_str:.4f}  (ssim_s={d_str['ssim_structure']:.4f}  n_pixels={d_str['n_pixels']})")
        print(f"  S_arti  = {S_arti:.4f}  (halo={d_arti['halo_score']:.4f}  dark={d_arti['dark_score']:.4f}  patch={d_arti['patch_score']:.4f})")
        print(f"  pow:  str^0.50={S_str**OTDQ_WEIGHTS['s_str']:.4f}  arti^0.50={S_arti**OTDQ_WEIGHTS['s_arti']:.4f}")
        print(f"  OTDQ    = {otdq:.4f}")

    return dict(
        otdq=float(otdq),
        s_vis=float(S_vis), s_str=float(S_str), s_arti=float(S_arti),
        tr=float(d_vis['tr']), sf=float(d_vis['sf']),
        sdc=float(d_vis['sdc']), sdc_factor=float(d_vis['sdc_factor']),
        patch_size=int(d_vis['patch_size']),
        ssim_structure=float(d_str['ssim_structure']),
        n_pixels=int(d_str['n_pixels']),
        n_textured=int(d_str['n_textured']),
        n_flat=int(d_str['n_flat']),
        n_patches=int(d_str['n_patches']),
        halo_score=float(d_arti['halo_score']),
        dark_score=float(d_arti['dark_score']),
        patch_score=float(d_arti['patch_score']),
        patchiness=float(d_arti['patchiness']),
        A=A.tolist(),
    )
