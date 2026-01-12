from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure


@dataclass(frozen=True)
class StarDetectionConfig:
    kernel_size: int = 11
    pool_size: int = 2
    min_separation_px: int = 30
    min_abs: float = 6.0
    sigma_k: float = 5.0
    clip_hi: float = 0.02
    clip_lo: float = 0.0
    background_degree: int = 3
    background_iters: int = 2
    k_bright: float = 6.0
    k_faint: float = 4.0
    dilate_bright: int = 3
    dilate_faint: int = 1


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _poly_terms(xx: np.ndarray, yy: np.ndarray, degree: int) -> list[np.ndarray]:
    terms = [np.ones_like(xx, dtype=np.float64)]
    if degree >= 1:
        terms += [xx, yy]
    if degree >= 2:
        terms += [xx**2, xx * yy, yy**2]
    if degree >= 3:
        terms += [xx**3, (xx**2) * yy, xx * (yy**2), yy**3]
    return terms


def _fit_poly2d(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    degree: int = 3,
    subsample: int = 1,
) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = (w - 1) / 2.0, (h - 1) / 2.0
    sx, sy = max(x0, 1.0), max(y0, 1.0)
    xn, yn = (xx - x0) / sx, (yy - y0) / sy

    sel = np.zeros_like(img, dtype=bool)
    sel[::subsample, ::subsample] = True
    sel = sel & (~mask) if mask is not None else sel

    terms = _poly_terms(xn, yn, degree)
    X = np.vstack([t[sel].ravel() for t in terms]).T
    y = img[sel].astype(np.float64).ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    bg = np.zeros_like(img, dtype=np.float64)
    for c, t in zip(coef, terms):
        bg += c * t
    return bg


def _detect_sources_mask(
    img: np.ndarray,
    k: float = 5.0,
    smooth_sigma: float = 2.0,
    dilate_px: int = 2,
) -> np.ndarray:
    smooth = gaussian_filter(img, sigma=smooth_sigma)
    thr = np.median(smooth) + k * _mad(smooth)
    mask = smooth > thr
    if dilate_px > 0:
        st = generate_binary_structure(2, 1)
        for _ in range(int(dilate_px)):
            mask = binary_dilation(mask, st)
    return mask


def subtract_background(
    img: np.ndarray,
    *,
    degree: int = 3,
    subsample: int = 1,
    iters: int = 2,
    k_bright: float = 6.0,
    k_faint: float = 3.0,
    dilate_bright: int = 3,
    dilate_faint: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _detect_sources_mask(img, k=k_bright, smooth_sigma=2.0, dilate_px=dilate_bright)
    for _ in range(max(1, iters - 1)):
        bg = _fit_poly2d(img, mask=mask, degree=degree, subsample=subsample)
        resid = img - bg
        mask |= _detect_sources_mask(resid, k=k_faint, smooth_sigma=1.5, dilate_px=dilate_faint)

    bg = _fit_poly2d(img, mask=mask, degree=degree, subsample=subsample)
    out = img - bg
    non_mask = ~mask if np.any(~mask) else np.ones_like(img, dtype=bool)
    out -= np.median(out[non_mask])
    return out.astype(np.float32), bg.astype(np.float32), mask


def _conv2d_valid(img: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    out_full = cv2.filter2D(img, -1, kernel)
    kh, kw = kernel.shape
    top, bottom = (kh - 1) // 2, kh // 2
    left, right = (kw - 1) // 2, kw // 2
    out_valid = out_full[top:(None if bottom == 0 else -bottom), left:(None if right == 0 else -right)]
    return out_valid, (top, bottom, left, right)


def _max_pool2d_with_indices(
    x: np.ndarray,
    pool_h: int = 2,
    pool_w: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = x.shape
    h2 = (height // pool_h) * pool_h
    w2 = (width // pool_w) * pool_w
    x = x[:h2, :w2]
    xb = x.reshape(h2 // pool_h, pool_h, w2 // pool_w, pool_w)
    pooled = xb.max(axis=(1, 3))
    flat = xb.reshape(h2 // pool_h, w2 // pool_w, pool_h * pool_w)
    idx_in_block = flat.argmax(axis=2)
    return pooled, idx_in_block


def _pooled_indices_to_image_coords(
    idx_in_block: np.ndarray,
    pool_h: int,
    pool_w: int,
    offset_top: int,
    offset_left: int,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = idx_in_block.shape
    gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    off_y = (idx_in_block // pool_w)
    off_x = (idx_in_block % pool_w)
    y_valid = gy * pool_h + off_y
    x_valid = gx * pool_w + off_x
    y_img = y_valid + offset_top
    x_img = x_valid + offset_left
    return y_img.astype(int), x_img.astype(int)


def _nms_min_distance(
    y: np.ndarray,
    x: np.ndarray,
    scores: np.ndarray,
    min_dist: int = 50,
) -> np.ndarray:
    order = np.argsort(scores.ravel())[::-1]
    ys, xs, ss = y[order], x[order], scores[order]
    keep = []
    for i in range(len(order)):
        yi, xi = ys[i], xs[i]
        too_close = False
        for j in keep:
            dy = yi - ys[j]
            dx = xi - xs[j]
            if (dy * dy + dx * dx) < (min_dist * min_dist):
                too_close = True
                break
        if not too_close:
            keep.append(i)
    return order[np.array(keep, dtype=int)]


def _global_threshold_from_pooled(
    pooled: np.ndarray,
    *,
    min_abs: float = 8.0,
    sigma_k: float = 5.0,
    clip_hi: float = 0.02,
    clip_lo: float = 0.0,
) -> float:
    x = pooled.astype(float).ravel()
    if clip_hi > 0 or clip_lo > 0:
        lo = np.percentile(x, 100 * clip_lo) if clip_lo > 0 else x.min()
        hi = np.percentile(x, 100 * (1.0 - clip_hi)) if clip_hi > 0 else x.max()
        sel = (x >= lo) & (x <= hi)
        x = x[sel] if np.any(sel) else x
    med = float(np.median(x))
    madv = float(_mad(x))
    thr_stat = med + sigma_k * madv
    return float(max(min_abs, thr_stat))


def _convolve_and_pool(img: np.ndarray, kernel: np.ndarray, pool_size: int = 2):
    out_valid, (top, _, left, _) = _conv2d_valid(img, kernel)
    pooled, idx_local = _max_pool2d_with_indices(out_valid, pool_size, pool_size)
    return pooled, idx_local, (top, left)


def _detect_star_candidates(
    pooled: np.ndarray,
    idx_local: np.ndarray,
    offsets: Tuple[int, int],
    pool_size: int,
    min_separation_px: int,
    min_score: float,
) -> List[Tuple[int, int, float]]:
    top, left = offsets
    ys, xs = _pooled_indices_to_image_coords(
        idx_local,
        pool_size,
        pool_size,
        offset_top=top,
        offset_left=left,
    )
    scores = pooled
    yv, xv, sv = ys.ravel(), xs.ravel(), scores.ravel()
    mask = sv >= float(min_score)
    if not np.any(mask):
        return []
    yv, xv, sv = yv[mask], xv[mask], sv[mask]
    keep_idx = _nms_min_distance(yv, xv, sv, min_dist=min_separation_px)
    return [(int(yv[i]), int(xv[i]), float(sv[i])) for i in keep_idx]


def detect_stars(
    gray_u8: np.ndarray,
    *,
    config: StarDetectionConfig | None = None,
) -> List[Tuple[int, int, float]]:
    cfg = config or StarDetectionConfig()
    img_corr, _, _ = subtract_background(
        gray_u8.astype(np.float32),
        degree=cfg.background_degree,
        subsample=1,
        iters=cfg.background_iters,
        k_bright=cfg.k_bright,
        k_faint=cfg.k_faint,
        dilate_bright=cfg.dilate_bright,
        dilate_faint=cfg.dilate_faint,
    )
    ks = int(cfg.kernel_size)
    if ks % 2 == 0:
        ks += 1
    ks = max(3, ks)
    kernel = np.ones((ks, ks), np.float32)
    kernel /= float(kernel.size)
    pooled, idx_local, offsets = _convolve_and_pool(img_corr, kernel, cfg.pool_size)
    thr = _global_threshold_from_pooled(
        pooled,
        min_abs=cfg.min_abs,
        sigma_k=cfg.sigma_k,
        clip_hi=cfg.clip_hi,
        clip_lo=cfg.clip_lo,
    )
    return _detect_star_candidates(
        pooled=pooled,
        idx_local=idx_local,
        offsets=offsets,
        pool_size=cfg.pool_size,
        min_separation_px=cfg.min_separation_px,
        min_score=thr,
    )
