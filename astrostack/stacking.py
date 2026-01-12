from __future__ import annotations

import numpy as np
import cv2


def mad_stats(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    sig = 1.4826 * mad
    return med, sig


def local_zscore_u16(img_u16, sigma_bg, floor_p, z_clip):
    x = img_u16.astype(np.float32)
    mu = cv2.GaussianBlur(x, (0, 0), float(sigma_bg))
    mu2 = cv2.GaussianBlur(x * x, (0, 0), float(sigma_bg))
    var = np.maximum(mu2 - mu * mu, 0.0)
    sig = np.sqrt(var)

    floor = np.percentile(sig, float(floor_p))
    sig = np.maximum(sig, floor + 1e-3)

    z = (x - mu) / sig
    z = np.clip(z, -float(z_clip), float(z_clip))
    return z


def make_sparse_reg(z, peak_p, blur_sigma, hot_mask=None):
    zpos = np.maximum(z, 0.0)
    if hot_mask is not None and hot_mask.any():
        zpos[hot_mask] = 0.0
    thr = np.percentile(zpos, float(peak_p))
    reg = (zpos >= thr).astype(np.float32)
    if blur_sigma > 0:
        reg = cv2.GaussianBlur(reg, (0, 0), float(blur_sigma))
    return reg


def remove_hot_pixels(frame_u16, hot_z=12.0, hot_max=200):
    med3 = cv2.medianBlur(frame_u16, 3)
    resid = frame_u16.astype(np.int32) - med3.astype(np.int32)

    rmed, rsig = mad_stats(resid.astype(np.float32))
    z = (resid - rmed) / (rsig + 1e-6)

    cand = z > float(hot_z)
    n_cand = int(cand.sum())
    if n_cand > int(hot_max):
        flat = z.ravel()
        thr = np.partition(flat, -int(hot_max))[-int(hot_max)]
        cand = z >= thr

    out = frame_u16.copy()
    out[cand] = med3[cand]
    return out
