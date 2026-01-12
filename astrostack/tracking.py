from __future__ import annotations

import numpy as np
import cv2

SUBTRACT_BG_EMA = True
BG_EMA_ALPHA = 0.03


def preprocess_for_phasecorr(
    frame_u16: np.ndarray,
    bg_ema_f32: np.ndarray,
    sigma_hp: float,
    sigma_smooth: float,
    bright_percentile: float,
    update_bg: bool = True,
):
    x = frame_u16.astype(np.float32)

    if SUBTRACT_BG_EMA:
        if bg_ema_f32 is None:
            bg_ema_f32 = x.copy()
        else:
            if update_bg:
                bg_ema_f32 = (1.0 - BG_EMA_ALPHA) * bg_ema_f32 + BG_EMA_ALPHA * x
        x = x - bg_ema_f32

    if sigma_hp and sigma_hp > 0:
        low = cv2.GaussianBlur(x, (0, 0), float(sigma_hp))
        x = x - low

    x = np.maximum(x, 0.0)

    if sigma_smooth and sigma_smooth > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(sigma_smooth))

    samp = x[::4, ::4] if (x.shape[0] > 64 and x.shape[1] > 64) else x
    thr = float(np.percentile(samp, float(bright_percentile)))
    mask = x >= thr

    reg = np.zeros_like(x, dtype=np.float32)
    if np.any(mask):
        vals = x[mask]
        m = float(vals.mean())
        s = float(vals.std()) + 1e-6
        reg[mask] = (vals - m) / s

    return reg.astype(np.float32), bg_ema_f32


def warp_translate(img, dx, dy, is_mask: bool = False):
    h, w = img.shape[:2]
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def phasecorr_delta(ref, cur):
    h, w = ref.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, resp = cv2.phaseCorrelate(ref * win, cur * win)
    dx, dy = shift
    return (-float(dx), -float(dy), float(resp))


def pyramid_phasecorr_delta(ref, cur, levels):
    if levels <= 1:
        return phasecorr_delta(ref, cur)

    ref_p = [ref]
    cur_p = [cur]
    for _ in range(levels - 1):
        ref_p.append(cv2.pyrDown(ref_p[-1]))
        cur_p.append(cv2.pyrDown(cur_p[-1]))

    dx_tot = 0.0
    dy_tot = 0.0
    resp_last = 0.0
    for lvl in reversed(range(levels)):
        if lvl != levels - 1:
            dx_tot *= 2.0
            dy_tot *= 2.0
        r = ref_p[lvl]
        c = cur_p[lvl]
        c_w = warp_translate(c, dx_tot, dy_tot)
        dx, dy, resp = phasecorr_delta(r, c_w)
        dx_tot += dx
        dy_tot += dy
        resp_last = resp
    return dx_tot, dy_tot, resp_last


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rate_ramp(cur, target, max_delta):
    d = target - cur
    if abs(d) <= max_delta:
        return target
    return cur + max_delta * (1.0 if d > 0 else -1.0)


def compute_A_pinv_dls(a: np.ndarray, lam: float) -> np.ndarray:
    at_a = a.T @ a
    reg = (lam * lam) * np.eye(at_a.shape[0], dtype=at_a.dtype)
    return np.linalg.inv(at_a + reg) @ a.T
