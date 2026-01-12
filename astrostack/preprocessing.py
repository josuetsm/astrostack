from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StretchConfig:
    plo: float = 5.0
    phi: float = 99.7
    gamma: float = 1.0
    downsample: int = 1
    blur_sigma: float = 0.0


def sw_bin2_u16(img_u16: np.ndarray) -> np.ndarray:
    a = img_u16[0::2, 0::2].astype(np.uint32)
    b = img_u16[0::2, 1::2].astype(np.uint32)
    c = img_u16[1::2, 0::2].astype(np.uint32)
    d = img_u16[1::2, 1::2].astype(np.uint32)
    return ((a + b + c + d) // 4).astype(np.uint16)


def stretch_to_u8(img_f: np.ndarray, config: StretchConfig) -> np.ndarray:
    x = img_f.astype(np.float32)
    if config.blur_sigma and config.blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(config.blur_sigma))
    if config.downsample > 1:
        x = cv2.resize(
            x,
            (x.shape[1] // config.downsample, x.shape[0] // config.downsample),
            interpolation=cv2.INTER_AREA,
        )
    samp = x[::4, ::4] if (x.shape[0] > 64 and x.shape[1] > 64) else x
    lo = np.percentile(samp, config.plo)
    hi = np.percentile(samp, config.phi)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    if config.gamma != 1.0:
        y = y ** (1.0 / config.gamma)
    return (y * 255).astype(np.uint8)


def preprocess_for_phasecorr(
    frame_u16: np.ndarray,
    bg_ema_f32: np.ndarray | None,
    sigma_hp: float,
    sigma_smooth: float,
    bright_percentile: float,
    bg_ema_alpha: float,
    subtract_bg_ema: bool = True,
    update_bg: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    x = frame_u16.astype(np.float32)

    if subtract_bg_ema:
        if bg_ema_f32 is None:
            bg_ema_f32 = x.copy()
        else:
            if update_bg:
                bg_ema_f32 = (1.0 - bg_ema_alpha) * bg_ema_f32 + bg_ema_alpha * x
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


def warp_translate(img: np.ndarray, dx: float, dy: float, is_mask: bool = False) -> np.ndarray:
    height, width = img.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def phasecorr_delta(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float, float]:
    height, width = ref.shape
    win = cv2.createHanningWindow((width, height), cv2.CV_32F)
    shift, resp = cv2.phaseCorrelate(ref * win, cur * win)
    dx, dy = shift
    return (-float(dx), -float(dy), float(resp))


def pyramid_phasecorr_delta(ref: np.ndarray, cur: np.ndarray, levels: int) -> Tuple[float, float, float]:
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


def mad_stats(x: np.ndarray) -> Tuple[float, float]:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    sig = 1.4826 * mad
    return float(med), float(sig)


def local_zscore_u16(img_u16: np.ndarray, sigma_bg: float, floor_p: float, z_clip: float) -> np.ndarray:
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


def make_sparse_reg(
    z: np.ndarray,
    peak_p: float,
    blur_sigma: float,
    hot_mask: np.ndarray | None = None,
) -> np.ndarray:
    zpos = np.maximum(z, 0.0)
    if hot_mask is not None and hot_mask.any():
        zpos[hot_mask] = 0.0
    thr = np.percentile(zpos, float(peak_p))
    reg = (zpos >= thr).astype(np.float32)
    if blur_sigma > 0:
        reg = cv2.GaussianBlur(reg, (0, 0), float(blur_sigma))
    return reg


def remove_hot_pixels(frame_u16: np.ndarray, hot_z: float = 12.0, hot_max: int = 200) -> np.ndarray:
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
