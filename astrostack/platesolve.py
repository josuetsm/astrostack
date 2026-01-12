from __future__ import annotations

import os

import numpy as np
import cv2
import pandas as pd

from astrostack import gaia_cache
from astrostack import plate_solve_pipeline

GAIA_USER = os.environ.get("GAIA_USER", "").strip()
GAIA_PASS = os.environ.get("GAIA_PASS", "").strip()


def _gaia_auth_tuple():
    if GAIA_USER and GAIA_PASS:
        return (GAIA_USER, GAIA_PASS)
    return None


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(img, (0.5, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    x = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (x * 255.0).astype(np.uint8)


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def conv2d_valid(img: np.ndarray, kernel: np.ndarray):
    out_full = cv2.filter2D(img, -1, kernel)
    kh, kw = kernel.shape
    top, bottom = (kh - 1) // 2, kh // 2
    left, right = (kw - 1) // 2, kw // 2
    out_valid = out_full[
        top : (None if bottom == 0 else -bottom),
        left : (None if right == 0 else -right),
    ]
    return out_valid, (top, bottom, left, right)


def max_pool2d_with_indices(x: np.ndarray, pool_h: int = 2, pool_w: int = 2):
    h, w = x.shape
    h2 = (h // pool_h) * pool_h
    w2 = (w // pool_w) * pool_w
    x = x[:h2, :w2]
    xb = x.reshape(h2 // pool_h, pool_h, w2 // pool_w, pool_w)
    pooled = xb.max(axis=(1, 3))
    flat = xb.reshape(h2 // pool_h, w2 // pool_w, pool_h * pool_w)
    idx_in_block = flat.argmax(axis=2)
    return pooled, idx_in_block


def pooled_indices_to_image_coords(
    idx_in_block: np.ndarray,
    pool_h: int,
    pool_w: int,
    offset_top: int,
    offset_left: int,
):
    hc, wc = idx_in_block.shape
    gy, gx = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    off_y = idx_in_block // pool_w
    off_x = idx_in_block % pool_w
    y_valid = gy * pool_h + off_y
    x_valid = gx * pool_w + off_x
    y_img = y_valid + offset_top
    x_img = x_valid + offset_left
    return y_img.astype(int), x_img.astype(int)


def nms_min_distance(y: np.ndarray, x: np.ndarray, scores: np.ndarray, min_dist: int = 50) -> np.ndarray:
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


def global_threshold_from_pooled(
    pooled: np.ndarray,
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
    madv = float(mad(x))
    thr_stat = med + sigma_k * madv
    return float(max(min_abs, thr_stat))


def convolve_and_pool(img: np.ndarray, kernel: np.ndarray, pool_size: int = 2):
    out_valid, (top, _, left, _) = conv2d_valid(img, kernel)
    pooled, idx_local = max_pool2d_with_indices(out_valid, pool_size, pool_size)
    return pooled, idx_local, (top, left)


def detect_star_candidates_with_threshold(
    pooled: np.ndarray,
    idx_local: np.ndarray,
    offsets,
    pool_size: int,
    min_separation_px: int,
    min_score: float,
):
    top, left = offsets
    ys, xs = pooled_indices_to_image_coords(idx_local, pool_size, pool_size, offset_top=top, offset_left=left)
    scores = pooled
    yv, xv, sv = ys.ravel(), xs.ravel(), scores.ravel()
    mask = sv >= float(min_score)
    if not np.any(mask):
        return []
    yv, xv, sv = yv[mask], xv[mask], sv[mask]
    keep_idx = nms_min_distance(yv, xv, sv, min_dist=min_separation_px)
    return [(int(yv[i]), int(xv[i]), float(sv[i])) for i in keep_idx]


def make_star_kernel(size=9, sigma=1.8):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).astype(np.float32)
    g /= g.sum() + 1e-12
    return g


def gaia_query_wrapper(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    gmag_max: float,
    use_healpix: bool = True,
):
    auth = _gaia_auth_tuple()
    if use_healpix:
        return gaia_cache.gaia_healpix_cone_with_mag(
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            radius_deg=float(radius_deg),
            gmag_max=float(gmag_max),
            auth=auth,
        )
    return gaia_cache.gaia_cone_with_mag(
        ra_deg=float(ra_deg),
        dec_deg=float(dec_deg),
        radius_deg=float(radius_deg),
        gmag_max=float(gmag_max),
        auth=auth,
    )


def _gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    dra = ra - ra0
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(dra)
    cosc = np.clip(cosc, 1e-12, None)
    xi = (cos_dec * np.sin(dra)) / cosc
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * np.cos(dra)) / cosc
    return xi, eta


def _gnomonic_inverse(xi, eta, ra0_deg, dec0_deg):
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    rho = np.sqrt(xi * xi + eta * eta)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    dec = np.arcsin(cos_c * sin_dec0 + (eta * sin_c * cos_dec0) / (np.maximum(rho, 1e-12)))
    ra = ra0 + np.arctan2(xi * sin_c, rho * cos_dec0 * cos_c - eta * sin_dec0 * sin_c)

    ra_deg = (np.rad2deg(ra) + 360.0) % 360.0
    dec_deg = np.rad2deg(dec)
    return ra_deg, dec_deg


def fit_similarity_tan(pix_xy: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float):
    x = pix_xy[:, 0].astype(np.float64)
    y = pix_xy[:, 1].astype(np.float64)
    xi, eta = _gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg)
    xi = xi.astype(np.float64)
    eta = eta.astype(np.float64)

    a_mat = np.zeros((2 * len(x), 4), dtype=np.float64)
    bvec = np.zeros((2 * len(x),), dtype=np.float64)

    a_mat[0::2, 0] = x
    a_mat[0::2, 1] = y
    a_mat[0::2, 2] = 1.0
    a_mat[0::2, 3] = 0.0
    bvec[0::2] = xi

    a_mat[1::2, 0] = y
    a_mat[1::2, 1] = -x
    a_mat[1::2, 2] = 0.0
    a_mat[1::2, 3] = 1.0
    bvec[1::2] = eta

    sol, *_ = np.linalg.lstsq(a_mat, bvec, rcond=None)
    a, b_, tx, ty = sol
    scale_rad_per_px = float(np.hypot(a, b_))
    rot_rad = float(np.arctan2(b_, a))
    return {
        "a": a,
        "b": b_,
        "tx": tx,
        "ty": ty,
        "scale_rad_per_px": scale_rad_per_px,
        "rot_rad": rot_rad,
    }


def pixel_to_radec_fn(map_params, ra0_deg, dec0_deg):
    a = map_params["a"]
    b_ = map_params["b"]
    tx = map_params["tx"]
    ty = map_params["ty"]

    def _pix_to_radec(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = a * x + b_ * y + tx
        eta = -b_ * x + a * y + ty
        return _gnomonic_inverse(xi, eta, ra0_deg, dec0_deg)

    return _pix_to_radec


def solve_plate_from_candidates(
    stars_yxs,
    img_shape_hw,
    gaia_radius_deg: float = 3.0,
    gaia_gmag_max: float = 14.5,
    verbose: bool = True,
):
    if len(stars_yxs) < 8:
        raise RuntimeError("Muy pocas estrellas detectadas para plate solving (<8).")

    pts = np.array([[x, y] for (y, x, s) in stars_yxs], dtype=np.float64)

    h, w = img_shape_hw
    x0 = w / 2.0
    y0 = h / 2.0

    stars_xy = pts.copy()

    pairs = plate_solve_pipeline.candidate_pairs_by_annulus(stars_xy)
    per_pair = plate_solve_pipeline.build_per_pair_tables(stars_xy, pairs)
    best, ransac_df, dbg = plate_solve_pipeline.solve_global_consensus(per_pair, max_iters=800)

    if best is None:
        return None

    assign = best.get("assign", {})
    if len(assign) < 6:
        return None

    star_idx = np.array(sorted(assign.keys()), dtype=int)
    gaia_idx = np.array([assign[i] for i in star_idx], dtype=int)

    if not isinstance(ransac_df, pd.DataFrame):
        raise RuntimeError("ransac_df no es DataFrame; revisa plate_solve_pipeline.")
    if not ("ra" in ransac_df.columns and "dec" in ransac_df.columns):
        raise RuntimeError("ransac_df no tiene columnas 'ra'/'dec'; revisa plate_solve_pipeline.")

    if ransac_df.index.is_unique and ransac_df.index.dtype != object:
        try:
            gaia_rows = ransac_df.loc[gaia_idx]
            ra = gaia_rows["ra"].to_numpy(dtype=np.float64)
            dec = gaia_rows["dec"].to_numpy(dtype=np.float64)
        except Exception:
            if "gaia_idx" in ransac_df.columns:
                gaia_rows = ransac_df.set_index("gaia_idx").loc[gaia_idx]
                ra = gaia_rows["ra"].to_numpy(dtype=np.float64)
                dec = gaia_rows["dec"].to_numpy(dtype=np.float64)
            else:
                raise
    else:
        if "gaia_idx" not in ransac_df.columns:
            raise RuntimeError("ransac_df no tiene index usable ni columna 'gaia_idx'.")
        gaia_rows = ransac_df.set_index("gaia_idx").loc[gaia_idx]
        ra = gaia_rows["ra"].to_numpy(dtype=np.float64)
        dec = gaia_rows["dec"].to_numpy(dtype=np.float64)

    ra0 = float(np.mean(ra))
    dec0 = float(np.mean(dec))

    pix_xy = pts[star_idx]
    map_params = fit_similarity_tan(pix_xy, ra, dec, ra0, dec0)
    pix_to_radec = pixel_to_radec_fn(map_params, ra0, dec0)

    scale_rad = map_params["scale_rad_per_px"]
    scale_arcsec = scale_rad * (180.0 / np.pi) * 3600.0

    if verbose:
        print(f"Plate solve OK: RA0={ra0:.5f} DEC0={dec0:.5f} scale={scale_arcsec:.2f} arcsec/px")

    return {
        "ra0_deg": ra0,
        "dec0_deg": dec0,
        "pix_to_radec": pix_to_radec,
        "scale_arcsec_per_px": scale_arcsec,
        "map_params": map_params,
    }
