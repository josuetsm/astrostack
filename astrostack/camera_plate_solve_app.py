# -*- coding: utf-8 -*-
"""
camera_plate_solve_app.py
Captura directa desde Player One (Mars-C) y ejecuta:
(1) detección automática de estrellas en el frame
(2) pipeline Gaia (HEALPix cache) + matching (plate_solve_pipeline.run_pipeline)

No escribe .ser.

Deps:
pip install numpy opencv-python astropy astroquery scipy astropy-healpix pandas matplotlib
"""

from __future__ import annotations

import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from . import pyPOACamera

from .plate_solve_pipeline import run_pipeline


# ----------------------------- POA helpers -----------------------------
def ensure_ok(err, where=""):
    if err != pyPOACamera.POAErrors.POA_OK:
        try:
            msg = pyPOACamera.GetErrorString(err)
        except Exception:
            msg = str(err)
        raise RuntimeError(f"{where} failed: {err} ({msg})")


def pick_first_camera():
    cnt = pyPOACamera.GetCameraCount()
    if cnt <= 0:
        raise RuntimeError("No hay cámaras Player One conectadas.")
    err, props = pyPOACamera.GetCameraProperties(0)
    ensure_ok(err, "GetCameraProperties(0)")
    return props.cameraID, props


def set_centered_roi(cam_id: int, props, roi_w: int, roi_h: int):
    roi_w = int(min(roi_w, props.maxWidth))
    roi_h = int(min(roi_h, props.maxHeight))
    start_x = max(0, (props.maxWidth - roi_w) // 2)
    start_y = max(0, (props.maxHeight - roi_h) // 2)
    ensure_ok(pyPOACamera.SetImageStartPos(cam_id, start_x, start_y), "SetImageStartPos(center)")
    ensure_ok(pyPOACamera.SetImageSize(cam_id, roi_w, roi_h), "SetImageSize(ROI)")
    return roi_w, roi_h, start_x, start_y


def bayer_code_to_cv2(bayer: str):
    b = bayer.strip().upper()
    table = {
        "RGGB": cv2.COLOR_BAYER_RGGB2BGR,
        "BGGR": cv2.COLOR_BAYER_BGGR2BGR,
        "GRBG": cv2.COLOR_BAYER_GRBG2BGR,
        "GBRG": cv2.COLOR_BAYER_GBRG2BGR,
    }
    if b not in table:
        raise ValueError(f"Bayer '{bayer}' no soportado. Usa RGGB/BGGR/GRBG/GBRG.")
    return table[b]


# ----------------------------- detección de estrellas -----------------------------
def robust_sigma_mad(x: np.ndarray) -> float:
    """Estimador robusto de sigma vía MAD."""
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def detect_stars(
    gray_u8: np.ndarray,
    *,
    kernel_size: int = 11,
    sigma_k: float = 6.0,
    min_abs: float = 6.0,
    min_separation_px: int = 30,
    max_stars: int = 12,
    bg_blur: int = 51,
) -> List[Tuple[int, int, float]]:
    """
    Retorna lista de (y, x, score).
    - bg_blur: tamaño de medianBlur para estimar fondo (impar).
    """
    if gray_u8.ndim != 2:
        raise ValueError("detect_stars espera gray_u8 2D.")

    g = gray_u8.astype(np.float32)

    # Fondo: median blur grande
    bb = int(bg_blur)
    if bb % 2 == 0:
        bb += 1
    bb = max(3, bb)
    bg = cv2.medianBlur(gray_u8, bb).astype(np.float32)

    img_corr = g - bg

    # Convolución con kernel caja (tipo tu notebook)
    ks = int(kernel_size)
    if ks % 2 == 0:
        ks += 1
    ks = max(3, ks)
    ker = np.ones((ks, ks), np.float32)
    ker /= float(ker.size)
    conv = cv2.filter2D(img_corr, -1, ker, borderType=cv2.BORDER_REFLECT)

    # Umbral robusto
    sig = robust_sigma_mad(conv)
    thr = max(float(min_abs), float(np.median(conv) + sigma_k * sig))

    # Máximos locales: conv == dilate(conv)
    dil = cv2.dilate(conv, np.ones((3, 3), np.uint8))
    peaks = (conv >= thr) & (conv == dil)

    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []

    scores = conv[ys, xs]
    order = np.argsort(scores)[::-1]
    ys = ys[order]
    xs = xs[order]
    scores = scores[order]

    # Enforce separación mínima (greedy)
    keep: List[Tuple[int, int, float]] = []
    min2 = float(min_separation_px * min_separation_px)

    for y, x, s in zip(ys, xs, scores):
        ok = True
        for (yy, xx, _) in keep:
            dy = float(y - yy)
            dx = float(x - xx)
            if dy * dy + dx * dx < min2:
                ok = False
                break
        if ok:
            keep.append((int(y), int(x), float(s)))
            if len(keep) >= int(max_stars):
                break

    return keep


def draw_detections(bgr: np.ndarray, stars: List[Tuple[int, int, float]]):
    out = bgr.copy()
    for (y, x, s) in stars:
        cv2.circle(out, (int(x), int(y)), 10, (0, 140, 255), 2)
        cv2.circle(out, (int(x), int(y)), 2, (0, 140, 255), -1)
    return out


# ----------------------------- App config -----------------------------
@dataclass
class AppCfg:
    roi: int = 1024
    exp_ms: int = 200
    gain: int = 200
    gain_auto: bool = False
    binning: int = 1

    # Captura/preview
    use_raw8: bool = True
    debayer: bool = True
    bayer: str = "RGGB"
    capture_n: int = 5  # promedia N frames al presionar 'c'

    # Detección estrellas
    kernel_size: int = 11
    sigma_k: float = 6.0
    min_abs: float = 6.0
    min_sep_px: int = 30
    max_stars: int = 10
    bg_blur: int = 51

    # Gaia / solve
    target: str = "0 0"     # edita/CLI
    pixel_um: float = 2.9   # edita/CLI
    focal_mm: float = 900.0 # edita/CLI
    gmax: float = 15.0
    nside: int = 32
    tol_rel: float = 0.05
    arcsec_err_cap: float = 0.05
    max_per_pair: int = 200
    max_gaia_sources: int = 8000
    plot: bool = True
    verbose: bool = False   # para NO loggear


def estimate_radius_deg(roi: int, pixel_um: float, focal_mm: float, margin: float = 1.25) -> float:
    pixel_size_m = float(pixel_um) * 1e-6
    focal_m = float(focal_mm) * 1e-3
    scale = (206265.0 * pixel_size_m) / focal_m  # arcsec/px
    diag_px = float(np.sqrt(2.0) * roi)
    diag_deg = (diag_px * scale) / 3600.0
    radius_deg = 0.5 * diag_deg * float(margin)
    return float(max(radius_deg, 0.10))  # piso para no quedar en radios ridículos


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, default=1024)
    ap.add_argument("--exp_ms", type=int, default=200)
    ap.add_argument("--gain", type=int, default=200)
    ap.add_argument("--gain_auto", action="store_true")
    ap.add_argument("--binning", type=int, default=1)

    ap.add_argument("--raw8", action="store_true", help="usar RAW8 (recomendado). Si no, RGB24.")
    ap.add_argument("--debayer", action="store_true", help="debayer en display/captura (si RAW8).")
    ap.add_argument("--bayer", type=str, default="RGGB")
    ap.add_argument("--capture_n", type=int, default=5)

    ap.add_argument("--target", type=str, default="0 0", help="Nombre (SIMBAD) o 'RA DEC' en grados / o sexagesimal.")
    ap.add_argument("--pixel_um", type=float, default=2.9)
    ap.add_argument("--focal_mm", type=float, default=900.0)
    ap.add_argument("--gmax", type=float, default=15.0)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--max_stars", type=int, default=10)
    ap.add_argument("--min_sep_px", type=int, default=30)
    args = ap.parse_args()

    cfg = AppCfg(
        roi=args.roi,
        exp_ms=args.exp_ms,
        gain=args.gain,
        gain_auto=bool(args.gain_auto),
        binning=args.binning,
        use_raw8=bool(args.raw8),
        debayer=bool(args.debayer),
        bayer=args.bayer,
        capture_n=args.capture_n,
        target=args.target,
        pixel_um=args.pixel_um,
        focal_mm=args.focal_mm,
        gmax=args.gmax,
        nside=args.nside,
        plot=bool(args.plot),
        verbose=bool(args.verbose),
        max_stars=args.max_stars,
        min_sep_px=args.min_sep_px,
    )

    cam_id, props = pick_first_camera()
    print(f"Using cameraID={cam_id} model={props.cameraModelName} color={bool(props.isColorCamera)}")

    ensure_ok(pyPOACamera.OpenCamera(cam_id), "OpenCamera")
    try:
        ensure_ok(pyPOACamera.InitCamera(cam_id), "InitCamera")

        ensure_ok(pyPOACamera.SetImageBin(cam_id, int(cfg.binning)), "SetImageBin")
        roi_w, roi_h, sx, sy = set_centered_roi(cam_id, props, cfg.roi, cfg.roi)
        print(f"ROI: {roi_w}x{roi_h} start=({sx},{sy}) bin={cfg.binning}")

        ensure_ok(pyPOACamera.SetExp(cam_id, int(cfg.exp_ms) * 1000, False), "SetExp")
        ensure_ok(pyPOACamera.SetGain(cam_id, int(cfg.gain), bool(cfg.gain_auto)), "SetGain")

        if props.isColorCamera and cfg.use_raw8:
            fmt = pyPOACamera.POAImgFormat.POA_RAW8
        elif props.isColorCamera:
            fmt = pyPOACamera.POAImgFormat.POA_RGB24
        else:
            fmt = pyPOACamera.POAImgFormat.POA_RAW8

        ensure_ok(pyPOACamera.SetImageFormat(cam_id, fmt), "SetImageFormat")

        err, iw, ih = pyPOACamera.GetImageSize(cam_id)
        ensure_ok(err, "GetImageSize")
        err, fmt2 = pyPOACamera.GetImageFormat(cam_id)
        ensure_ok(err, "GetImageFormat")
        fmt = fmt2

        buf_size = pyPOACamera.ImageCalcSize(ih, iw, fmt)
        buf = np.zeros(buf_size, dtype=np.uint8)

        ensure_ok(pyPOACamera.StartExposure(cam_id, False), "StartExposure(Video)")
        print("App started. Keys: [c]=capture+solve  [b]=debayer  [d]=toggle detections  [q]=quit")

        win = "Camera Plate Solve (Mars-C)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        do_debayer = bool(cfg.debayer)
        draw_det = True
        last_stars: List[Tuple[int, int, float]] = []
        last_metrics = None

        bayer_cv2 = None
        if do_debayer and fmt == pyPOACamera.POAImgFormat.POA_RAW8 and props.isColorCamera:
            bayer_cv2 = bayer_code_to_cv2(cfg.bayer)

        while True:
            err, ready = pyPOACamera.ImageReady(cam_id)
            ensure_ok(err, "ImageReady")
            if not ready:
                time.sleep(0.001)
                continue

            ensure_ok(pyPOACamera.GetImageData(cam_id, buf, 1000), "GetImageData")
            img = pyPOACamera.ImageDataConvert(buf, ih, iw, fmt)

            # display frame
            if img.ndim == 2:
                if do_debayer and props.isColorCamera:
                    if bayer_cv2 is None:
                        bayer_cv2 = bayer_code_to_cv2(cfg.bayer)
                    bgr = cv2.cvtColor(img, bayer_cv2)
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                # RGB24 (SDK lo entrega como RGB)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            show = bgr
            if draw_det and last_stars:
                show = draw_detections(show, last_stars)

            # overlay mínimo (ASCII para no pelear con putText)
            status1 = f"ROI={iw}x{ih} Exp={cfg.exp_ms}ms Gain={cfg.gain}{'(A)' if cfg.gain_auto else ''}  RAW8={int(cfg.use_raw8)} Debayer={int(do_debayer)}"
            status2 = f"Stars={len(last_stars)}  Target='{cfg.target}'  gmax={cfg.gmax} nside={cfg.nside}"
            if last_metrics is not None:
                status3 = f"err_med={last_metrics.get('err_median', -1):.3f}\"  err_max={last_metrics.get('err_max', -1):.3f}\"  n_img={last_metrics.get('n_img', 0)}"
            else:
                status3 = "Press 'c' to capture+solve"

            cv2.putText(show, status1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(show, status2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(show, status3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("b"):
                do_debayer = not do_debayer
                bayer_cv2 = None
                print(f"[UI] Debayer={do_debayer}")

            elif key == ord("d"):
                draw_det = not draw_det
                print(f"[UI] Draw detections={draw_det}")

            elif key == ord("c"):
                # Captura N frames y promedia (reduce ruido)
                frames = []
                t0 = time.time()
                for _ in range(int(max(1, cfg.capture_n))):
                    # esperar frame listo
                    while True:
                        err, ready = pyPOACamera.ImageReady(cam_id)
                        ensure_ok(err, "ImageReady")
                        if ready:
                            break
                        time.sleep(0.001)
                    ensure_ok(pyPOACamera.GetImageData(cam_id, buf, 1000), "GetImageData")
                    im = pyPOACamera.ImageDataConvert(buf, ih, iw, fmt)

                    if im.ndim == 2:
                        if do_debayer and props.isColorCamera:
                            if bayer_cv2 is None:
                                bayer_cv2 = bayer_code_to_cv2(cfg.bayer)
                            bgr_cap = cv2.cvtColor(im, bayer_cv2)
                            gcap = cv2.cvtColor(bgr_cap, cv2.COLOR_BGR2GRAY)
                        else:
                            gcap = im
                    else:
                        bgr_cap = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                        gcap = cv2.cvtColor(bgr_cap, cv2.COLOR_BGR2GRAY)

                    frames.append(gcap.astype(np.float32))

                gmean = np.mean(frames, axis=0)
                gray_cap = np.clip(gmean, 0, 255).astype(np.uint8)
                dt = time.time() - t0
                print(f"[CAP] captured {len(frames)} frames in {dt:.2f}s")

                stars = detect_stars(
                    gray_cap,
                    kernel_size=cfg.kernel_size,
                    sigma_k=cfg.sigma_k,
                    min_abs=cfg.min_abs,
                    min_separation_px=cfg.min_sep_px,
                    max_stars=cfg.max_stars,
                    bg_blur=cfg.bg_blur,
                )
                last_stars = stars
                print(f"[DET] stars={len(stars)} -> {stars[:5]}{' ...' if len(stars) > 5 else ''}")

                if len(stars) < 4:
                    last_metrics = None
                    print("[SOLVE] Not enough stars (need ~4+). Increase exposure/gain or lower thresholds.")
                    continue

                # radius para Gaia = radio que cubra el ROI con margen
                radius_deg = estimate_radius_deg(cfg.roi, cfg.pixel_um, cfg.focal_mm, margin=1.25)
                pixel_size_m = float(cfg.pixel_um) * 1e-6
                focal_m = float(cfg.focal_mm) * 1e-3

                print(f"[SOLVE] running pipeline: radius_deg={radius_deg:.3f} gmax={cfg.gmax} (verbose={cfg.verbose})")
                out = run_pipeline(
                    stars=stars,
                    target=cfg.target,
                    radius_deg=radius_deg,
                    gmax=cfg.gmax,
                    pixel_size_m=pixel_size_m,
                    focal_m=focal_m,
                    tol_rel=cfg.tol_rel,
                    max_per_pair=cfg.max_per_pair,
                    arcsec_err_cap=cfg.arcsec_err_cap,
                    nside=cfg.nside,
                    auth=None,
                    row_limit=-1,
                    plot=cfg.plot,
                    verbose=cfg.verbose,           # <- si está todo cacheado, no debería imprimir
                    label_brightest=20 if cfg.plot else 0,
                    simbad_radius_arcsec=1.0,
                    max_gaia_sources=cfg.max_gaia_sources,
                )
                last_metrics = out.get("metrics", None)
                print(f"[SOLVE] metrics={last_metrics}")

        ensure_ok(pyPOACamera.StopExposure(cam_id), "StopExposure")
        cv2.destroyAllWindows()

    finally:
        err = pyPOACamera.CloseCamera(cam_id)
        if err != pyPOACamera.POAErrors.POA_OK:
            print("Warning: CloseCamera returned error:", pyPOACamera.GetErrorString(err))
        else:
            print("Camera closed.")


if __name__ == "__main__":
    main()
