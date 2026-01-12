# ============================================================
# ASTRO APP INTEGRADA (TRACKING + STACKING MOSAIC FULL-RES
# + DETECCIÓN ESTRELLAS + PLATE SOLVE + GOTO + OBJETOS "AHORA")
#
# Reqs (según tus módulos): numpy, scipy, opencv-python, matplotlib, pandas, tqdm
# + (opcional) astropy para planetas/AltAz
#
# Archivos esperados (tú ya los subiste):
#   - gaia_cache.py
#   - plate_solve_pipeline.py
#
# ============================================================

from __future__ import annotations

import os, time, threading, traceback, queue, math
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

from astrostack import goto as goto_mod
from astrostack import platesolve as platesolve_mod
from astrostack import stacking as stacking_mod
from astrostack import tracking as tracking_mod

# ---------------------------
# 0) ENV VARS: Gaia auth
# ---------------------------
# Define aquí o en tu shell antes de correr:
#   export GAIA_USER="tu_usuario"
#   export GAIA_PASS="tu_password"
#
# En notebook puedes hacer:
#   os.environ["GAIA_USER"]="..."
#   os.environ["GAIA_PASS"]="..."

GAIA_USER = os.environ.get("GAIA_USER", "").strip()
GAIA_PASS = os.environ.get("GAIA_PASS", "").strip()

def _gaia_auth_tuple():
    if GAIA_USER and GAIA_PASS:
        return (GAIA_USER, GAIA_PASS)
    return None

# ---------------------------
# 0.1) Importa tus módulos (sin inventar nombres)
# ---------------------------
# Ajusta el path si hace falta
# En tu entorno actual (tipo sandbox), suelen estar en /mnt/data/
try:
    import sys
    if "/mnt/data" not in sys.path:
        sys.path.append("/mnt/data")

    from astrostack import gaia_cache
    from astrostack import plate_solve_pipeline
except Exception as e:
    raise RuntimeError(
        "No pude importar gaia_cache / plate_solve_pipeline. "
        "Verifica que estén en el mismo directorio o en /mnt/data/."
    ) from e


# ============================================================
# 1) UBICACIÓN / TIEMPO (Santiago de Chile)
# ============================================================
# Santiago ~ lat -33.4489, lon -70.6693, elev ~ 570m (aprox)
SITE_LAT_DEG = -33.4489
SITE_LON_DEG = -70.6693
SITE_ELEV_M  = 570.0

# ============================================================
# 2) CÁMARA + ARDUINO (tu setup)
# ============================================================
import pyPOACamera
import serial

# ============================================================
# CONFIG (tu base, sin “recortar” features)
# ============================================================
# Camera
CAM_INDEX = 0
ROI_X, ROI_Y = 0, 0
ROI_W, ROI_H = 1944, 1096
BIN_HW = 1
IMG_FMT = pyPOACamera.POAImgFormat.POA_RAW16

SW_BIN2 = True              # tracking uses proc=bin2 if True
EXP_MS_INIT = 100.0
GAIN_INIT = 360
AUTO_GAIN_INIT = True

# Display
JPEG_QUALITY = 80
DISPLAY_DS = 2
IMG_WIDTH_PX = "680px"
PLO, PHI = 5.0, 99.7
DISPLAY_BLUR_SIGMA = 0.9
DISPLAY_GAMMA_INIT = 1.0

# Tracking preproc
SUBTRACT_BG_EMA = True
BG_EMA_ALPHA = 0.03

SIGMA_HP_INIT = 10.0
SIGMA_SMOOTH_INIT = 2.0
BRIGHT_PERCENTILE_INIT = 99.3

RESP_MIN_INIT = 0.06
MAX_SHIFT_PER_FRAME_PX = 25.0

# Tracker timings
OBSERVE_S = 5.0
UPDATE_S  = 0.5
FAIL_RESET_N = 12

# Control PI
Kp_INIT = 0.20
Ki_INIT = 0.015
Kd_INIT = 0.00
EINT_CLAMP = 400.0

# DLS
LAMBDA_DLS_INIT = 0.05

# Keyframe correction
ABS_CORR_EVERY_S_INIT = 2.5
ABS_RESP_MIN_INIT = 0.08
ABS_MAX_PX = 140.0
ABS_BLEND_BETA_INIT = 0.35
KEYFRAME_REFRESH_PX_INIT = 2.5

# Rate limit
RATE_MAX = 300.0
RATE_SLEW_PER_UPDATE = 50.0

# Manual calibration
CAL_TRY_MAX = 6
CAL_STEPS_INIT = 6
CAL_STEPS_MAX  = 140
CAL_DELAY_US   = 5000
CAL_TARGET_PX_MIN = 1.0
CAL_TARGET_PX_MAX = 5.0
CAL_RESP_MIN = 0.08

# Auto-bootstrap
AUTO_BOOT_ENABLE = True
AUTO_BOOT_RATE = 25.0
AUTO_BOOT_BASE_S = 2.0
AUTO_BOOT_AXIS_S = 2.0
AUTO_BOOT_SETTLE_S = 0.6
AUTO_BOOT_MIN_SAMPLES = 8

# Auto-calibración online (RLS)
AUTO_RLS_ENABLE_DEFAULT = True
RLS_FORGET_INIT = 0.990
RLS_P0 = 2000.0
RLS_MIN_DET = 1e-4
RLS_MAX_COND = 250.0

# Recording
RECORD_SECONDS_DEFAULT = 10.0

# ============================================================
# LIVE-STACKING (MODIFICADO): FULL-RES + MOSAICO
#   - tracking sigue con proc
#   - stacking acumula sobre RAW full-res (o grayscale full-res)
#   - si el drift ve “más cielo”, expande canvas dinámicamente
# ============================================================
STACK_ENABLE_DEFAULT = False
STACK_SHOW_EVERY_N = 2

STACK_SIGMA_BG   = 35.0
STACK_SIGMA_FLOOR_P = 10.0
STACK_Z_CLIP     = 6.0
STACK_PEAK_P     = 99.75
STACK_PEAK_BLUR  = 1.0

STACK_RESP_MIN_INIT = 0.05
STACK_MAX_RAD_INIT  = 1200.0   # full-res puede ser mayor

STACK_HOT_Z_INIT    = 12.0
STACK_HOT_MAX_INIT  = 200

STACK_PLO, STACK_PHI = 5.0, 99.7
STACK_GAMMA_PREVIEW = 1.0
STACK_SAVE_DIR = "./stack_out"

# ============================================================
# Arduino
# ============================================================
ARDUINO_PORT = "/dev/cu.usbserial-1130"
ARDUINO_BAUD = 115200

# ============================================================
# Arduino helpers (thread-safe)
# ============================================================
motor_ser = None
motor_lock = threading.Lock()

def _send_cmd(cmd: str, timeout_s: float = 1.0) -> str:
    global motor_ser
    if motor_ser is None:
        return ""
    cmd = cmd.strip()
    if not cmd:
        return ""
    with motor_lock:
        try:
            motor_ser.reset_input_buffer()
        except Exception:
            pass
        motor_ser.write((cmd + "\n").encode("ascii", errors="ignore"))
        motor_ser.flush()
        t0 = time.time()
        while True:
            try:
                line = motor_ser.readline().decode(errors="ignore").strip()
            except Exception:
                line = ""
            if line:
                return line
            if (time.time() - t0) > timeout_s:
                return ""

def arduino_enable(on: bool) -> str:
    return _send_cmd(f"ENABLE {1 if on else 0}")

def arduino_rate(v_az: float, v_alt: float) -> str:
    return _send_cmd(f"RATE {float(v_az):.3f} {float(v_alt):.3f}")

def arduino_stop() -> str:
    return _send_cmd("STOP")

def mover_motor(axis: str, direction: str, steps: int, delay_us: int) -> str:
    return _send_cmd(f"MOVE {axis} {direction} {int(steps)} {int(delay_us)}", timeout_s=3.0)

def arduino_set_microsteps(az_div: int, alt_div: int) -> str:
    return _send_cmd(f"MS {int(az_div)} {int(alt_div)}")

# Conectar Arduino
ARD_STATE_STR = "<b>Arduino:</b> sin intentar conectar"
try:
    motor_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(1.8)
    r1 = _send_cmd("PING")
    arduino_enable(True)
    arduino_stop()
    ARD_STATE_STR = f"<b>Arduino:</b> conectado en {ARDUINO_PORT} (PING={r1 or '???'})"
    print("Arduino OK:", ARD_STATE_STR)
except Exception as e:
    motor_ser = None
    ARD_STATE_STR = f"<b>Arduino:</b> error al conectar ({e})"
    print("Arduino FAIL:", e)

# ============================================================
# Image helpers
# ============================================================
def sw_bin2_u16(img_u16: np.ndarray) -> np.ndarray:
    a = img_u16[0::2, 0::2].astype(np.uint32)
    b = img_u16[0::2, 1::2].astype(np.uint32)
    c = img_u16[1::2, 0::2].astype(np.uint32)
    d = img_u16[1::2, 1::2].astype(np.uint32)
    return ((a + b + c + d) // 4).astype(np.uint16)

def stretch_to_u8(img_f: np.ndarray, plo: float, phi: float, gamma: float = 1.0,
                  ds: int = 1, blur_sigma: float = 0.0) -> np.ndarray:
    x = img_f.astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(blur_sigma))
    if ds > 1:
        x = cv2.resize(x, (x.shape[1] // ds, x.shape[0] // ds), interpolation=cv2.INTER_AREA)
    samp = x[::4, ::4] if (x.shape[0] > 64 and x.shape[1] > 64) else x
    lo = np.percentile(samp, plo)
    hi = np.percentile(samp, phi)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    if gamma != 1.0:
        y = y ** (1.0 / gamma)
    return (y * 255).astype(np.uint8)

def jpeg_bytes(u8: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""

def placeholder_jpeg(w=720, h=380, text="Idle") -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else b""

# ============================================================
# Tracking preproc (robusto)
# ============================================================
def preprocess_for_phasecorr(frame_u16: np.ndarray,
                             bg_ema_f32: np.ndarray,
                             sigma_hp: float,
                             sigma_smooth: float,
                             bright_percentile: float,
                             update_bg: bool = True):
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

def warp_translate(img, dx, dy, is_mask=False):
    H, W = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (W, H), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def phasecorr_delta(ref, cur):
    H, W = ref.shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
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


# ============================================================
# STACKING: TU pipeline helpers
# ============================================================
def mad_stats(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    sig = 1.4826 * mad
    return med, sig

def local_zscore_u16(img_u16, sigma_bg, floor_p, z_clip):
    x = img_u16.astype(np.float32)
    mu  = cv2.GaussianBlur(x, (0,0), float(sigma_bg))
    mu2 = cv2.GaussianBlur(x*x, (0,0), float(sigma_bg))
    var = np.maximum(mu2 - mu*mu, 0.0)
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
        reg = cv2.GaussianBlur(reg, (0,0), float(blur_sigma))
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


# ============================================================
# 3) DETECCIÓN DE ESTRELLAS (tu snippet, con gráficos)
# ============================================================
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
    top, bottom = (kh - 1)//2, kh//2
    left, right = (kw - 1)//2, kw//2
    out_valid = out_full[top:(None if bottom == 0 else -bottom),
                         left:(None if right == 0 else -right)]
    return out_valid, (top, bottom, left, right)

def max_pool2d_with_indices(x: np.ndarray, pool_h: int = 2, pool_w: int = 2):
    H, W = x.shape
    H2 = (H // pool_h) * pool_h
    W2 = (W // pool_w) * pool_w
    x = x[:H2, :W2]
    xb = x.reshape(H2 // pool_h, pool_h, W2 // pool_w, pool_w)
    pooled = xb.max(axis=(1, 3))
    flat = xb.reshape(H2 // pool_h, W2 // pool_w, pool_h * pool_w)
    idx_in_block = flat.argmax(axis=2)
    return pooled, idx_in_block

def pooled_indices_to_image_coords(idx_in_block: np.ndarray, pool_h: int, pool_w: int, offset_top: int, offset_left: int):
    Hc, Wc = idx_in_block.shape
    gy, gx = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing='ij')
    off_y = (idx_in_block // pool_w)
    off_x = (idx_in_block %  pool_w)
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
            dy = yi - ys[j]; dx = xi - xs[j]
            if (dy*dy + dx*dx) < (min_dist * min_dist):
                too_close = True; break
        if not too_close:
            keep.append(i)
    return order[np.array(keep, dtype=int)]

def global_threshold_from_pooled(pooled: np.ndarray, min_abs: float = 8.0, sigma_k: float = 5.0, clip_hi: float = 0.02, clip_lo: float = 0.0) -> float:
    x = pooled.astype(float).ravel()
    if clip_hi > 0 or clip_lo > 0:
        lo = np.percentile(x, 100*clip_lo) if clip_lo > 0 else x.min()
        hi = np.percentile(x, 100*(1.0 - clip_hi)) if clip_hi > 0 else x.max()
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

def detect_star_candidates_with_threshold(pooled: np.ndarray, idx_local: np.ndarray, offsets, pool_size: int, min_separation_px: int, min_score: float):
    top, left = offsets
    ys, xs = pooled_indices_to_image_coords(idx_local, pool_size, pool_size,
                                            offset_top=top, offset_left=left)
    scores = pooled
    yv, xv, sv = ys.ravel(), xs.ravel(), scores.ravel()
    mask = sv >= float(min_score)
    if not np.any(mask):
        return []
    yv, xv, sv = yv[mask], xv[mask], sv[mask]
    keep_idx = nms_min_distance(yv, xv, sv, min_dist=min_separation_px)
    return [(int(yv[i]), int(xv[i]), float(sv[i])) for i in keep_idx]

# Un kernel simple “blob-ish” para detección (puedes reemplazarlo por tu kernel real si tienes uno)
def make_star_kernel(size=9, sigma=1.8):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    g = np.exp(-(xx*xx + yy*yy)/(2*sigma*sigma)).astype(np.float32)
    g /= (g.sum() + 1e-12)
    return g

# ============================================================
# 4) GEOMETRÍA MONTURA + GOTO
# ============================================================

# ---- AZ: GT2 belt on disk radius 24cm, pulley 20T, pitch 2mm
AZ_DISK_R_MM = 240.0
GT2_PITCH_MM = 2.0
AZ_PULLEY_T  = 20
AZ_PULLEY_C_MM = AZ_PULLEY_T * GT2_PITCH_MM
AZ_DISK_C_MM  = 2.0 * math.pi * AZ_DISK_R_MM
AZ_GEAR_RATIO = AZ_DISK_C_MM / AZ_PULLEY_C_MM   # motor revs per disk rev? (actually disk_mm / pulley_mm)
# disk angle per motor revolution:
AZ_DEG_PER_MOTOR_REV = 360.0 / AZ_GEAR_RATIO

# ---- ALT: triángulo a=39.5, b=44.5, c variable, theta(c)=phi_b - gamma(c)
ALT_A_CM = 39.5
ALT_B_CM = 44.5
ALT_C0_CM = 65.5
ALT_THETA0_DEG = 5.8

# husillo: 7 dientes por 1cm -> paso lineal p = 10/7 mm por vuelta (single-start)
ALT_PITCH_MM_PER_REV = 10.0 / 7.0

def gamma_from_c_cm(c_cm: float) -> float:
    a = ALT_A_CM
    b = ALT_B_CM
    c = float(c_cm)
    num = a*a + b*b - c*c
    den = 2.0*a*b
    x = num/den
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))

GAMMA0_DEG = gamma_from_c_cm(ALT_C0_CM)
PHI_B_DEG  = ALT_THETA0_DEG + GAMMA0_DEG

def theta_from_c_cm(c_cm: float) -> float:
    return PHI_B_DEG - gamma_from_c_cm(c_cm)

# ---- Estado “calibrado” para goto
MOUNT = {
    "cal_ok": False,

    # contador de motor (microsteps acumulados) para AZ/ALT
    "az_micro": 0.0,
    "alt_micro": 0.0,

    # microstepping actual (se toma de dropdown)
    "ms_az": 64,
    "ms_alt": 64,

    # ALT: longitud c actual (cm) (modelo)
    "alt_c_cm": ALT_C0_CM,

    # offsets de orientación (para alinear signo y cero)
    "az0_sky_deg": None,     # azimut del centro (deg) en el momento del solve aceptado
    "alt0_sky_deg": None,    # altitud del centro (deg) en el momento del solve aceptado

    "az0_micro": None,
    "alt0_micro": None,

    # signos (depende de tu montaje). arrancamos +1 y tú lo ajustas si queda al revés
    "sign_az": +1.0,
    "sign_alt": +1.0,
}

def az_micro_to_deg(delta_micro: float, ms_az: int) -> float:
    # microsteps -> motor rev -> disk deg
    motor_rev = delta_micro / (200.0 * ms_az)
    return motor_rev * AZ_DEG_PER_MOTOR_REV

def alt_micro_to_c_cm(delta_micro: float, ms_alt: int) -> float:
    # microsteps -> motor rev -> mm -> cm
    motor_rev = delta_micro / (200.0 * ms_alt)
    delta_mm = motor_rev * ALT_PITCH_MM_PER_REV
    return delta_mm / 10.0

def mount_predict_altaz_from_micro(az_micro: float, alt_micro: float):
    if not MOUNT["cal_ok"]:
        return None
    ms_az = int(MOUNT["ms_az"])
    ms_alt = int(MOUNT["ms_alt"])

    d_az_micro = az_micro - float(MOUNT["az0_micro"])
    d_alt_micro = alt_micro - float(MOUNT["alt0_micro"])

    az_deg = float(MOUNT["az0_sky_deg"]) + MOUNT["sign_az"] * az_micro_to_deg(d_az_micro, ms_az)

    # ALT: actualizamos c desde el punto de calibración
    c_cm = float(MOUNT["alt_c_cm"]) + MOUNT["sign_alt"] * alt_micro_to_c_cm(d_alt_micro, ms_alt)
    # clamp básico para evitar NaNs (debes poner tus límites reales)
    c_cm = float(np.clip(c_cm, 20.0, 200.0))
    alt_deg = theta_from_c_cm(c_cm)

    # normaliza az a [0,360)
    az_deg = az_deg % 360.0
    return az_deg, alt_deg, c_cm

def mount_set_calibration(az_center_deg: float, alt_center_deg: float, az_micro_now: float, alt_micro_now: float, ms_az: int, ms_alt: int, c_cm_now: float):
    MOUNT["cal_ok"] = True
    MOUNT["az0_sky_deg"] = float(az_center_deg)
    MOUNT["alt0_sky_deg"] = float(alt_center_deg)
    MOUNT["az0_micro"] = float(az_micro_now)
    MOUNT["alt0_micro"] = float(alt_micro_now)
    MOUNT["ms_az"] = int(ms_az)
    MOUNT["ms_alt"] = int(ms_alt)
    MOUNT["alt_c_cm"] = float(c_cm_now)

def mount_goto_altaz(target_az_deg: float, target_alt_deg: float, rate_az_uS: float = 220.0, rate_alt_uS: float = 220.0, tol_deg: float = 0.05):
    """
    GoTo simple: calcula delta microsteps y ejecuta MOVE bloqueante.
    Asume steppers perfectos (sin pérdida de pasos).
    """
    if motor_ser is None:
        log("GOTO: Arduino no conectado")
        flush_log_to_widget()
        return
    if not MOUNT["cal_ok"]:
        log("GOTO: no calibrado aún (haz plate-solve y Accept primero)")
        flush_log_to_widget()
        return

    ms_az = int(MOUNT["ms_az"])
    ms_alt = int(MOUNT["ms_alt"])

    # --- AZ: delta deg -> delta micro
    cur = mount_predict_altaz_from_micro(MOUNT["az_micro"], MOUNT["alt_micro"])
    if cur is None:
        return
    cur_az, cur_alt, cur_c = cur

    # delta az en el sentido más corto
    da = (float(target_az_deg) - float(cur_az) + 540.0) % 360.0 - 180.0
    # microsteps necesarios
    deg_per_micro = AZ_DEG_PER_MOTOR_REV / (200.0 * ms_az)
    d_micro_az = (da / deg_per_micro) * (1.0 / MOUNT["sign_az"])

    # --- ALT: invertimos theta(c) numéricamente para obtener c_target
    # Buscamos c tal que theta_from_c_cm(c)=target_alt
    target_alt = float(target_alt_deg)
    # búsqueda binaria en un rango razonable
    lo, hi = 20.0, 200.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        th = theta_from_c_cm(mid)
        if th < target_alt:
            # si theta sube cuando c baja (en tu modelo actual, acortar c sube theta),
            # entonces cuando th < target, necesitamos bajar c => mover hi hacia mid.
            # Pero esto depende de la monotonicidad real. Aquí asumimos la rama estándar.
            hi = mid
        else:
            lo = mid
    c_target = 0.5*(lo+hi)

    # delta c cm -> delta micro
    dc = (c_target - cur_c)
    cm_per_micro = (ALT_PITCH_MM_PER_REV/10.0) / (200.0 * ms_alt)  # cm por microstep
    d_micro_alt = (dc / cm_per_micro) * (1.0 / MOUNT["sign_alt"])

    # Ejecuta MOVE (bloqueante) con delay_us fijo
    # Tú usas delay_us como “velocidad”: menor delay => más rápido. Conservador aquí.
    delay_az = int(max(200, min(5000, rate_az_uS)))
    delay_alt = int(max(200, min(5000, rate_alt_uS)))

    # AZ move
    if abs(d_micro_az) >= 1.0:
        axis = "A"
        direction = "FWD" if d_micro_az >= 0 else "REV"
        steps = int(abs(d_micro_az))
        arduino_rate(0, 0)
        mover_motor(axis, direction, steps, delay_az)
        MOUNT["az_micro"] += float(np.sign(d_micro_az) * steps)

    # ALT move
    if abs(d_micro_alt) >= 1.0:
        axis = "B"
        direction = "FWD" if d_micro_alt >= 0 else "REV"
        steps = int(abs(d_micro_alt))
        arduino_rate(0, 0)
        mover_motor(axis, direction, steps, delay_alt)
        MOUNT["alt_micro"] += float(np.sign(d_micro_alt) * steps)

    # actualiza c estimada en el modelo
    cur2 = mount_predict_altaz_from_micro(MOUNT["az_micro"], MOUNT["alt_micro"])
    if cur2 is not None:
        _, _, c2 = cur2
        MOUNT["alt_c_cm"] = float(c2)

    log(f"GOTO: target(AZ,ALT)=({target_az_deg:.2f},{target_alt_deg:.2f}) "
        f"da={da:+.2f}deg dµ_az={d_micro_az:+.0f} dµ_alt={d_micro_alt:+.0f} -> done")
    flush_log_to_widget()


# ============================================================
# 5) PLATE SOLVE: wrapper con Gaia cache + ajuste TAN similarity
# ============================================================

def gaia_query_wrapper(ra_deg: float, dec_deg: float, radius_deg: float, gmag_max: float, use_healpix=True):
    """
    Intenta resolver Gaia desde cache; si falta patch, permite auth si está disponible.
    Usa tus funciones reales:
      - gaia_cache.gaia_cone_with_mag(...)
      - gaia_cache.gaia_healpix_cone_with_mag(...)
    """
    auth = _gaia_auth_tuple()  # None si no hay credenciales
    # NOTA: gaia_cache internamente maneja cache; si no está, descarga.
    # Aquí no “forzamos” login si no hay necesidad: pasamos auth sólo si existe.
    if use_healpix:
        return gaia_cache.gaia_healpix_cone_with_mag(
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            radius_deg=float(radius_deg),
            gmag_max=float(gmag_max),
            auth=auth
        )
    else:
        return gaia_cache.gaia_cone_with_mag(
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            radius_deg=float(radius_deg),
            gmag_max=float(gmag_max),
            auth=auth
        )

def _gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    dra = ra - ra0
    sin_dec = np.sin(dec); cos_dec = np.cos(dec)
    sin_dec0 = np.sin(dec0); cos_dec0 = np.cos(dec0)

    cosc = sin_dec0*sin_dec + cos_dec0*cos_dec*np.cos(dra)
    cosc = np.clip(cosc, 1e-12, None)
    xi = (cos_dec*np.sin(dra)) / cosc
    eta = (cos_dec0*sin_dec - sin_dec0*cos_dec*np.cos(dra)) / cosc
    return xi, eta  # radians

def _gnomonic_inverse(xi, eta, ra0_deg, dec0_deg):
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    rho = np.sqrt(xi*xi + eta*eta)
    c = np.arctan(rho)
    sin_c = np.sin(c); cos_c = np.cos(c)

    sin_dec0 = np.sin(dec0); cos_dec0 = np.cos(dec0)

    dec = np.arcsin(cos_c*sin_dec0 + (eta*sin_c*cos_dec0)/(np.maximum(rho, 1e-12)))
    ra = ra0 + np.arctan2(xi*sin_c, rho*cos_dec0*cos_c - eta*sin_dec0*sin_c)

    ra_deg = (np.rad2deg(ra) + 360.0) % 360.0
    dec_deg = np.rad2deg(dec)
    return ra_deg, dec_deg

def fit_similarity_tan(pix_xy: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float):
    """
    Ajusta:
      xi = a*x + b*y + tx
      eta= -b*x + a*y + ty
    donde (xi,eta) son coordenadas gnomónicas (radianes) respecto a (ra0,dec0)
    """
    x = pix_xy[:,0].astype(np.float64)
    y = pix_xy[:,1].astype(np.float64)
    xi, eta = _gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg)
    xi = xi.astype(np.float64)
    eta = eta.astype(np.float64)

    # Construye sistema lineal para (a,b,tx,ty)
    # xi = a*x + b*y + tx
    # eta= -b*x + a*y + ty
    # => [ x  y  1  0 ] [a] = xi
    #    [ y -x  0  1 ] [b] = eta  (reordenado)
    A = np.zeros((2*len(x), 4), dtype=np.float64)
    bvec = np.zeros((2*len(x),), dtype=np.float64)

    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1.0
    A[0::2, 3] = 0.0
    bvec[0::2] = xi

    A[1::2, 0] = y
    A[1::2, 1] = -x
    A[1::2, 2] = 0.0
    A[1::2, 3] = 1.0
    bvec[1::2] = eta

    sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    a, b_, tx, ty = sol
    scale_rad_per_px = float(np.hypot(a, b_))
    rot_rad = float(np.arctan2(b_, a))
    return {"a":a, "b":b_, "tx":tx, "ty":ty, "scale_rad_per_px":scale_rad_per_px, "rot_rad":rot_rad}

def pixel_to_radec_fn(map_params, ra0_deg, dec0_deg):
    a = map_params["a"]; b_ = map_params["b"]; tx = map_params["tx"]; ty = map_params["ty"]
    def _pix_to_radec(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi  = a*x + b_*y + tx
        eta = -b_*x + a*y + ty
        return _gnomonic_inverse(xi, eta, ra0_deg, dec0_deg)
    return _pix_to_radec

def solve_plate_from_candidates(stars_yxs, img_shape_hw, gaia_radius_deg=3.0, gaia_gmag_max=14.5, verbose=True):
    """
    Usa plate_solve_pipeline:
      - candidate_pairs_by_annulus
      - build_per_pair_tables
      - solve_global_consensus
    y después ajusta TAN similarity para obtener WCS-like mapping.
    """
    if len(stars_yxs) < 8:
        raise RuntimeError("Muy pocas estrellas detectadas para plate solving (<8).")

    # pix coords en (x,y) float
    pts = np.array([[x, y] for (y, x, s) in stars_yxs], dtype=np.float64)

    # Suposición inicial de centro en el centro de imagen; luego lo iteramos con el ajuste
    H, W = img_shape_hw
    x0 = W/2.0
    y0 = H/2.0

    # Normaliza a coords relativas para invariantes
    # plate_solve_pipeline espera arrays de coords; inspecciona tu módulo:
    stars_xy = pts.copy()

    # Genera parejas candidatas (en tu módulo)
    pairs = plate_solve_pipeline.candidate_pairs_by_annulus(stars_xy)

    # Tabla de pares contra Gaia: tu módulo provee builder
    # NOTA: build_per_pair_tables llama a Gaia dentro de tu pipeline original.
    # Aquí no asumimos esa llamada; en su lugar, pedimos un “campo” Gaia alrededor de
    # una conjetura. Como todavía no sabemos RA/Dec, el pipeline interno de tu módulo
    # suele hacer búsqueda multi-hipótesis. Para mantenerlo consistente, usamos su función
    # build_per_pair_tables tal cual (no inventamos otra).
    per_pair = plate_solve_pipeline.build_per_pair_tables(stars_xy, pairs)

    best, ransac_df, dbg = plate_solve_pipeline.solve_global_consensus(per_pair, max_iters=800)

    if best is None:
        return None

    assign = best.get("assign", {})
    if len(assign) < 6:
        return None

    # Extrae matches
    star_idx = np.array(sorted(assign.keys()), dtype=int)
    gaia_idx = np.array([assign[i] for i in star_idx], dtype=int)

    # En ransac_df tu pipeline ya trae RA/Dec Gaia por fila (según tu implementación).
    # Para no inventar: buscamos columnas estándar:
    #   - 'ra', 'dec' en grados
    # y que el índice corresponda a “gaia index”.
    if not isinstance(ransac_df, pd.DataFrame):
        raise RuntimeError("ransac_df no es DataFrame; revisa plate_solve_pipeline.")
    if not ("ra" in ransac_df.columns and "dec" in ransac_df.columns):
        raise RuntimeError("ransac_df no tiene columnas 'ra'/'dec'; revisa plate_solve_pipeline.")

    # Mapea gaia_idx a ra/dec:
    # si ransac_df usa filas indexadas por gaia_idx, esto funciona.
    # si no, tu pipeline debe proveer una columna 'gaia_idx'. Intentamos ambos.
    if ransac_df.index.is_unique and ransac_df.index.dtype != object:
        # intentamos .loc
        try:
            gaia_rows = ransac_df.loc[gaia_idx]
            ra = gaia_rows["ra"].to_numpy(dtype=np.float64)
            dec = gaia_rows["dec"].to_numpy(dtype=np.float64)
        except Exception:
            if "gaia_idx" in ransac_df.columns:
                sub = ransac_df[ransac_df["gaia_idx"].isin(gaia_idx)].copy()
                sub = sub.set_index("gaia_idx").loc[gaia_idx]
                ra = sub["ra"].to_numpy(dtype=np.float64)
                dec = sub["dec"].to_numpy(dtype=np.float64)
            else:
                raise
    else:
        if "gaia_idx" in ransac_df.columns:
            sub = ransac_df[ransac_df["gaia_idx"].isin(gaia_idx)].copy()
            sub = sub.set_index("gaia_idx").loc[gaia_idx]
            ra = sub["ra"].to_numpy(dtype=np.float64)
            dec = sub["dec"].to_numpy(dtype=np.float64)
        else:
            raise RuntimeError("No puedo mapear gaia_idx->ra/dec en ransac_df.")

    pix = stars_xy[star_idx]  # (x,y)

    # Estima (ra0,dec0) inicial como mediana de matches
    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))

    # Refina 2-3 iteraciones (recentrando ra0,dec0 en el pixel center)
    params = None
    for _ in range(3):
        params = fit_similarity_tan(pix, ra, dec, ra0, dec0)
        pix_to_radec = pixel_to_radec_fn(params, ra0, dec0)
        ra_c, dec_c = pix_to_radec(x0, y0)
        ra0 = float(np.atleast_1d(ra_c)[0])
        dec0 = float(np.atleast_1d(dec_c)[0])

    # escala y rotación
    scale_arcsec_per_px = params["scale_rad_per_px"] * (180.0/math.pi) * 3600.0
    rot_deg = params["rot_rad"] * (180.0/math.pi)

    out = {
        "best": best,
        "ransac_df": ransac_df,
        "dbg": dbg,
        "assign": assign,
        "pix_matches_xy": pix,
        "sky_matches_ra": ra,
        "sky_matches_dec": dec,
        "ra0_deg": ra0,
        "dec0_deg": dec0,
        "scale_arcsec_per_px": float(scale_arcsec_per_px),
        "rot_deg": float(rot_deg),
        "map_params": params,
        "pixel_to_radec": pixel_to_radec_fn(params, ra0, dec0),
        "img_center_xy": (x0, y0),
    }
    return out


# ============================================================
# Rebind helpers to module implementations
# ============================================================
preprocess_for_phasecorr = tracking_mod.preprocess_for_phasecorr
warp_translate = tracking_mod.warp_translate
phasecorr_delta = tracking_mod.phasecorr_delta
pyramid_phasecorr_delta = tracking_mod.pyramid_phasecorr_delta
clamp = tracking_mod.clamp
rate_ramp = tracking_mod.rate_ramp
compute_A_pinv_dls = tracking_mod.compute_A_pinv_dls

mad_stats = stacking_mod.mad_stats
local_zscore_u16 = stacking_mod.local_zscore_u16
make_sparse_reg = stacking_mod.make_sparse_reg
remove_hot_pixels = stacking_mod.remove_hot_pixels

normalize_to_uint8 = platesolve_mod.normalize_to_uint8
mad = platesolve_mod.mad
conv2d_valid = platesolve_mod.conv2d_valid
max_pool2d_with_indices = platesolve_mod.max_pool2d_with_indices
pooled_indices_to_image_coords = platesolve_mod.pooled_indices_to_image_coords
nms_min_distance = platesolve_mod.nms_min_distance
global_threshold_from_pooled = platesolve_mod.global_threshold_from_pooled
convolve_and_pool = platesolve_mod.convolve_and_pool
detect_star_candidates_with_threshold = platesolve_mod.detect_star_candidates_with_threshold
make_star_kernel = platesolve_mod.make_star_kernel
gaia_query_wrapper = platesolve_mod.gaia_query_wrapper
_gnomonic_forward = platesolve_mod._gnomonic_forward
_gnomonic_inverse = platesolve_mod._gnomonic_inverse
fit_similarity_tan = platesolve_mod.fit_similarity_tan
pixel_to_radec_fn = platesolve_mod.pixel_to_radec_fn
solve_plate_from_candidates = platesolve_mod.solve_plate_from_candidates


# ============================================================
# 6) OBJETOS AHORA (planetas) - opcional con astropy
# ============================================================
_ASTROPY_OK = False
try:
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_body
    import astropy.units as u
    _ASTROPY_OK = True
except Exception:
    _ASTROPY_OK = False

def now_planets_altaz():
    if not _ASTROPY_OK:
        return None, "Astropy no disponible: planetas/AltAz deshabilitado."
    loc = EarthLocation(lat=SITE_LAT_DEG*u.deg, lon=SITE_LON_DEG*u.deg, height=SITE_ELEV_M*u.m)
    t = Time.now()
    frame = AltAz(obstime=t, location=loc)
    names = ["moon","mercury","venus","mars","jupiter","saturn","uranus","neptune"]
    rows = []
    for nm in names:
        try:
            sc = get_body(nm, t, loc).transform_to(frame)
            rows.append([nm, float(sc.alt.deg), float(sc.az.deg), float(sc.distance.to(u.au).value)])
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["name","alt_deg","az_deg","dist_au"]).sort_values("alt_deg", ascending=False)
    return df, None

def radec_to_altaz(ra_deg: float, dec_deg: float):
    if not _ASTROPY_OK:
        raise RuntimeError("Astropy no disponible para RA/Dec -> AltAz.")
    loc = EarthLocation(lat=SITE_LAT_DEG*u.deg, lon=SITE_LON_DEG*u.deg, height=SITE_ELEV_M*u.m)
    t = Time.now()
    frame = AltAz(obstime=t, location=loc)
    sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    aa = sc.transform_to(frame)
    return float(aa.az.deg), float(aa.alt.deg)


# ============================================================
# 7) UI (tu UI base + tabs nuevos: PlateSolve + GoTo)
# ============================================================

btn_start = W.Button(description="Start", button_style="success", layout=W.Layout(width="90px"))
btn_stop  = W.Button(description="Stop",  button_style="danger",  layout=W.Layout(width="90px"))

ck_tracking = W.Checkbox(description="Tracking (PI + keyframe)", value=False)

ck_auto_rls = W.Checkbox(description="AutoCal online (RLS)", value=AUTO_RLS_ENABLE_DEFAULT)
sl_rls_forget = W.BoundedFloatText(description="RLS λ", min=0.950, max=0.999, step=0.001,
                                   value=RLS_FORGET_INIT, layout=W.Layout(width="180px"))

# Cámara
txt_exp_ms = W.BoundedIntText(description="Exp (ms)", min=1, max=60000, value=int(EXP_MS_INIT),
                              layout=W.Layout(width="180px"))
txt_gain = W.BoundedIntText(description="Gain", min=0, max=2000, value=int(GAIN_INIT),
                            layout=W.Layout(width="150px"))
ck_auto_gain = W.Checkbox(description="Auto gain", value=AUTO_GAIN_INIT)
sl_gamma  = W.BoundedFloatText(description="gamma", min=0.5, max=3.0, step=0.05,
                               value=DISPLAY_GAMMA_INIT, layout=W.Layout(width="200px"))

ck_record_raw = W.Checkbox(description="Record RAW full-res", value=False)

btn_cal_az    = W.Button(description="Calib AZ",  layout=W.Layout(width="110px"))
btn_cal_alt   = W.Button(description="Calib ALT", layout=W.Layout(width="110px"))
btn_cal_reset = W.Button(description="Reset Cal", button_style="warning", layout=W.Layout(width="110px"))
btn_key_reset = W.Button(description="Reset Keyframe", layout=W.Layout(width="130px"))

ms_opts = [8, 16, 32, 64]
dd_ms_az  = W.Dropdown(options=ms_opts, value=64, description="MS AZ", layout=W.Layout(width="170px"))
dd_ms_alt = W.Dropdown(options=ms_opts, value=64, description="MS ALT", layout=W.Layout(width="170px"))
btn_apply_ms = W.Button(description="Apply MS", layout=W.Layout(width="110px"))

# Tracking preproc params
sl_bright = W.BoundedFloatText(description="bright%", min=90.0, max=99.9, step=0.1,
                               value=BRIGHT_PERCENTILE_INIT, layout=W.Layout(width="220px"))
sl_hp = W.BoundedFloatText(description="σ_hp", min=2.0, max=40.0, step=0.5,
                           value=SIGMA_HP_INIT, layout=W.Layout(width="200px"))
sl_sm = W.BoundedFloatText(description="σ_sm", min=0.0, max=10.0, step=0.2,
                           value=SIGMA_SMOOTH_INIT, layout=W.Layout(width="200px"))
sl_respmin = W.BoundedFloatText(description="resp_min", min=0.01, max=0.40, step=0.01,
                                value=RESP_MIN_INIT, layout=W.Layout(width="200px"))

# Control PI
sl_kp = W.BoundedFloatText(description="Kp", min=0.02, max=1.00, step=0.01,
                           value=Kp_INIT, layout=W.Layout(width="200px"))
sl_ki = W.BoundedFloatText(description="Ki", min=0.000, max=0.200, step=0.001,
                           value=Ki_INIT, layout=W.Layout(width="200px"))
sl_kd = W.BoundedFloatText(description="Kd", min=0.00, max=1.00, step=0.01,
                           value=Kd_INIT, layout=W.Layout(width="200px"))

sl_lam = W.BoundedFloatText(description="λ_dls", min=0.00, max=1.00, step=0.01,
                            value=LAMBDA_DLS_INIT, layout=W.Layout(width="200px"))

sl_abs_every = W.BoundedFloatText(description="ABS_s", min=0.5, max=20.0, step=0.5,
                                  value=ABS_CORR_EVERY_S_INIT, layout=W.Layout(width="200px"))
sl_abs_beta  = W.BoundedFloatText(description="ABS_β", min=0.0, max=0.9, step=0.05,
                                  value=ABS_BLEND_BETA_INIT, layout=W.Layout(width="200px"))
sl_abs_resp  = W.BoundedFloatText(description="ABS_resp", min=0.01, max=0.40, step=0.01,
                                  value=ABS_RESP_MIN_INIT, layout=W.Layout(width="200px"))
sl_kref_px   = W.BoundedFloatText(description="key_px", min=0.5, max=20.0, step=0.5,
                                  value=KEYFRAME_REFRESH_PX_INIT, layout=W.Layout(width="200px"))

# Manual slews
txt_slew_steps_az = W.IntText(description="AZ steps", value=600, layout=W.Layout(width="180px"))
txt_slew_delay_az = W.IntText(description="AZ delay (µs)", value=1800, layout=W.Layout(width="180px"))
txt_slew_steps_alt = W.IntText(description="ALT steps", value=600, layout=W.Layout(width="180px"))
txt_slew_delay_alt = W.IntText(description="ALT delay (µs)", value=1800, layout=W.Layout(width="180px"))

btn_az_left  = W.Button(description="AZ ←", layout=W.Layout(width="80px"))
btn_az_right = W.Button(description="AZ →", layout=W.Layout(width="80px"))
btn_alt_up   = W.Button(description="ALT ↑", layout=W.Layout(width="80px"))
btn_alt_down = W.Button(description="ALT ↓", layout=W.Layout(width="80px"))

# Recording
btn_rec10 = W.Button(description="Record 10s (.npy)", layout=W.Layout(width="150px"))

# Stacking UI
ck_stack_enable = W.Checkbox(description="Enable stacking (mosaic full-res)", value=STACK_ENABLE_DEFAULT)
btn_stack_start = W.Button(description="Start stack", button_style="success", layout=W.Layout(width="120px"))
btn_stack_stop  = W.Button(description="Stop stack",  button_style="warning", layout=W.Layout(width="120px"))
btn_stack_reset = W.Button(description="Reset stack", layout=W.Layout(width="120px"))
btn_stack_save  = W.Button(description="Save PNG",    layout=W.Layout(width="120px"))

sl_stack_resp = W.BoundedFloatText(description="RESP_MIN", min=0.01, max=0.40, step=0.01,
                                   value=STACK_RESP_MIN_INIT, layout=W.Layout(width="200px"))
sl_stack_maxrad = W.BoundedFloatText(description="MAX_RAD", min=20.0, max=5000.0, step=10.0,
                                     value=STACK_MAX_RAD_INIT, layout=W.Layout(width="200px"))

sl_hot_z = W.BoundedFloatText(description="HOT_Z", min=4.0, max=40.0, step=0.5,
                              value=STACK_HOT_Z_INIT, layout=W.Layout(width="200px"))
txt_hot_max = W.BoundedIntText(description="HOT_MAX", min=0, max=5000, step=10,
                               value=int(STACK_HOT_MAX_INIT), layout=W.Layout(width="200px"))

# ---- Plate solve UI
btn_detect_stars = W.Button(description="Detect stars", button_style="", layout=W.Layout(width="140px"))
btn_plate_solve  = W.Button(description="Plate solve",  button_style="info", layout=W.Layout(width="140px"))
btn_accept_solve = W.Button(description="Accept solve", button_style="success", layout=W.Layout(width="140px"))

txt_min_sep = W.BoundedIntText(description="min_sep(px)", min=3, max=300, value=30, layout=W.Layout(width="180px"))
txt_sigma_k = W.BoundedFloatText(description="sigma_k", min=1.0, max=12.0, value=5.0, step=0.5, layout=W.Layout(width="160px"))
txt_min_abs = W.BoundedFloatText(description="min_abs", min=0.0, max=200.0, value=8.0, step=0.5, layout=W.Layout(width="160px"))
txt_pool = W.Dropdown(options=[1,2,3,4], value=2, description="pool", layout=W.Layout(width="130px"))

txt_gaia_rad = W.BoundedFloatText(description="Gaia rad(deg)", min=0.5, max=10.0, value=3.0, step=0.25, layout=W.Layout(width="200px"))
txt_gmag_max = W.BoundedFloatText(description="Gmag max", min=5.0, max=20.0, value=14.5, step=0.5, layout=W.Layout(width="170px"))

out_plots = W.Output(layout=W.Layout(width="980px", height="420px", border="1px solid #ddd"))

# ---- GoTo UI
btn_refresh_objs = W.Button(description="Refresh objects now", layout=W.Layout(width="180px"))
dd_targets = W.Dropdown(options=["(none)"], value="(none)", description="Target", layout=W.Layout(width="320px"))
btn_goto = W.Button(description="GoTo target", button_style="warning", layout=W.Layout(width="150px"))

txt_goto_delay_az  = W.BoundedIntText(description="AZ delay_us", min=200, max=6000, value=220, layout=W.Layout(width="180px"))
txt_goto_delay_alt = W.BoundedIntText(description="ALT delay_us", min=200, max=6000, value=220, layout=W.Layout(width="190px"))

lab_ard = W.HTML(value=ARD_STATE_STR)
lab = W.HTML(value="<b>Status:</b> idle")
lab_cal = W.HTML(value="<b>Cal:</b> AZ=? ALT=? | A_pinv=None | AutoA=None")
lab_stack = W.HTML(value="<b>Stack:</b> OFF")
lab_solve = W.HTML(value="<b>PlateSolve:</b> none")
lab_goto  = W.HTML(value="<b>GoTo:</b> not calibrated")

img_live = W.Image(format="jpeg", value=placeholder_jpeg(text="Idle - press Start"))
img_live.layout = W.Layout(width=IMG_WIDTH_PX)

img_stack = W.Image(format="jpeg", value=placeholder_jpeg(text="Stack mosaic preview (idle)"))
img_stack.layout = W.Layout(width=IMG_WIDTH_PX)

log_area = W.Textarea(value="", layout=W.Layout(width="980px", height="240px"))
log_area.disabled = True

tab_cam = W.VBox([
    W.HBox([txt_exp_ms, txt_gain, ck_auto_gain, sl_gamma, ck_record_raw]),
    W.HBox([dd_ms_az, dd_ms_alt, btn_apply_ms, btn_cal_az, btn_cal_alt, btn_cal_reset, btn_key_reset, btn_rec10]),
    W.HTML("<b>Movimiento manual</b>"),
    W.HBox([
        W.VBox([
            W.HTML("<b>AZ</b>"),
            txt_slew_steps_az,
            txt_slew_delay_az,
            W.HBox([btn_az_left, btn_az_right]),
        ]),
        W.VBox([
            W.HTML("<b>ALT</b>"),
            txt_slew_steps_alt,
            txt_slew_delay_alt,
            W.HBox([btn_alt_up, btn_alt_down]),
        ]),
    ]),
])

tab_track = W.VBox([
    W.HBox([ck_tracking, ck_auto_rls, sl_rls_forget, lab_ard]),
    W.HBox([sl_respmin, sl_lam]),
    W.HBox([sl_bright, sl_hp, sl_sm]),
    W.HBox([sl_kp, sl_ki, sl_kd]),
    W.HBox([sl_abs_every, sl_abs_beta, sl_abs_resp, sl_kref_px]),
    lab_cal,
])

tab_stack = W.VBox([
    W.HBox([ck_stack_enable, btn_stack_start, btn_stack_stop, btn_stack_reset, btn_stack_save]),
    W.HBox([sl_stack_resp, sl_stack_maxrad, sl_hot_z, txt_hot_max]),
    lab_stack,
    img_stack
])

tab_solve = W.VBox([
    W.HBox([btn_detect_stars, btn_plate_solve, btn_accept_solve, txt_pool, txt_min_sep, txt_sigma_k, txt_min_abs]),
    W.HBox([txt_gaia_rad, txt_gmag_max]),
    lab_solve,
    out_plots
])

tab_goto = W.VBox([
    W.HBox([btn_refresh_objs, dd_targets, btn_goto, txt_goto_delay_az, txt_goto_delay_alt]),
    lab_goto
])

tab_log = W.VBox([lab, W.HTML("<b>Log</b>"), log_area])

tabs = W.Tab(children=[tab_cam, tab_track, tab_stack, tab_solve, tab_goto, tab_log])
tabs.set_title(0, "Cámara")
tabs.set_title(1, "Tracking")
tabs.set_title(2, "Stacking")
tabs.set_title(3, "PlateSolve")
tabs.set_title(4, "GoTo")
tabs.set_title(5, "Log")

ui = W.VBox([
    W.HBox([btn_start, btn_stop]),
    tabs,
    img_live,
])
display(ui)


# ============================================================
# Logging (tu estilo)
# ============================================================
t0 = None
log_lines = []
log_lock = threading.Lock()

def log(msg: str):
    global log_lines, t0
    if t0 is None:
        t0 = time.time()
    ts = time.time() - t0
    line = f"[{ts:7.1f}s] {msg}"
    with log_lock:
        log_lines.append(line)
        if len(log_lines) > 900:
            log_lines[:] = log_lines[-900:]

def flush_log_to_widget():
    with log_lock:
        txt = "\n".join(log_lines[-300:])
    log_area.value = txt


# ============================================================
# Action queue (tu patrón)
# ============================================================
actions = queue.Queue()

def enqueue_action(kind: str, payload=None):
    actions.put((kind, payload))
    log(f"UI: action='{kind}' encolada")
    flush_log_to_widget()

def drain_actions(max_items=12):
    out = []
    for _ in range(max_items):
        try:
            out.append(actions.get_nowait())
        except queue.Empty:
            break
    return out

btn_cal_az.on_click(lambda _: enqueue_action("CAL_AZ"))
btn_cal_alt.on_click(lambda _: enqueue_action("CAL_ALT"))
btn_cal_reset.on_click(lambda _: enqueue_action("CAL_RESET"))
btn_key_reset.on_click(lambda _: enqueue_action("KEYFRAME_RESET"))
btn_apply_ms.on_click(lambda _: enqueue_action("APPLY_MS", (int(dd_ms_az.value), int(dd_ms_alt.value))))
btn_rec10.on_click(lambda _: enqueue_action("RECORD", float(RECORD_SECONDS_DEFAULT)))

btn_stack_start.on_click(lambda _: enqueue_action("STACK_START"))
btn_stack_stop.on_click(lambda _: enqueue_action("STACK_STOP"))
btn_stack_reset.on_click(lambda _: enqueue_action("STACK_RESET"))
btn_stack_save.on_click(lambda _: enqueue_action("STACK_SAVE"))

btn_detect_stars.on_click(lambda _: enqueue_action("DETECT_STARS"))
btn_plate_solve.on_click(lambda _: enqueue_action("PLATE_SOLVE"))
btn_accept_solve.on_click(lambda _: enqueue_action("ACCEPT_SOLVE"))

btn_refresh_objs.on_click(lambda _: enqueue_action("REFRESH_OBJECTS"))
btn_goto.on_click(lambda _: enqueue_action("GOTO_TARGET"))


# ============================================================
# Camera open/close (tu código)
# ============================================================
def open_camera():
    if pyPOACamera.GetCameraCount() <= 0:
        raise RuntimeError("No se detectan cámaras.")
    err, props = pyPOACamera.GetCameraProperties(CAM_INDEX)
    if err != pyPOACamera.POAErrors.POA_OK:
        raise RuntimeError("GetCameraProperties falló.")
    cam_id = props.cameraID

    if pyPOACamera.OpenCamera(cam_id) != pyPOACamera.POAErrors.POA_OK:
        raise RuntimeError("OpenCamera falló.")
    if pyPOACamera.InitCamera(cam_id) != pyPOACamera.POAErrors.POA_OK:
        raise RuntimeError("InitCamera falló.")

    pyPOACamera.StopExposure(cam_id)
    pyPOACamera.SetImageStartPos(cam_id, int(ROI_X), int(ROI_Y))
    pyPOACamera.SetImageSize(cam_id, int(ROI_W), int(ROI_H))
    pyPOACamera.SetImageBin(cam_id, int(BIN_HW))
    pyPOACamera.SetImageFormat(cam_id, IMG_FMT)

    _, w, h = pyPOACamera.GetImageSize(cam_id)

    exp_ms = max(1, int(txt_exp_ms.value))
    gain = max(0, int(txt_gain.value))
    auto_gain = bool(ck_auto_gain.value)

    pyPOACamera.SetExp(cam_id, int(exp_ms * 1000), False)
    pyPOACamera.SetGain(cam_id, int(gain), auto_gain)

    if pyPOACamera.StartExposure(cam_id, False) != pyPOACamera.POAErrors.POA_OK:
        raise RuntimeError("StartExposure falló.")
    return cam_id, int(w), int(h)

def close_camera(cam_id):
    try: pyPOACamera.StopExposure(cam_id)
    except: pass
    try: pyPOACamera.CloseCamera(cam_id)
    except: pass

def wait_frame_ready(cam_id, stop_event):
    while True:
        _, ready = pyPOACamera.ImageReady(cam_id)
        if ready or stop_event.is_set():
            return bool(ready)
        time.sleep(0.002)

def get_frame_raw_u16(cam_id, buf_u8: np.ndarray, h_raw: int, w_raw: int):
    err = pyPOACamera.GetImageData(cam_id, buf_u8, 1000)
    if err != pyPOACamera.POAErrors.POA_OK:
        return None
    raw = buf_u8.view("<u2").reshape(h_raw, w_raw)
    return raw.copy()

# ============================================================
# Utils / clamp
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rate_ramp(cur, target, max_delta):
    d = target - cur
    d = clamp(d, -max_delta, +max_delta)
    return cur + d


# ============================================================
# Calibration state (tu estructura)
# ============================================================
CAL = {"AZ_full": None, "ALT_full": None, "A_micro": None, "A_pinv": None}

def _ms_state():
    return int(dd_ms_az.value), int(dd_ms_alt.value)

def compute_A_pinv_dls(A: np.ndarray, lam: float) -> np.ndarray:
    AtA = A.T @ A
    I = np.eye(2, dtype=np.float64)
    M = AtA + (lam * lam) * I
    return np.linalg.inv(M) @ A.T

# ============================================================
# Auto-calibration online (RLS): v = A u + b (tu código)
# ============================================================
AUTO = {"ok": False, "theta": None, "P": None, "A": None, "b": None, "A_pinv": None, "src": "none", "last_upd": None}

def auto_reset(theta=None, src="none"):
    AUTO["ok"] = False
    AUTO["src"] = src
    if theta is None:
        theta = np.array([[0.20, 0.00, 0.0],
                          [0.00, 0.10, 0.0]], dtype=np.float64)
    AUTO["theta"] = theta.astype(np.float64)
    AUTO["P"] = (RLS_P0 * np.eye(3, dtype=np.float64))
    AUTO["A"] = AUTO["theta"][:, :2].copy()
    AUTO["b"] = AUTO["theta"][:, 2].copy()
    AUTO["A_pinv"] = None
    AUTO["last_upd"] = None

def auto_recompute_A_pinv():
    A = AUTO["A"]
    if A is None:
        AUTO["A_pinv"] = None
        AUTO["ok"] = False
        return
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or abs(det) < RLS_MIN_DET:
        AUTO["A_pinv"] = None
        AUTO["ok"] = False
        return
    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = 1e9
    lam = float(sl_lam.value)
    lam_eff = max(lam, 0.15) if cond > RLS_MAX_COND else lam
    try:
        AUTO["A_pinv"] = compute_A_pinv_dls(A, lam_eff)
        AUTO["ok"] = True
    except Exception:
        AUTO["A_pinv"] = None
        AUTO["ok"] = False

def auto_set_from_A(A_micro: np.ndarray, b_pxps=None, src="boot"):
    A_micro = np.array(A_micro, dtype=np.float64).reshape(2, 2)
    if b_pxps is None:
        b_pxps = np.zeros(2, dtype=np.float64)
    b_pxps = np.array(b_pxps, dtype=np.float64).reshape(2,)
    theta = np.concatenate([A_micro, b_pxps.reshape(2,1)], axis=1)  # 2x3
    auto_reset(theta=theta, src=src)
    AUTO["A"] = A_micro.copy()
    AUTO["b"] = b_pxps.copy()
    auto_recompute_A_pinv()

def auto_rls_update(u_az, u_alt, vx, vy, forget):
    if not ck_auto_rls.value:
        return
    if not np.isfinite(vx) or not np.isfinite(vy):
        return
    phi = np.array([float(u_az), float(u_alt), 1.0], dtype=np.float64).reshape(3,1)
    P = AUTO["P"]
    theta = AUTO["theta"]
    if P is None or theta is None:
        auto_reset(src="none")
        P = AUTO["P"]; theta = AUTO["theta"]

    lam = float(forget)
    denom = lam + float((phi.T @ P @ phi)[0,0])
    if denom <= 1e-9 or not np.isfinite(denom):
        return
    K = (P @ phi) / denom

    y = np.array([float(vx), float(vy)], dtype=np.float64).reshape(2,1)
    y_hat = theta @ phi
    err = y - y_hat

    theta_new = theta + (err @ K.T)
    P_new = (P - (K @ (phi.T @ P))) / lam

    AUTO["theta"] = theta_new
    AUTO["P"] = P_new
    AUTO["A"] = theta_new[:, :2].copy()
    AUTO["b"] = theta_new[:, 2].copy()
    AUTO["last_upd"] = time.time()
    auto_recompute_A_pinv()


# ============================================================
# UI labels (tu patrón)
# ============================================================
def update_cal_labels():
    az_ok = "OK" if CAL["AZ_full"] is not None else "?"
    al_ok = "OK" if CAL["ALT_full"] is not None else "?"
    pinv_ok = "OK" if CAL["A_pinv"] is not None else "None"

    auto_ok = "OK" if (AUTO.get("A_pinv", None) is not None and AUTO.get("ok", False)) else "None"
    auto_src = AUTO.get("src", "none")

    az_ms, alt_ms = _ms_state()
    lab_cal.value = (
        f"<b>Cal:</b> AZ={az_ok} ALT={al_ok} | MS(AZ,ALT)=({az_ms},{alt_ms}) | "
        f"A_pinv(manual)={pinv_ok} | AutoA={auto_ok} ({auto_src})"
    )

def recompute_A_micro():
    az_ms, alt_ms = _ms_state()
    if CAL["AZ_full"] is None or CAL["ALT_full"] is None:
        CAL["A_micro"] = None
        CAL["A_pinv"] = None
        update_cal_labels()
        return

    col0_full = np.array(CAL["AZ_full"], dtype=np.float64).reshape(2, 1)
    col1_full = np.array(CAL["ALT_full"], dtype=np.float64).reshape(2, 1)

    col0_micro = col0_full / float(az_ms)
    col1_micro = col1_full / float(alt_ms)
    A = np.concatenate([col0_micro, col1_micro], axis=1)

    det = float(np.linalg.det(A))
    CAL["A_micro"] = A

    lam = float(sl_lam.value)
    try:
        pinv = compute_A_pinv_dls(A, lam)
        CAL["A_pinv"] = pinv
        log(f"CAL: A_micro det={det:+.3e}, λ={lam:.3f}, A={A.tolist()}, A_pinv={pinv.tolist()}")
        if ck_auto_rls.value:
            auto_set_from_A(A_micro=A, b_pxps=np.array([0.0, 0.0]), src="manual_init")
    except Exception as e:
        CAL["A_pinv"] = None
        log(f"CAL: ERROR construyendo A_pinv (det={det:+.3e}, λ={lam:.3f}) -> {e}")

    update_cal_labels()
    flush_log_to_widget()


# ============================================================
# Manual slews (tu patrón)
# ============================================================
def slew(axis: str, direction: str):
    if motor_ser is None:
        log("SLEW: Arduino no conectado")
        flush_log_to_widget()
        return

    if axis == "AZ":
        steps = int(txt_slew_steps_az.value)
        delay_us = int(txt_slew_delay_az.value)
        motor_id = "A"
        # actualiza contador mount
        MOUNT["az_micro"] += (+1.0 if direction=="FWD" else -1.0) * steps
    elif axis == "ALT":
        steps = int(txt_slew_steps_alt.value)
        delay_us = int(txt_slew_delay_alt.value)
        motor_id = "B"
        MOUNT["alt_micro"] += (+1.0 if direction=="FWD" else -1.0) * steps
    else:
        log(f"SLEW: eje desconocido '{axis}'")
        flush_log_to_widget()
        return

    if steps <= 0 or delay_us <= 0:
        log(f"SLEW {axis}: parámetros inválidos (steps={steps}, delay_us={delay_us})")
        flush_log_to_widget()
        return

    arduino_rate(0, 0)
    mover_motor(motor_id, direction, steps, delay_us)
    log(f"SLEW {axis} dir={direction}, steps={steps}, delay_us={delay_us} (RATE 0 0)")
    enqueue_action("KEYFRAME_RESET")
    flush_log_to_widget()

btn_az_left.on_click(lambda _: slew("AZ", "REV"))
btn_az_right.on_click(lambda _: slew("AZ", "FWD"))
btn_alt_up.on_click(lambda _: slew("ALT", "FWD"))
btn_alt_down.on_click(lambda _: slew("ALT", "REV"))


# ============================================================
# Tracker + recording + stacking state (tu base)
# ============================================================
stop_event = threading.Event()
thr = None

S = {
    "cam_id": None,
    "w_raw": None, "h_raw": None,
    "w": None, "h": None,

    "bg_ema": None,

    "prev_reg": None,
    "prev_t": None,
    "fail": 0,

    "vpx": 0.0,
    "vpy": 0.0,
    "vx_inst": 0.0,
    "vy_inst": 0.0,
    "resp_inc": 0.0,

    "key_reg": None,
    "key_t": None,
    "x_hat": 0.0,
    "y_hat": 0.0,
    "abs_last_t": None,
    "abs_resp_last": 0.0,

    "eint_x": 0.0,
    "eint_y": 0.0,

    "rate_az": 0.0,
    "rate_alt": 0.0,

    "mode": "INIT",
    "t_mode": None,

    "rec_on": False,
    "rec_t0": None,
    "rec_secs": 0.0,
    "rec_frames": [],

    "loop": 0,
    "fps_ema": None,

    "boot": {
        "active": False,
        "phase": "IDLE",
        "t_phase": None,
        "t_set": None,
        "samples": [],
        "v_base": None,
        "v_az": None,
        "v_alt": None,
    },

    # último frame guardado para plate solve
    "last_raw": None,
    "last_proc": None,
}

# ---- Plate solve session state
SOLVE = {
    "stars": None,
    "kernel": make_star_kernel(9, 1.8),
    "last_solution": None,
    "accepted": False,
}

# ---- Mosaic full-res stack state
STACK = {
    "enabled": False,
    "active": False,

    # referencia para registro (usamos reg de PROC, pero aplicamos al RAW con factor)
    "ref_reg": None,
    "ref_set": False,

    # acumulación mosaico (float32)
    "sum": None,
    "w": None,

    # offset origen (top-left del frame0 dentro del canvas)
    "origin_x": 0,
    "origin_y": 0,

    # drift acumulado en coords RAW (subpixel)
    "acc_dx_raw": 0.0,
    "acc_dy_raw": 0.0,

    "frames_total": 0,
    "frames_used": 0,
    "preview_counter": 0,

    "last_dx": 0.0,
    "last_dy": 0.0,
    "last_resp": 0.0,
    "last_used": 0,
}

def reset_keyframe(reg_now=None):
    S["key_reg"] = reg_now
    S["key_t"] = time.time()
    S["x_hat"] = 0.0
    S["y_hat"] = 0.0
    S["eint_x"] = 0.0
    S["eint_y"] = 0.0
    S["abs_last_t"] = time.time()
    S["abs_resp_last"] = 0.0
    log("KEYFRAME: reset (x_hat=y_hat=0, integral=0)")
    flush_log_to_widget()

def reset_tracker(mode="STABILIZE"):
    S["prev_reg"] = None
    S["prev_t"] = None
    S["fail"] = 0
    S["vpx"] = 0.0
    S["vpy"] = 0.0
    S["vx_inst"] = 0.0
    S["vy_inst"] = 0.0
    S["resp_inc"] = 0.0
    S["rate_az"] = 0.0
    S["rate_alt"] = 0.0
    S["mode"] = mode
    S["t_mode"] = time.time()
    S["key_reg"] = None
    S["key_t"] = None
    S["x_hat"] = 0.0
    S["y_hat"] = 0.0
    S["eint_x"] = 0.0
    S["eint_y"] = 0.0
    S["abs_last_t"] = None
    S["abs_resp_last"] = 0.0
    log(f"RESET tracker: {mode}")
    flush_log_to_widget()

# ============================================================
# Update camera params live
# ============================================================
def update_camera_params(change=None):
    cam_id = S.get("cam_id", None)
    if cam_id is None:
        return
    exp_ms = max(1, int(txt_exp_ms.value))
    gain = max(0, int(txt_gain.value))
    auto_gain = bool(ck_auto_gain.value)
    try:
        pyPOACamera.SetExp(cam_id, int(exp_ms * 1000), False)
        pyPOACamera.SetGain(cam_id, int(gain), auto_gain)
        log(f"CAM: updated exp={exp_ms}ms gain={gain} auto_gain={auto_gain}")
        flush_log_to_widget()
    except Exception as e:
        log(f"CAM: error al actualizar parámetros: {e}")
        flush_log_to_widget()

txt_exp_ms.observe(update_camera_params, names="value")
txt_gain.observe(update_camera_params, names="value")
ck_auto_gain.observe(update_camera_params, names="value")


# ============================================================
# Capture reg average (para calibración manual)
# ============================================================
def capture_reg_average(cam_id, buf_u8, h_raw, w_raw, n=4, update_bg=False):
    regs = []
    last_proc = None
    for _ in range(n):
        ok = wait_frame_ready(cam_id, stop_event)
        if not ok:
            break
        raw = get_frame_raw_u16(cam_id, buf_u8, h_raw, w_raw)
        if raw is None:
            continue
        if SW_BIN2:
            proc = sw_bin2_u16(raw)
        else:
            proc = raw
        last_proc = proc
        reg, _bg = preprocess_for_phasecorr(
            proc, S["bg_ema"],
            sigma_hp=float(sl_hp.value),
            sigma_smooth=float(sl_sm.value),
            bright_percentile=float(sl_bright.value),
            update_bg=update_bg,
        )
        regs.append(reg)
    if len(regs) == 0:
        return None, last_proc
    return np.mean(regs, axis=0).astype(np.float32), last_proc


# ============================================================
# Manual Calibration (tu código)
# ============================================================
def calibrate_axis(axis_name: str, cam_id, buf_u8, h_raw, w_raw):
    if motor_ser is None:
        log(f"CAL {axis_name}: Arduino no conectado")
        flush_log_to_widget()
        return

    arduino_rate(0, 0)
    bg_backup = S["bg_ema"].copy() if S["bg_ema"] is not None else None
    steps = int(CAL_STEPS_INIT)

    for attempt in range(CAL_TRY_MAX):
        if stop_event.is_set():
            break

        reg0, _ = capture_reg_average(cam_id, buf_u8, h_raw, w_raw, n=4, update_bg=False)
        if reg0 is None:
            log(f"CAL {axis_name}: no pude capturar reg0")
            continue

        if axis_name == "AZ":
            mover_motor("A", "FWD", steps, CAL_DELAY_US)
        else:
            mover_motor("B", "FWD", steps, CAL_DELAY_US)
        time.sleep(0.10)

        reg1, _ = capture_reg_average(cam_id, buf_u8, h_raw, w_raw, n=4, update_bg=False)
        if reg1 is None:
            continue
        dx1, dy1, resp1 = pyramid_phasecorr_delta(reg0, reg1, levels=3)

        if axis_name == "AZ":
            mover_motor("A", "REV", steps, CAL_DELAY_US)
        else:
            mover_motor("B", "REV", steps, CAL_DELAY_US)
        time.sleep(0.10)

        reg2, _ = capture_reg_average(cam_id, buf_u8, h_raw, w_raw, n=4, update_bg=False)
        if reg2 is None:
            continue
        dx2, dy2, resp2 = pyramid_phasecorr_delta(reg1, reg2, levels=3)

        dx = 0.5 * (dx1 - dx2)
        dy = 0.5 * (dy1 - dy2)
        resp = float(min(resp1, resp2))
        mag = float(np.hypot(dx, dy))

        log(f"CAL {axis_name} try#{attempt+1}: steps={steps}, dp≈({dx:+.2f},{dy:+.2f}) |dp|={mag:.2f} resp≈{resp:.3f}")
        flush_log_to_widget()

        if (not np.isfinite(resp)) or (resp < CAL_RESP_MIN):
            steps = min(CAL_STEPS_MAX, steps + 4)
            continue

        if mag < CAL_TARGET_PX_MIN and steps < CAL_STEPS_MAX:
            steps = min(CAL_STEPS_MAX, steps + 4)
            continue

        if mag > CAL_TARGET_PX_MAX and steps > 4:
            steps = max(4, steps // 2)
            continue

        col_micro = (dx / steps, dy / steps)

        az_ms, alt_ms = _ms_state()
        if axis_name == "AZ":
            col_full = (col_micro[0] * az_ms, col_micro[1] * az_ms)
            CAL["AZ_full"] = col_full
        else:
            col_full = (col_micro[0] * alt_ms, col_micro[1] * alt_ms)
            CAL["ALT_full"] = col_full

        log(f"CAL {axis_name}: OK col_full(px/fullstep)=({col_full[0]:+.4e},{col_full[1]:+.4e})")
        recompute_A_micro()

        if bg_backup is not None:
            S["bg_ema"] = bg_backup

        flush_log_to_widget()
        return

    log(f"CAL {axis_name}: falló tras {CAL_TRY_MAX} intentos")
    if bg_backup is not None:
        S["bg_ema"] = bg_backup
    flush_log_to_widget()


# ============================================================
# Recording (.npy) (tu código)
# ============================================================
def start_recording(seconds: float):
    if S["rec_on"]:
        log("RECORD: ya estaba grabando -> ignorado")
        flush_log_to_widget()
        return
    S["rec_on"] = True
    S["rec_t0"] = time.time()
    S["rec_secs"] = float(seconds)
    S["rec_frames"] = []
    mode = "RAW full-res" if ck_record_raw.value else "PROC (bin2 si SW_BIN2)"
    log(f"RECORD: iniciado ({seconds:.1f}s) mode={mode} -> se guardará .npy al terminar")
    flush_log_to_widget()

def maybe_finish_recording():
    if not S["rec_on"]:
        return
    if (time.time() - S["rec_t0"]) >= S["rec_secs"]:
        arr = np.stack(S["rec_frames"], axis=0) if len(S["rec_frames"]) > 0 else None
        S["rec_on"] = False
        S["rec_t0"] = None
        S["rec_secs"] = 0.0
        S["rec_frames"] = []
        if arr is None:
            log("RECORD: sin frames -> no se guardó nada")
            flush_log_to_widget()
            return
        ts = int(time.time())
        fname = f"star_record_{ts}.npy"
        np.save(fname, arr)
        log(f"RECORD: guardado {fname} con shape={arr.shape} dtype={arr.dtype}")
        flush_log_to_widget()


# ============================================================
# Auto-bootstrap state machine (tu código)
# ============================================================
def boot_start():
    if not AUTO_BOOT_ENABLE or motor_ser is None:
        return
    b = S["boot"]
    b["active"] = True
    b["phase"] = "BASE"
    b["t_phase"] = time.time()
    b["t_set"] = time.time()
    b["samples"] = []
    b["v_base"] = None
    b["v_az"] = None
    b["v_alt"] = None
    arduino_rate(0, 0)
    log(f"AUTO_BOOT: iniciado (RATE={AUTO_BOOT_RATE:.1f})")
    flush_log_to_widget()

def boot_collect_sample(vx, vy, resp_ok):
    if not resp_ok:
        return
    S["boot"]["samples"].append((float(vx), float(vy)))

def boot_mean_samples(samples):
    if len(samples) == 0:
        return None
    arr = np.array(samples, dtype=np.float64)
    return arr.mean(axis=0)

def boot_step(now_t):
    b = S["boot"]
    if not b["active"]:
        return False

    if (now_t - b["t_set"]) < AUTO_BOOT_SETTLE_S:
        return True

    phase = b["phase"]
    dur = AUTO_BOOT_BASE_S if phase == "BASE" else AUTO_BOOT_AXIS_S

    if (now_t - b["t_phase"]) >= dur:
        m = boot_mean_samples(b["samples"])
        if m is None or len(b["samples"]) < AUTO_BOOT_MIN_SAMPLES:
            log(f"AUTO_BOOT: fase {phase} con pocas muestras ({len(b['samples'])}) -> reinicio")
            b["phase"] = "BASE"
            b["t_phase"] = now_t
            b["t_set"] = now_t
            b["samples"] = []
            arduino_rate(0, 0)
            flush_log_to_widget()
            return True

        if phase == "BASE":
            b["v_base"] = m
            b["phase"] = "AZ"
            b["t_phase"] = now_t
            b["t_set"] = now_t
            b["samples"] = []
            arduino_rate(+AUTO_BOOT_RATE, 0.0)
            log(f"AUTO_BOOT: BASE ok v0=({m[0]:+.3f},{m[1]:+.3f}) -> AZ")
            flush_log_to_widget()
            return True

        if phase == "AZ":
            b["v_az"] = m
            b["phase"] = "ALT"
            b["t_phase"] = now_t
            b["t_set"] = now_t
            b["samples"] = []
            arduino_rate(0.0, +AUTO_BOOT_RATE)
            log(f"AUTO_BOOT: AZ ok v1=({m[0]:+.3f},{m[1]:+.3f}) -> ALT")
            flush_log_to_widget()
            return True

        if phase == "ALT":
            b["v_alt"] = m
            arduino_rate(0.0, 0.0)

            v0 = b["v_base"]
            v1 = b["v_az"]
            v2 = b["v_alt"]

            col_az = (v1 - v0) / float(AUTO_BOOT_RATE)
            col_alt = (v2 - v0) / float(AUTO_BOOT_RATE)
            A_micro = np.array([[col_az[0], col_alt[0]],
                                [col_az[1], col_alt[1]]], dtype=np.float64)

            det = float(np.linalg.det(A_micro))
            log(f"AUTO_BOOT: ALT ok v2=({m[0]:+.3f},{m[1]:+.3f}) -> A det={det:+.3e}, A={A_micro.tolist()}")
            auto_set_from_A(A_micro=A_micro, b_pxps=v0, src="boot")
            update_cal_labels()

            b["active"] = False
            b["phase"] = "IDLE"
            b["samples"] = []
            flush_log_to_widget()
            return True

    return True


# ============================================================
# STACK MOSAIC FULL-RES
# ============================================================
def stack_label():
    if not ck_stack_enable.value:
        lab_stack.value = "<b>Stack:</b> OFF"
        return
    if not STACK["active"]:
        lab_stack.value = f"<b>Stack:</b> ready total={STACK['frames_total']} used={STACK['frames_used']}"
        return
    lab_stack.value = (
        f"<b>Stack:</b> ACTIVE total={STACK['frames_total']} used={STACK['frames_used']} "
        f"resp={STACK['last_resp']:.3f} dx={STACK['last_dx']:+.2f} dy={STACK['last_dy']:+.2f} used={STACK['last_used']}"
    )

def stack_reset(full=True):
    STACK["enabled"] = bool(ck_stack_enable.value)
    STACK["active"] = False
    STACK["ref_reg"] = None
    STACK["ref_set"] = False
    STACK["sum"] = None
    STACK["w"] = None
    STACK["origin_x"] = 0
    STACK["origin_y"] = 0
    STACK["acc_dx_raw"] = 0.0
    STACK["acc_dy_raw"] = 0.0
    STACK["frames_total"] = 0
    STACK["frames_used"] = 0
    STACK["preview_counter"] = 0
    STACK["last_dx"] = 0.0
    STACK["last_dy"] = 0.0
    STACK["last_resp"] = 0.0
    STACK["last_used"] = 0
    if full:
        img_stack.value = placeholder_jpeg(text="Stack mosaic preview (idle)")
    stack_label()

def _ensure_canvas(new_h, new_w, pad_top, pad_left, pad_bottom, pad_right):
    # pad existing sum/w to accommodate new frame placement
    if STACK["sum"] is None:
        STACK["sum"] = np.zeros((new_h, new_w), dtype=np.float32)
        STACK["w"]   = np.zeros((new_h, new_w), dtype=np.float32)
        STACK["origin_x"] += pad_left
        STACK["origin_y"] += pad_top
        return

    if pad_top==pad_left==pad_bottom==pad_right==0:
        return

    STACK["sum"] = np.pad(STACK["sum"], ((pad_top,pad_bottom),(pad_left,pad_right)), mode="constant", constant_values=0.0)
    STACK["w"]   = np.pad(STACK["w"],   ((pad_top,pad_bottom),(pad_left,pad_right)), mode="constant", constant_values=0.0)
    STACK["origin_x"] += pad_left
    STACK["origin_y"] += pad_top

def stack_start(raw_h, raw_w):
    if not ck_stack_enable.value:
        log("STACK: Enable stacking OFF -> no inicia")
        flush_log_to_widget()
        return
    os.makedirs(STACK_SAVE_DIR, exist_ok=True)

    STACK["enabled"] = True
    STACK["active"] = True

    STACK["ref_reg"] = None
    STACK["ref_set"] = False

    # canvas inicia del tamaño del raw
    STACK["sum"] = np.zeros((raw_h, raw_w), dtype=np.float32)
    STACK["w"]   = np.zeros((raw_h, raw_w), dtype=np.float32)
    STACK["origin_x"] = 0
    STACK["origin_y"] = 0
    STACK["acc_dx_raw"] = 0.0
    STACK["acc_dy_raw"] = 0.0

    STACK["frames_total"] = 0
    STACK["frames_used"] = 0

    log(f"STACK: START MOSAIC (RAW={raw_h}x{raw_w}, tracking_proc={'bin2' if SW_BIN2 else 'raw'})")
    stack_label()
    flush_log_to_widget()

def stack_stop():
    if STACK["active"]:
        STACK["active"] = False
        log("STACK: STOP")
        stack_label()
        flush_log_to_widget()

def stack_preview_update():
    if STACK["sum"] is None or STACK["w"] is None:
        return
    avg = STACK["sum"] / np.maximum(STACK["w"], 1e-6)
    u8 = stretch_to_u8(avg, STACK_PLO, STACK_PHI, gamma=STACK_GAMMA_PREVIEW, ds=2, blur_sigma=0.0)
    jb = jpeg_bytes(u8, quality=85)
    if jb:
        img_stack.value = jb

def stack_save_png():
    if STACK["sum"] is None or STACK["w"] is None:
        log("STACK: no hay nada que guardar")
        flush_log_to_widget()
        return
    os.makedirs(STACK_SAVE_DIR, exist_ok=True)
    avg = STACK["sum"] / np.maximum(STACK["w"], 1e-6)
    s16 = np.clip(avg, 0, 65535).astype(np.uint16)
    ts = int(time.time())
    png_path = os.path.join(STACK_SAVE_DIR, f"stack_mosaic_{ts}.png")
    ok = cv2.imwrite(png_path, s16)
    log(f"STACK: guardado PNG 16-bit en {png_path} ({'OK' if ok else 'FAIL'}) shape={s16.shape}")
    flush_log_to_widget()

def stack_step(raw_u16, proc_u16, reg_proc):
    """
    FULL-RES MOSAIC:
      - registra usando reg en PROC (para ser estable y rápido)
      - aplica shift escalado a RAW
      - expande canvas si la posición se sale
    """
    if (not ck_stack_enable.value) or (not STACK["active"]):
        return

    STACK["frames_total"] += 1

    raw_h, raw_w = raw_u16.shape
    scale_factor = 2.0 if SW_BIN2 else 1.0

    # hot pixels en raw (full-res)
    raw_fix = remove_hot_pixels(raw_u16, hot_z=float(sl_hot_z.value), hot_max=int(txt_hot_max.value))

    # reg para stacking: sobre PROC (pero robusto)
    z = local_zscore_u16(proc_u16, sigma_bg=float(STACK_SIGMA_BG),
                         floor_p=float(STACK_SIGMA_FLOOR_P),
                         z_clip=float(STACK_Z_CLIP))
    reg = make_sparse_reg(z, peak_p=float(STACK_PEAK_P),
                          blur_sigma=float(STACK_PEAK_BLUR),
                          hot_mask=None)

    if STACK["ref_reg"] is None:
        STACK["ref_reg"] = reg.astype(np.float32).copy()
        STACK["ref_set"] = True
        dxp = dyp = 0.0
        resp = 1.0
        used = True
        STACK["acc_dx_raw"] = 0.0
        STACK["acc_dy_raw"] = 0.0
    else:
        dxp, dyp, resp = pyramid_phasecorr_delta(STACK["ref_reg"].astype(np.float32),
                                                 reg.astype(np.float32),
                                                 levels=3)
        rad = float(np.hypot(dxp, dyp))
        used = True
        if (resp < float(sl_stack_resp.value)) or (rad > float(sl_stack_maxrad.value)) or (not np.isfinite(rad)):
            used = False

    STACK["last_dx"] = float(dxp)
    STACK["last_dy"] = float(dyp)
    STACK["last_resp"] = float(resp)
    STACK["last_used"] = int(used)

    if not used:
        stack_label()
        return

    # Acumula drift en RAW coords
    STACK["acc_dx_raw"] += float(dxp * scale_factor)
    STACK["acc_dy_raw"] += float(dyp * scale_factor)

    # Posición donde “va” este frame dentro del canvas:
    # (origin + acc_shift)
    tx = STACK["origin_x"] + STACK["acc_dx_raw"]
    ty = STACK["origin_y"] + STACK["acc_dy_raw"]

    # Bounding box del frame en canvas coords
    x0 = tx
    y0 = ty
    x1 = tx + raw_w
    y1 = ty + raw_h

    # Si se sale, pad canvas
    Hc, Wc = STACK["sum"].shape
    pad_left = int(max(0, math.ceil(-x0)))
    pad_top  = int(max(0, math.ceil(-y0)))
    pad_right = int(max(0, math.ceil(x1 - Wc)))
    pad_bottom= int(max(0, math.ceil(y1 - Hc)))

    if (pad_left or pad_top or pad_right or pad_bottom):
        _ensure_canvas(Hc+pad_top+pad_bottom, Wc+pad_left+pad_right, pad_top, pad_left, pad_bottom, pad_right)
        # recomputa con nuevo origin
        tx = STACK["origin_x"] + STACK["acc_dx_raw"]
        ty = STACK["origin_y"] + STACK["acc_dy_raw"]

    # warp raw_fix hacia canvas en (tx,ty)
    Hc, Wc = STACK["sum"].shape
    M = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty]], dtype=np.float32)

    warped = cv2.warpAffine(raw_fix.astype(np.float32), M, (Wc, Hc),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    mask = (warped > 0).astype(np.float32)  # simple mask
    STACK["sum"] += warped
    STACK["w"]   += mask

    STACK["frames_used"] += 1
    STACK["preview_counter"] += 1
    if STACK["preview_counter"] >= int(STACK_SHOW_EVERY_N):
        STACK["preview_counter"] = 0
        stack_preview_update()

    stack_label()


# ============================================================
# Start/Stop wiring
# ============================================================
def on_start(_):
    global thr, t0, log_lines
    if thr is not None and thr.is_alive():
        return
    stop_event.clear()
    t0 = time.time()
    with log_lock:
        log_lines = []
    img_live.value = placeholder_jpeg(text="Starting...")
    img_stack.value = placeholder_jpeg(text="Stack mosaic preview (idle)")
    flush_log_to_widget()

    if motor_ser is not None:
        arduino_enable(True)
        arduino_rate(0, 0)

    auto_reset(src="none")
    stack_reset(full=True)
    update_cal_labels()

    reset_tracker("STABILIZE")

    thr = threading.Thread(target=live_loop, daemon=True)
    thr.start()

def on_stop(_):
    stop_event.set()
    try:
        if motor_ser is not None:
            arduino_rate(0, 0)
            arduino_enable(False)
    except Exception:
        pass

btn_start.on_click(on_start)
btn_stop.on_click(on_stop)


# ============================================================
# Live loop principal (tu base + hooks para solve/goto/stack mosaic)
# ============================================================
def live_loop():
    cam_id = None
    try:
        log("live_loop iniciado")

        if motor_ser is not None:
            arduino_enable(True)
            arduino_rate(0, 0)
            az_ms, alt_ms = _ms_state()
            r = arduino_set_microsteps(az_ms, alt_ms)
            log(f"MS aplicado: AZ={az_ms} ALT={alt_ms} (arduino='{r or 'no-reply'}')")
            log(f"Arduino habilitado, RATE 0 0, MS(AZ,ALT)=({az_ms},{alt_ms})")
            flush_log_to_widget()

        cam_id, w_raw, h_raw = open_camera()
        S["cam_id"] = cam_id
        S["w_raw"], S["h_raw"] = w_raw, h_raw
        if SW_BIN2:
            S["w"], S["h"] = w_raw // 2, h_raw // 2
        else:
            S["w"], S["h"] = w_raw, h_raw

        S["bg_ema"] = None
        log(f"Cámara abierta: w={w_raw}, h={h_raw} (proc: {S['w']}x{S['h']})")
        flush_log_to_widget()

        buf_u8 = np.zeros(w_raw * h_raw * 2, dtype=np.uint8)

        reset_tracker("STABILIZE")
        update_cal_labels()
        stack_label()

        last_update_t = time.time()

        while not stop_event.is_set():
            ok = wait_frame_ready(cam_id, stop_event)
            if not ok:
                break

            raw = get_frame_raw_u16(cam_id, buf_u8, h_raw, w_raw)
            if raw is None:
                continue

            if SW_BIN2:
                frame_proc = sw_bin2_u16(raw)
            else:
                frame_proc = raw

            # guarda últimos frames para plate solve
            S["last_raw"] = raw
            S["last_proc"] = frame_proc

            # Recording
            if S["rec_on"]:
                if ck_record_raw.value:
                    S["rec_frames"].append(raw.copy())
                else:
                    S["rec_frames"].append(frame_proc.copy())
                maybe_finish_recording()

            # Display live
            u8 = stretch_to_u8(
                frame_proc,
                PLO,
                PHI,
                gamma=float(sl_gamma.value),
                ds=int(DISPLAY_DS),
                blur_sigma=float(DISPLAY_BLUR_SIGMA)
            )
            jb = jpeg_bytes(u8, JPEG_QUALITY)
            if jb:
                img_live.value = jb

            # Preproc tracking
            reg, S["bg_ema"] = preprocess_for_phasecorr(
                frame_proc, S["bg_ema"],
                sigma_hp=float(sl_hp.value),
                sigma_smooth=float(sl_sm.value),
                bright_percentile=float(sl_bright.value),
                update_bg=True
            )

            # Acciones UI
            for kind, payload in drain_actions(max_items=18):
                if kind == "CAL_RESET":
                    CAL["AZ_full"] = None
                    CAL["ALT_full"] = None
                    CAL["A_micro"] = None
                    CAL["A_pinv"] = None
                    auto_reset(src="none")
                    log("CAL: reset completo (manual + auto)")
                    update_cal_labels()
                    flush_log_to_widget()

                elif kind == "CAL_AZ":
                    log("CAL AZ: iniciado (dither y medición)")
                    flush_log_to_widget()
                    calibrate_axis("AZ", cam_id, buf_u8, h_raw, w_raw)

                elif kind == "CAL_ALT":
                    log("CAL ALT: iniciado (dither y medición)")
                    flush_log_to_widget()
                    calibrate_axis("ALT", cam_id, buf_u8, h_raw, w_raw)

                elif kind == "APPLY_MS":
                    az_ms, alt_ms = payload
                    if motor_ser is not None:
                        arduino_rate(0, 0)
                        r = arduino_set_microsteps(az_ms, alt_ms)
                        log(f"MS aplicado: AZ={az_ms} ALT={alt_ms} (arduino='{r or 'no-reply'}')")
                    # actualiza mount ms
                    MOUNT["ms_az"] = int(az_ms)
                    MOUNT["ms_alt"] = int(alt_ms)
                    recompute_A_micro()
                    update_cal_labels()

                elif kind == "KEYFRAME_RESET":
                    S["key_reg"] = "PENDING"

                elif kind == "RECORD":
                    secs = float(payload) if payload is not None else float(RECORD_SECONDS_DEFAULT)
                    start_recording(secs)

                # --- Stacking
                elif kind == "STACK_START":
                    if not ck_stack_enable.value:
                        log("STACK: Enable stacking OFF -> ignorado")
                    else:
                        stack_start(h_raw, w_raw)
                        stack_step(raw, frame_proc, reg)
                    flush_log_to_widget()

                elif kind == "STACK_STOP":
                    stack_stop()

                elif kind == "STACK_RESET":
                    stack_reset(full=True)
                    log("STACK: reset")
                    flush_log_to_widget()

                elif kind == "STACK_SAVE":
                    stack_save_png()

                # --- Plate solve: Detect
                elif kind == "DETECT_STARS":
                    if S["last_proc"] is None:
                        log("DETECT: no hay frame aún")
                        flush_log_to_widget()
                        continue
                    img = S["last_proc"].astype(np.float32)

                    kernel = SOLVE["kernel"]
                    pool = int(txt_pool.value)
                    pooled, idx_local, offsets = convolve_and_pool(img, kernel, pool_size=pool)
                    thr = global_threshold_from_pooled(pooled, min_abs=float(txt_min_abs.value), sigma_k=float(txt_sigma_k.value))
                    stars = detect_star_candidates_with_threshold(
                        pooled, idx_local, offsets,
                        pool_size=pool,
                        min_separation_px=int(txt_min_sep.value),
                        min_score=float(thr),
                    )
                    SOLVE["stars"] = stars
                    SOLVE["last_solution"] = None
                    SOLVE["accepted"] = False
                    lab_solve.value = f"<b>PlateSolve:</b> detected {len(stars)} stars (thr={thr:.2f})"

                    with out_plots:
                        out_plots.clear_output(wait=True)
                        fig = plt.figure(figsize=(8.8, 4.2))
                        ax1 = fig.add_subplot(1,2,1)
                        ax1.set_title("Frame (proc) + detections")
                        ax1.imshow(normalize_to_uint8(img), cmap="gray")
                        if len(stars)>0:
                            ys = [s[0] for s in stars]; xs=[s[1] for s in stars]
                            ax1.scatter(xs, ys, s=25, facecolors="none", edgecolors="lime")
                        ax1.set_axis_off()

                        ax2 = fig.add_subplot(1,2,2)
                        ax2.set_title("Pooled response (norm)")
                        ax2.imshow(normalize_to_uint8(pooled), cmap="magma")
                        ax2.set_axis_off()
                        plt.tight_layout()
                        plt.show()

                    log(f"DETECT: {len(stars)} estrellas")
                    flush_log_to_widget()

                # --- Plate solve: Solve
                elif kind == "PLATE_SOLVE":
                    stars = SOLVE.get("stars", None)
                    if not stars or len(stars) < 8:
                        log("PLATE_SOLVE: detecta estrellas primero (>=8)")
                        flush_log_to_widget()
                        continue

                    # OJO: plate solving debe correr sobre el mismo frame donde detectaste
                    img_h, img_w = S["last_proc"].shape
                    try:
                        sol = solve_plate_from_candidates(
                            stars,
                            (img_h, img_w),
                            gaia_radius_deg=float(txt_gaia_rad.value),
                            gaia_gmag_max=float(txt_gmag_max.value),
                            verbose=True
                        )
                        SOLVE["last_solution"] = sol
                        SOLVE["accepted"] = False

                        if sol is None:
                            lab_solve.value = "<b>PlateSolve:</b> FAILED (no best)"
                            log("PLATE_SOLVE: FAILED (best=None)")
                            flush_log_to_widget()
                        else:
                            ra0 = sol["ra0_deg"]; dec0 = sol["dec0_deg"]
                            scale = sol["scale_arcsec_per_px"]; rot = sol["rot_deg"]
                            nm = len(sol["assign"])
                            lab_solve.value = f"<b>PlateSolve:</b> OK matches={nm} center(RA,Dec)=({ra0:.4f},{dec0:.4f}) scale={scale:.3f}\"/px rot={rot:.2f}°"
                            log(f"PLATE_SOLVE: OK matches={nm} RA0={ra0:.4f} Dec0={dec0:.4f} scale={scale:.3f}\"/px rot={rot:.2f}°")
                            flush_log_to_widget()

                            # plot overlay: matched stars vs gaia tangent coords
                            with out_plots:
                                out_plots.clear_output(wait=True)
                                fig = plt.figure(figsize=(9.0, 4.2))
                                ax = fig.add_subplot(1,2,1)
                                ax.set_title("Matched stars on image")
                                ax.imshow(normalize_to_uint8(S["last_proc"]), cmap="gray")
                                pix = sol["pix_matches_xy"]
                                ax.scatter(pix[:,0], pix[:,1], s=35, facecolors="none", edgecolors="cyan")
                                ax.set_axis_off()

                                ax2 = fig.add_subplot(1,2,2)
                                ax2.set_title("Tangent plane (xi,eta) of matches")
                                xi, eta = _gnomonic_forward(sol["sky_matches_ra"], sol["sky_matches_dec"], ra0, dec0)
                                ax2.scatter(xi, eta, s=25)
                                ax2.set_xlabel("xi (rad)")
                                ax2.set_ylabel("eta (rad)")
                                ax2.grid(True, alpha=0.3)
                                plt.tight_layout()
                                plt.show()

                    except Exception as e:
                        lab_solve.value = f"<b>PlateSolve:</b> ERROR ({e})"
                        log(f"PLATE_SOLVE: ERROR {e}")
                        flush_log_to_widget()

                # --- Plate solve: Accept
                elif kind == "ACCEPT_SOLVE":
                    sol = SOLVE.get("last_solution", None)
                    if sol is None:
                        log("ACCEPT: no hay solución (haz Plate solve primero)")
                        flush_log_to_widget()
                        continue
                    if not _ASTROPY_OK:
                        log("ACCEPT: Astropy no disponible -> no puedo convertir RA/Dec a AltAz para calibrar goto")
                        flush_log_to_widget()
                        continue

                    ra0 = float(sol["ra0_deg"])
                    dec0 = float(sol["dec0_deg"])

                    # convierte a alt/az ahora (momento de aceptación)
                    az_deg, alt_deg = radec_to_altaz(ra0, dec0)

                    # fija calibración mount usando el estado actual de microsteps
                    ms_az, ms_alt = _ms_state()
                    goto_mod.mount_set_calibration(
                        MOUNT,
                        az_center_deg=az_deg,
                        alt_center_deg=alt_deg,
                        az_micro_now=float(MOUNT["az_micro"]),
                        alt_micro_now=float(MOUNT["alt_micro"]),
                        ms_az=ms_az,
                        ms_alt=ms_alt,
                        c_cm_now=float(MOUNT["alt_c_cm"]),
                    )
                    SOLVE["accepted"] = True
                    lab_goto.value = f"<b>GoTo:</b> CALIBRATED at center(AZ,ALT)=({az_deg:.2f},{alt_deg:.2f}) deg"
                    log(f"ACCEPT: goto calibrado con center(AZ,ALT)=({az_deg:.2f},{alt_deg:.2f}) a partir de solve")
                    flush_log_to_widget()

                # --- Objects now
                elif kind == "REFRESH_OBJECTS":
                    df, err = now_planets_altaz()
                    if err is not None:
                        log(f"OBJECTS: {err}")
                        flush_log_to_widget()
                        dd_targets.options = ["(none)"]
                        dd_targets.value = "(none)"
                    else:
                        # sólo sobre horizonte
                        df2 = df[df["alt_deg"] > 0].copy()
                        opts = ["(none)"] + [f"{r['name']} | alt={r['alt_deg']:.1f} az={r['az_deg']:.1f}" for _,r in df2.iterrows()]
                        dd_targets.options = opts
                        dd_targets.value = "(none)"
                        log(f"OBJECTS: {len(df2)} sobre el horizonte")
                        flush_log_to_widget()

                # --- GoTo
                elif kind == "GOTO_TARGET":
                    if dd_targets.value == "(none)":
                        log("GOTO: selecciona un target")
                        flush_log_to_widget()
                        continue
                    if not _ASTROPY_OK:
                        log("GOTO: Astropy no disponible (no puedo computar planetas)")
                        flush_log_to_widget()
                        continue

                    # parse nombre
                    name = dd_targets.value.split("|")[0].strip().lower()
                    df, _ = now_planets_altaz()
                    row = df[df["name"]==name]
                    if len(row)==0:
                        log(f"GOTO: target '{name}' no encontrado")
                        flush_log_to_widget()
                        continue
                    targ_alt = float(row.iloc[0]["alt_deg"])
                    targ_az  = float(row.iloc[0]["az_deg"])

                    goto_mod.mount_goto_altaz(
                        MOUNT,
                        target_az_deg=targ_az,
                        target_alt_deg=targ_alt,
                        arduino_rate=arduino_rate,
                        mover_motor=mover_motor,
                        log_fn=log,
                        flush_fn=flush_log_to_widget,
                        motor_ser=motor_ser,
                        rate_az_uS=float(txt_goto_delay_az.value),
                        rate_alt_uS=float(txt_goto_delay_alt.value),
                        tol_deg=0.05,
                    )

            # Si stacking activo: paso mosaico
            if STACK["active"]:
                stack_step(raw, frame_proc, reg)

            # Keyframe init/reset
            now = time.time()
            if S["key_reg"] is None:
                reset_keyframe(reg_now=reg)
            elif isinstance(S["key_reg"], str) and S["key_reg"] == "PENDING":
                reset_keyframe(reg_now=reg)

            if S["prev_reg"] is None:
                S["prev_reg"] = reg
                S["prev_t"] = now
                continue

            dt = now - S["prev_t"]
            if dt <= 1e-6:
                continue

            dx_inc, dy_inc, resp_inc = phasecorr_delta(S["prev_reg"], reg)
            mag_inc = float(np.hypot(dx_inc, dy_inc))

            good_inc = (
                resp_inc >= float(sl_respmin.value)
                and mag_inc <= MAX_SHIFT_PER_FRAME_PX
                and np.isfinite(mag_inc)
            )
            S["resp_inc"] = float(resp_inc)

            if good_inc:
                S["fail"] = 0
                S["x_hat"] += dx_inc
                S["y_hat"] += dy_inc

                vx = dx_inc / dt
                vy = dy_inc / dt
                S["vx_inst"] = float(vx)
                S["vy_inst"] = float(vy)

                a = 0.18
                S["vpx"] = (1 - a) * S["vpx"] + a * vx
                S["vpy"] = (1 - a) * S["vpy"] + a * vy

                auto_rls_update(
                    u_az=S["rate_az"],
                    u_alt=S["rate_alt"],
                    vx=S["vx_inst"],
                    vy=S["vy_inst"],
                    forget=float(sl_rls_forget.value)
                )

                if S["boot"]["active"]:
                    boot_collect_sample(S["vx_inst"], S["vy_inst"], resp_ok=True)

            else:
                S["fail"] += 1
                if S["fail"] % 6 == 0:
                    log(
                        f"Align fallo: dx={dx_inc:+.1f}, dy={dy_inc:+.1f}, "
                        f"|dp|={mag_inc:.1f}, resp={resp_inc:.3f} (fail={S['fail']})"
                    )
                    flush_log_to_widget()

            S["prev_reg"] = reg
            S["prev_t"] = now

            if S["fail"] >= FAIL_RESET_N:
                log("Align perdido: demasiados fallos -> RATE 0 0 y STABILIZE + reset keyframe")
                if motor_ser is not None:
                    arduino_rate(0, 0)
                S["rate_az"] = 0.0
                S["rate_alt"] = 0.0
                reset_tracker("STABILIZE")
                last_update_t = time.time()
                continue

            # ABS correction
            if S["key_reg"] is not None and isinstance(S["key_reg"], np.ndarray):
                if (S["abs_last_t"] is None) or ((now - S["abs_last_t"]) >= float(sl_abs_every.value)):
                    dx_abs, dy_abs, resp_abs = pyramid_phasecorr_delta(S["key_reg"], reg, levels=3)
                    S["abs_last_t"] = now
                    S["abs_resp_last"] = float(resp_abs)
                    mag_abs = float(np.hypot(dx_abs, dy_abs))

                    if (
                        resp_abs >= float(sl_abs_resp.value)
                        and mag_abs <= ABS_MAX_PX
                        and np.isfinite(mag_abs)
                    ):
                        beta = float(sl_abs_beta.value)
                        S["x_hat"] = (1 - beta) * S["x_hat"] + beta * dx_abs
                        S["y_hat"] = (1 - beta) * S["y_hat"] + beta * dy_abs

            # AUTO_BOOT
            if S["boot"]["active"]:
                boot_step(now)

            # Mode machine
            if S["mode"] == "STABILIZE":
                if (now - S["t_mode"]) >= OBSERVE_S:
                    log(f"OBSERVE terminado: v_est=({S['vpx']:+.3f},{S['vpy']:+.3f}) px/s")
                    if ck_tracking.value:
                        if (
                            AUTO_BOOT_ENABLE
                            and ck_auto_rls.value
                            and (CAL["A_pinv"] is None)
                            and (AUTO["A_pinv"] is None)
                        ):
                            S["mode"] = "AUTOBOOT"
                            log("MODE -> AUTOBOOT (estimación inicial A sin botones)")
                            S["rate_az"] = 0.0
                            S["rate_alt"] = 0.0
                            if motor_ser is not None:
                                arduino_rate(0, 0)
                            boot_start()
                        else:
                            S["mode"] = "TRACK"
                            log("MODE -> TRACK (PI + keyframe)")
                            S["rate_az"] = 0.0
                            S["rate_alt"] = 0.0
                            if motor_ser is not None:
                                arduino_rate(0, 0)
                    else:
                        S["mode"] = "IDLE"
                        log("MODE -> IDLE (tracking OFF)")
                    S["t_mode"] = now
                    last_update_t = now
                    update_cal_labels()
                    flush_log_to_widget()

            elif S["mode"] == "AUTOBOOT":
                if not ck_tracking.value:
                    log("AUTOBOOT: tracking OFF -> IDLE")
                    if motor_ser is not None:
                        arduino_rate(0, 0)
                    S["rate_az"] = 0.0
                    S["rate_alt"] = 0.0
                    S["boot"]["active"] = False
                    S["mode"] = "IDLE"
                else:
                    if (not S["boot"]["active"]) and (AUTO["A_pinv"] is not None):
                        log("AUTOBOOT: listo -> TRACK")
                        reset_keyframe(reg_now=reg)
                        S["mode"] = "TRACK"
                        last_update_t = now

            elif S["mode"] == "IDLE":
                if ck_tracking.value:
                    reset_tracker("STABILIZE")
                    last_update_t = time.time()

            elif S["mode"] == "TRACK":
                if not ck_tracking.value:
                    log("Tracking OFF por usuario -> RATE 0 0 y STABILIZE")
                    if motor_ser is not None:
                        arduino_rate(0, 0)
                    S["rate_az"] = 0.0
                    S["rate_alt"] = 0.0
                    reset_tracker("STABILIZE")
                    last_update_t = time.time()
                else:
                    if (now - last_update_t) >= UPDATE_S:
                        A_pinv_use = CAL["A_pinv"] if CAL["A_pinv"] is not None else AUTO["A_pinv"]
                        b_use = (
                            AUTO["b"]
                            if (AUTO["b"] is not None and AUTO["ok"])
                            else np.array([0.0, 0.0], dtype=np.float64)
                        )

                        if A_pinv_use is None:
                            log("TRACK: falta A_pinv (manual y auto). Esperando/boot/calibrar...")
                            if motor_ser is not None:
                                arduino_rate(0, 0)
                            S["rate_az"] = 0.0
                            S["rate_alt"] = 0.0
                        else:
                            ex = float(S["x_hat"])
                            ey = float(S["y_hat"])

                            S["eint_x"] = clamp(S["eint_x"] + ex * UPDATE_S, -EINT_CLAMP, +EINT_CLAMP)
                            S["eint_y"] = clamp(S["eint_y"] + ey * UPDATE_S, -EINT_CLAMP, +EINT_CLAMP)

                            Kp = float(sl_kp.value)
                            Ki = float(sl_ki.value)
                            Kd = float(sl_kd.value)

                            vx_d = float(S["vpx"])
                            vy_d = float(S["vpy"])

                            v_cmd_x = -(Kp * ex + Ki * S["eint_x"] + Kd * vx_d)
                            v_cmd_y = -(Kp * ey + Ki * S["eint_y"] + Kd * vy_d)

                            v_target = np.array(
                                [[v_cmd_x - float(b_use[0])],
                                 [v_cmd_y - float(b_use[1])]],
                                dtype=np.float64
                            )

                            u_dot = (A_pinv_use @ v_target).reshape(-1)

                            rate_az_t  = clamp(float(u_dot[0]), -RATE_MAX, +RATE_MAX)
                            rate_alt_t = clamp(float(u_dot[1]), -RATE_MAX, +RATE_MAX)

                            rate_az  = rate_ramp(S["rate_az"],  rate_az_t,  RATE_SLEW_PER_UPDATE)
                            rate_alt = rate_ramp(S["rate_alt"], rate_alt_t, RATE_SLEW_PER_UPDATE)

                            S["rate_az"] = rate_az
                            S["rate_alt"] = rate_alt

                            if motor_ser is not None:
                                arduino_rate(rate_az, rate_alt)

                            e_mag = float(np.hypot(ex, ey))
                            if (
                                e_mag <= float(sl_kref_px.value)
                                and S["abs_resp_last"] >= float(sl_abs_resp.value)
                            ):
                                reset_keyframe(reg_now=reg)

                            src = (
                                "manual" if CAL["A_pinv"] is not None
                                else ("auto" if AUTO["A_pinv"] is not None else "none")
                            )
                            log(
                                f"TRACK[{src}]: e=({ex:+.2f},{ey:+.2f})px |e|={e_mag:.2f} "
                                f"v_est=({vx_d:+.2f},{vy_d:+.2f})px/s "
                                f"b=({float(b_use[0]):+.2f},{float(b_use[1]):+.2f}) "
                                f"abs_resp={S['abs_resp_last']:.3f} -> "
                                f"RATE=({rate_az:+.2f},{rate_alt:+.2f})"
                            )
                            flush_log_to_widget()

                        last_update_t = now
                        update_cal_labels()

            # FPS + status
            S["loop"] += 1
            fps = 1.0 / max(1e-6, dt)
            S["fps_ema"] = fps if S["fps_ema"] is None else (0.9 * S["fps_ema"] + 0.1 * fps)

            lab.value = (
                f"<b>Status:</b> {S['mode']} | Loop={S['loop']} | FPS(ema)={S['fps_ema']:.2f} | "
                f"inc_resp={S['resp_inc']:.3f} | abs_resp={S['abs_resp_last']:.3f} | "
                f"e=({S['x_hat']:+.1f},{S['y_hat']:+.1f})px | "
                f"RATE=({S['rate_az']:+.1f},{S['rate_alt']:+.1f}) µsteps/s"
            )

    except Exception:
        log("ERROR en live_loop (ver traceback en output)")
        flush_log_to_widget()
        traceback.print_exc()
    finally:
        try:
            if motor_ser is not None:
                arduino_rate(0, 0)
                arduino_enable(False)
        except Exception:
            pass
        if cam_id is not None:
            close_camera(cam_id)
        S["cam_id"] = None
        img_live.value = placeholder_jpeg(text="Stopped")
        img_stack.value = placeholder_jpeg(text="Stack mosaic preview (stopped)")
        lab.value = "<b>Status:</b> stopped"
        log("live_loop terminado")
        flush_log_to_widget()


# ============================================================
# Init UI states
# ============================================================
auto_reset(src="none")
update_cal_labels()
stack_reset(full=True)
flush_log_to_widget()

# set initial mount microstep counters at zero; you can set them if you have an encoder
MOUNT["az_micro"] = 0.0
MOUNT["alt_micro"] = 0.0
MOUNT["ms_az"], MOUNT["ms_alt"] = _ms_state()
MOUNT["alt_c_cm"] = ALT_C0_CM

lab_goto.value = "<b>GoTo:</b> not calibrated (do PlateSolve -> Accept)"






