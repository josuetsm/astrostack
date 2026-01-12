# ============================================================
# Live tracking (pyPOACamera) + Arduino STEP/DIR + Microsteps
# + Live-stacking (GRAYSCALE) INTEGRADO usando TU pipeline:
#     - remove_hot_pixels
#     - local_zscore_u16
#     - make_sparse_reg
#     - pyramid_phasecorr_delta
#     - warp_translate + máscara
#
# Cambios solicitados / respetados:
#   - Seguimiento: OK, RATE_MAX = 600
#   - Gamma en panel Cámara
#   - Botones manuales en panel Cámara
#   - Live-stacking NO parte automáticamente
#   - Live-stacking: GRAYSCALE (no color)
#   - Alineación stacking usa "tu" limpieza/reg (no la del tracking)
#   - Guardado stacking: SOLO PNG (16-bit)
#   - No cambio parámetros que no pediste (exp sigue 100ms)
#   - Log restaurado (con timestamps)
#
# Nota sobre tu preocupación (548x972):
#   Eso es EXACTAMENTE el tamaño de "proc" cuando SW_BIN2=True con ROI 1096x1944:
#     1096/2=548 y 1944/2=972.
#   En este código:
#     - Tracking/stacking usan la imagen procesada (proc) si SW_BIN2=True.
#     - Recording (.npy) se hace en "proc" por defecto (como antes).
#     - Si quieres que el .npy sea FULL-RES RAW, activa "Record RAW full-res".
#
# Requisitos:
#   numpy, opencv-python, ipywidgets, pyPOACamera, pyserial
# ============================================================

import os, time, threading, traceback, queue
import numpy as np
import cv2
import ipywidgets as W
from IPython.display import display

from astrostack import stacking as stacking_mod
from astrostack import tracking as tracking_mod

import pyPOACamera
import serial

# ============================================================
# CONFIG (mantengo defaults que ya tenías)
# ============================================================
# Camera
CAM_INDEX = 0
ROI_X, ROI_Y = 0, 0
ROI_W, ROI_H = 1944, 1096
BIN_HW = 1
IMG_FMT = pyPOACamera.POAImgFormat.POA_RAW16

SW_BIN2 = True              # 2x2 bin por software -> gris (promedio 2x2 sobre Bayer RAW)
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

# Tracking preproc (robusto y liviano)
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
# LIVE-STACKING (TU pipeline) CONFIG
# ============================================================
STACK_ENABLE_DEFAULT = False      # NO partir automáticamente
STACK_SHOW_EVERY_N = 2            # refresco UI cada N frames usados

# Stacking preproc (tal cual tu script)
STACK_SIGMA_BG   = 35.0
STACK_SIGMA_FLOOR_P = 10.0
STACK_Z_CLIP     = 6.0
STACK_PEAK_P     = 99.75
STACK_PEAK_BLUR  = 1.0

STACK_RESP_MIN_INIT = 0.05
STACK_MAX_RAD_INIT  = 400.0

STACK_HOT_Z_INIT    = 12.0
STACK_HOT_MAX_INIT  = 200

STACK_PLO, STACK_PHI = 5.0, 99.7   # stretch para preview en u8 (solo UI)
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
    # devuelve dx,dy tal que warp(cur, dx,dy) ~ ref
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

# ============================================================
# UI
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

# Record RAW toggle (solo agrega opción; por defecto mantiene comportamiento previo)
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
ck_stack_enable = W.Checkbox(description="Enable stacking", value=STACK_ENABLE_DEFAULT)
btn_stack_start = W.Button(description="Start stack", button_style="success", layout=W.Layout(width="120px"))
btn_stack_stop  = W.Button(description="Stop stack",  button_style="warning", layout=W.Layout(width="120px"))
btn_stack_reset = W.Button(description="Reset stack", layout=W.Layout(width="120px"))
btn_stack_save  = W.Button(description="Save PNG",    layout=W.Layout(width="120px"))

sl_stack_resp = W.BoundedFloatText(description="RESP_MIN", min=0.01, max=0.40, step=0.01,
                                   value=STACK_RESP_MIN_INIT, layout=W.Layout(width="200px"))
sl_stack_maxrad = W.BoundedFloatText(description="MAX_RAD", min=20.0, max=3000.0, step=10.0,
                                     value=STACK_MAX_RAD_INIT, layout=W.Layout(width="200px"))

sl_hot_z = W.BoundedFloatText(description="HOT_Z", min=4.0, max=40.0, step=0.5,
                              value=STACK_HOT_Z_INIT, layout=W.Layout(width="200px"))
txt_hot_max = W.BoundedIntText(description="HOT_MAX", min=0, max=5000, step=10,
                               value=int(STACK_HOT_MAX_INIT), layout=W.Layout(width="200px"))

lab_ard = W.HTML(value=ARD_STATE_STR)
lab = W.HTML(value="<b>Status:</b> idle")
lab_cal = W.HTML(value="<b>Cal:</b> AZ=? ALT=? | A_pinv=None | AutoA=None")
lab_stack = W.HTML(value="<b>Stack:</b> OFF")

img_live = W.Image(format="jpeg", value=placeholder_jpeg(text="Idle - press Start"))
img_live.layout = W.Layout(width=IMG_WIDTH_PX)

# Stacking preview (separado)
img_stack = W.Image(format="jpeg", value=placeholder_jpeg(text="Stack preview (idle)"))
img_stack.layout = W.Layout(width=IMG_WIDTH_PX)

log_area = W.Textarea(value="", layout=W.Layout(width="980px", height="240px"))
log_area.disabled = True

# Tabs
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

tab_log = W.VBox([lab, W.HTML("<b>Log</b>"), log_area])

tabs = W.Tab(children=[tab_cam, tab_track, tab_stack, tab_log])
tabs.set_title(0, "Cámara")
tabs.set_title(1, "Tracking")
tabs.set_title(2, "Stacking")
tabs.set_title(3, "Log")

ui = W.VBox([
    W.HBox([btn_start, btn_stop]),
    tabs,
    img_live,
])
display(ui)

# ============================================================
# Logging
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
# Action queue
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

# ============================================================
# Camera open/close (pyPOACamera)
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
# Calibration state
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
# Auto-calibration online (RLS): v = A u + b
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
# UI labels
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
# Manual slews
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
    elif axis == "ALT":
        steps = int(txt_slew_steps_alt.value)
        delay_us = int(txt_slew_delay_alt.value)
        motor_id = "B"
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
# Tracker + recording + stacking state
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
    }
}

# Stacking state (fixed canvas, como tu script)
STACK = {
    "enabled": False,
    "active": False,
    "pending_start": False,

    "H2": None,
    "W2": None,
    "ones": None,

    "ref_reg": None,
    "stack_sum": None,
    "stack_w": None,
    "frames_total": 0,
    "frames_used": 0,

    "last_dx": 0.0,
    "last_dy": 0.0,
    "last_resp": 0.0,
    "last_used": 0,

    "preview_counter": 0,
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
# Manual Calibration
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
# Recording (.npy)
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
# Auto-bootstrap state machine
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
# STACKING (integrado) - funciones
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
    STACK["pending_start"] = False

    STACK["ref_reg"] = None
    STACK["stack_sum"] = None
    STACK["stack_w"] = None

    STACK["frames_total"] = 0
    STACK["frames_used"] = 0

    STACK["last_dx"] = 0.0
    STACK["last_dy"] = 0.0
    STACK["last_resp"] = 0.0
    STACK["last_used"] = 0

    STACK["preview_counter"] = 0

    if full:
        img_stack.value = placeholder_jpeg(text="Stack preview (idle)")
    stack_label()

def stack_start(H2, W2):
    if not ck_stack_enable.value:
        log("STACK: Enable stacking está OFF -> no inicia")
        flush_log_to_widget()
        return
    os.makedirs(STACK_SAVE_DIR, exist_ok=True)

    STACK["enabled"] = True
    STACK["active"] = True
    STACK["pending_start"] = False

    STACK["H2"], STACK["W2"] = int(H2), int(W2)
    STACK["ones"] = np.ones((H2, W2), dtype=np.float32)

    STACK["ref_reg"] = None
    STACK["stack_sum"] = np.zeros((H2, W2), dtype=np.float32)
    STACK["stack_w"] = np.zeros((H2, W2), dtype=np.float32)

    STACK["frames_total"] = 0
    STACK["frames_used"] = 0

    log(f"STACK: START (H2xW2={H2}x{W2}, SW_BIN2={SW_BIN2})")
    stack_label()
    flush_log_to_widget()

def stack_stop():
    if STACK["active"]:
        STACK["active"] = False
        log("STACK: STOP")
        stack_label()
        flush_log_to_widget()

def stack_preview_update():
    if STACK["stack_sum"] is None or STACK["stack_w"] is None:
        return
    avg = STACK["stack_sum"] / np.maximum(STACK["stack_w"], 1e-6)
    u8 = stretch_to_u8(avg, STACK_PLO, STACK_PHI, gamma=STACK_GAMMA_PREVIEW, ds=1, blur_sigma=0.0)
    jb = jpeg_bytes(u8, quality=85)
    if jb:
        img_stack.value = jb

def stack_save_png():
    if STACK["stack_sum"] is None or STACK["stack_w"] is None:
        log("STACK: no hay nada que guardar")
        flush_log_to_widget()
        return
    os.makedirs(STACK_SAVE_DIR, exist_ok=True)
    avg = STACK["stack_sum"] / np.maximum(STACK["stack_w"], 1e-6)

    # Guardado 16-bit lineal (tal cual tu script)
    s16 = np.clip(avg, 0, 65535).astype(np.uint16)
    ts = int(time.time())
    png_path = os.path.join(STACK_SAVE_DIR, f"stack_{ts}.png")
    ok = cv2.imwrite(png_path, s16)
    log(f"STACK: guardado PNG 16-bit en {png_path} ({'OK' if ok else 'FAIL'}) shape={s16.shape}")
    flush_log_to_widget()

def stack_step(frame_proc_u16):
    """
    Implementa EXACTAMENTE el loop de tu script, pero frame a frame.
    frame_proc_u16: uint16 2D (ya en H2xW2, o sea, después de SW_BIN2 si SW_BIN2=True).
    """
    if (not ck_stack_enable.value) or (not STACK["active"]):
        return

    STACK["frames_total"] += 1

    # 1) Hot pixels (por frame)
    frame_fix = remove_hot_pixels(frame_proc_u16, hot_z=float(sl_hot_z.value), hot_max=int(txt_hot_max.value))

    # 2) Imagen para registro: local zscore + sparse reg
    z = local_zscore_u16(frame_fix, sigma_bg=float(STACK_SIGMA_BG),
                         floor_p=float(STACK_SIGMA_FLOOR_P),
                         z_clip=float(STACK_Z_CLIP))
    reg = make_sparse_reg(z, peak_p=float(STACK_PEAK_P),
                          blur_sigma=float(STACK_PEAK_BLUR),
                          hot_mask=None)

    # 3) Si no hay referencia, setearla
    if STACK["ref_reg"] is None:
        STACK["ref_reg"] = reg.astype(np.float32).copy()
        dx = dy = 0.0
        resp = 1.0
        used = True
        warped = frame_fix.astype(np.float32)
        wmask  = STACK["ones"].copy()
    else:
        # Alineación contra ref_reg (pirámide 3 niveles como tu script)
        dx, dy, resp = pyramid_phasecorr_delta(STACK["ref_reg"].astype(np.float32),
                                               reg.astype(np.float32),
                                               levels=3)
        rad = float(np.hypot(dx, dy))
        used = True
        if (resp < float(sl_stack_resp.value)) or (rad > float(sl_stack_maxrad.value)) or (not np.isfinite(rad)):
            used = False
            warped = None
            wmask = None
        else:
            warped = warp_translate(frame_fix.astype(np.float32), dx, dy, is_mask=False)
            wmask  = warp_translate(STACK["ones"], dx, dy, is_mask=True)

    STACK["last_dx"] = float(dx)
    STACK["last_dy"] = float(dy)
    STACK["last_resp"] = float(resp)
    STACK["last_used"] = int(used)

    if used:
        STACK["frames_used"] += 1
        STACK["stack_sum"] += warped
        STACK["stack_w"]   += wmask
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
    img_stack.value = placeholder_jpeg(text="Stack preview (idle)")
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
# Live loop principal
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

            # proc frame (grayscale) usado por tracking/stacking
            if SW_BIN2:
                frame_proc = sw_bin2_u16(raw)
            else:
                frame_proc = raw

            # Recording: RAW o PROC según checkbox
            if S["rec_on"]:
                if ck_record_raw.value:
                    S["rec_frames"].append(raw.copy())
                else:
                    S["rec_frames"].append(frame_proc.copy())
                maybe_finish_recording()

            # Display live cámara
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

            # Acciones UI
            for kind, payload in drain_actions(max_items=14):
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
                    recompute_A_micro()
                    update_cal_labels()

                elif kind == "KEYFRAME_RESET":
                    S["key_reg"] = "PENDING"

                elif kind == "RECORD":
                    secs = float(payload) if payload is not None else float(RECORD_SECONDS_DEFAULT)
                    start_recording(secs)

                # Stacking actions
                elif kind == "STACK_START":
                    if not ck_stack_enable.value:
                        log("STACK: Enable stacking OFF -> ignorado")
                    else:
                        # iniciar con tamaño actual proc
                        stack_start(S["h"], S["w"])
                        # y consumir inmediatamente el frame actual como primer paso (no esperar)
                        stack_step(frame_proc)
                    flush_log_to_widget()

                elif kind == "STACK_STOP":
                    stack_stop()

                elif kind == "STACK_RESET":
                    stack_reset(full=True)
                    log("STACK: reset")
                    flush_log_to_widget()

                elif kind == "STACK_SAVE":
                    stack_save_png()

            # Si stacking activo, sumar este frame (en proc)
            if STACK["active"]:
                stack_step(frame_proc)

            # Preproc tracking (separado)
            reg, S["bg_ema"] = preprocess_for_phasecorr(
                frame_proc, S["bg_ema"],
                sigma_hp=float(sl_hp.value),
                sigma_smooth=float(sl_sm.value),
                bright_percentile=float(sl_bright.value),
                update_bg=True
            )

            now = time.time()

            # Keyframe init/reset
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
        img_stack.value = placeholder_jpeg(text="Stack preview (stopped)")
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




