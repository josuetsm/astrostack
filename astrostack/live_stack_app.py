# app.py
# Live Stacking (shift-and-add) para Player One Mars-C con pyPOACamera
# UI: una sola ventana OpenCV, con:
#   - Botones clickeables
#   - Spinners tipo FireCapture (valor + flechas ▲▼), sin sliders
#   - Tooltips al hover (dwell)
#   - Entrada directa: click en valor -> escribe -> Enter aplica, Esc cancela
#   - Modo rápido: mantener presionado ▲/▼ repite + aceleración
#       * Shift + mantener presionado => step_fast inmediato
#
# Debayer:
#   - Toggle Debayer (color) + patrón Bayer + Debayer HQ (VNG)
#   - Siempre capturamos RAW8 para consistencia y rapidez.
#
# Unicode/tildes:
#   - Con opencv-contrib-python: usa cv2.freetype + TTF => Unicode OK
#   - Sin contrib: fallback ASCII (sin tildes) para evitar "M??ximo"
#
# Requisitos:
#   pip install numpy opencv-python pyPOACamera
#   (opcional, recomendado para tildes) pip install opencv-contrib-python
#
# Uso:
#   python app.py

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict, Union

import numpy as np
import cv2
from . import pyPOACamera

cv2.setUseOptimized(True)


# =========================
# Text rendering (Unicode if possible)
# =========================
def _find_system_ttf() -> Optional[str]:
    candidates = [
        # macOS common
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        # linux common
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    env = Path.home() / ".live_stack_font.ttf"
    if env.exists():
        return str(env)
    return None


class TextRenderer:
    def __init__(self):
        self.use_freetype = False
        self.ft = None
        self.ttf_path = None

        if hasattr(cv2, "freetype"):
            try:
                ft = cv2.freetype.createFreeType2()
                ttf = _find_system_ttf()
                if ttf:
                    ft.loadFontData(fontFileName=ttf, id=0)
                    self.use_freetype = True
                    self.ft = ft
                    self.ttf_path = ttf
            except Exception:
                self.use_freetype = False
                self.ft = None
                self.ttf_path = None

    @staticmethod
    def _ascii_fallback(s: str) -> str:
        s2 = unicodedata.normalize("NFKD", s)
        s2 = s2.encode("ascii", "ignore").decode("ascii")
        return s2

    def safe(self, s: str) -> str:
        return s if self.use_freetype else self._ascii_fallback(s)

    def put(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_face: int,
        font_scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        line_type: int = cv2.LINE_AA,
        height_px: Optional[int] = None,
    ):
        text = self.safe(text)
        if not self.use_freetype:
            cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)
            return

        if height_px is None:
            height_px = int(max(10, 22 * font_scale))
        try:
            self.ft.putText(
                img, text, org, height_px, color,
                thickness=max(1, thickness), line_type=line_type, bottomLeftOrigin=False
            )
        except Exception:
            cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)

    def get_text_size(
        self,
        text: str,
        font_face: int,
        font_scale: float,
        thickness: int,
        height_px: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], int]:
        text = self.safe(text)
        if not self.use_freetype:
            return cv2.getTextSize(text, font_face, font_scale, thickness)
        if height_px is None:
            height_px = int(max(10, 22 * font_scale))
        try:
            (w, h), baseline = self.ft.getTextSize(text, height_px, thickness=max(1, thickness))
            return (w, h), baseline
        except Exception:
            return cv2.getTextSize(text, font_face, font_scale, thickness)


TR = TextRenderer()


# =========================
# SDK helpers
# =========================
def ensure_ok(err, where: str = ""):
    if err != pyPOACamera.POAErrors.POA_OK:
        try:
            msg = pyPOACamera.GetErrorString(err)
        except Exception:
            msg = str(err)
        raise RuntimeError(f"{where} failed: {err} ({msg})")


def pick_first_camera():
    cnt = pyPOACamera.GetCameraCount()
    if cnt <= 0:
        raise RuntimeError("No hay camaras Player One conectadas.")
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


# =========================
# Image utils
# =========================
def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        gray = img[..., 0] if img.ndim >= 3 else img

    if gray.dtype == np.uint8:
        return gray
    if gray.dtype == np.uint16:
        return (gray >> 8).astype(np.uint8)

    g = np.clip(gray.astype(np.float32), 0, 255)
    return g.astype(np.uint8)


def dog_highpass(gray_u8: np.ndarray, s1: float = 1.2, s2: float = 3.0) -> np.ndarray:
    g = gray_u8.astype(np.float32) / 255.0
    a = cv2.GaussianBlur(g, (0, 0), s1)
    b = cv2.GaussianBlur(g, (0, 0), s2)
    return (a - b).astype(np.float32)


def sharpness_laplacian(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_32F).var())


def phase_corr_shift(ref_hp: np.ndarray, img_hp: np.ndarray) -> Tuple[float, float, float]:
    (dx, dy), response = cv2.phaseCorrelate(ref_hp, img_hp)
    return float(dx), float(dy), float(response)


def warp_shift(img_f32: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = img_f32.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img_f32, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def percentile_stretch_u8(img_u8: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    if img_u8.size == 0:
        return img_u8
    a = np.percentile(img_u8, lo)
    b = np.percentile(img_u8, hi)
    if b <= a + 1e-6:
        return img_u8
    x = img_u8.astype(np.float32)
    x = (x - a) * (255.0 / (b - a))
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_gamma_u8(img_u8: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return img_u8
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_u8, lut)


def star_proxy_count(gray_u8: np.ndarray) -> int:
    g = cv2.medianBlur(gray_u8, 3)
    thr = np.percentile(g, 99.5)
    if thr < 5:
        return 0
    bw = (g >= thr).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cnt = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 <= area <= 80:
            cnt += 1
    return cnt


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# Debayer helpers
# =========================
_BAYER_GRAY = {
    "RGGB": cv2.COLOR_BAYER_RG2GRAY,
    "BGGR": cv2.COLOR_BAYER_BG2GRAY,
    "GBRG": cv2.COLOR_BAYER_GB2GRAY,
    "GRBG": cv2.COLOR_BAYER_GR2GRAY,
}
_BAYER_BGR = {
    "RGGB": cv2.COLOR_BAYER_RG2BGR,
    "BGGR": cv2.COLOR_BAYER_BG2BGR,
    "GBRG": cv2.COLOR_BAYER_GB2BGR,
    "GRBG": cv2.COLOR_BAYER_GR2BGR,
}
# VNG may not be compiled in all builds; we guard it.
_BAYER_BGR_VNG = {}
for k, base_code in _BAYER_BGR.items():
    name = f"COLOR_BAYER_{k[:2]}2BGR_VNG"
    # OpenCV uses RG, BG, GB, GR not full tokens; we map explicitly:
    # We'll fill manually with getattr if present.
_BAYER_BGR_VNG = {
    "RGGB": getattr(cv2, "COLOR_BAYER_RG2BGR_VNG", None),
    "BGGR": getattr(cv2, "COLOR_BAYER_BG2BGR_VNG", None),
    "GBRG": getattr(cv2, "COLOR_BAYER_GB2BGR_VNG", None),
    "GRBG": getattr(cv2, "COLOR_BAYER_GR2BGR_VNG", None),
}


def demosaic_raw8(raw: np.ndarray, pattern: str, want_color: bool, hq: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    raw: (H,W) uint8 Bayer mosaic
    returns: (gray_u8, bgr_u8)
    """
    if raw.ndim != 2:
        raw = ensure_gray_u8(raw)  # salvage
    raw_u8 = raw.astype(np.uint8, copy=False)

    pat = pattern if pattern in _BAYER_GRAY else "RGGB"
    gray = cv2.cvtColor(raw_u8, _BAYER_GRAY[pat])

    if not want_color:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray, bgr

    vng = _BAYER_BGR_VNG.get(pat, None) if hq else None
    code = vng if (vng is not None) else _BAYER_BGR[pat]
    bgr = cv2.cvtColor(raw_u8, code)
    return gray, bgr


# =========================
# Live stacker
# =========================
@dataclass
class StackConfig:
    roi: int = 1024
    max_shift_frac: float = 0.35

    min_sharpness: float = 2.0
    min_response_lock: float = 0.02
    min_response_reacq: float = 0.03
    min_star_proxy: int = 2

    update_ref_every_accepted: int = 50

    use_quality_weight: bool = True
    weight_clip: Tuple[float, float] = (0.25, 2.0)


class LiveStacker:
    def __init__(self, cfg: StackConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = "LOCKED"
        self.paused = False
        self.ref_gray_u8: Optional[np.ndarray] = None
        self.ref_hp: Optional[np.ndarray] = None
        self.acc: Optional[np.ndarray] = None
        self.wacc: float = 0.0

        self.seen = 0
        self.accepted = 0
        self.last_dxdy = (0.0, 0.0)
        self.last_scores = (0.0, 0.0)
        self.last_star_proxy = 0

    def _init_ref(self, gray_u8: np.ndarray):
        gray_u8 = gray_u8.astype(np.uint8, copy=False)
        self.ref_gray_u8 = gray_u8.copy()
        self.ref_hp = dog_highpass(self.ref_gray_u8)
        H, W = gray_u8.shape[:2]
        self.acc = np.zeros((H, W), dtype=np.float32)
        self.wacc = 0.0

    def _compute_weight(self, response: float, sharp: float) -> float:
        if not self.cfg.use_quality_weight:
            return 1.0
        w = (response / max(self.cfg.min_response_lock, 1e-6)) * (sharp / max(self.cfg.min_sharpness, 1e-6))
        return float(np.clip(w, self.cfg.weight_clip[0], self.cfg.weight_clip[1]))

    def _update_ref_from_stack(self):
        st = self.get_stack_u8(raw=True)
        if st is None:
            return
        self.ref_gray_u8 = st
        self.ref_hp = dog_highpass(self.ref_gray_u8)

    def process(self, gray_u8: np.ndarray):
        gray_u8 = gray_u8.astype(np.uint8, copy=False)
        self.seen += 1
        sp = star_proxy_count(gray_u8)
        self.last_star_proxy = sp

        if self.ref_hp is None:
            self._init_ref(gray_u8)
            self.last_scores = (1.0, sharpness_laplacian(gray_u8))
            self.last_dxdy = (0.0, 0.0)
            return

        sharp = sharpness_laplacian(gray_u8)
        img_hp = dog_highpass(gray_u8)
        dx, dy, resp = phase_corr_shift(self.ref_hp, img_hp)

        self.last_dxdy = (dx, dy)
        self.last_scores = (resp, sharp)

        H, W = gray_u8.shape[:2]
        max_shift = self.cfg.max_shift_frac * min(H, W)
        too_far = (abs(dx) > max_shift) or (abs(dy) > max_shift)

        if self.paused:
            return

        if self.state == "LOCKED":
            if too_far or resp < self.cfg.min_response_lock or sp < self.cfg.min_star_proxy:
                self.state = "LOST"
                return
            if sharp < self.cfg.min_sharpness:
                return

            aligned = warp_shift(gray_u8.astype(np.float32), dx, dy)
            w = self._compute_weight(resp, sharp)
            self.acc += aligned * w
            self.wacc += w
            self.accepted += 1

            if self.accepted % max(5, self.cfg.update_ref_every_accepted) == 0:
                self._update_ref_from_stack()
        else:
            if (not too_far) and (resp >= self.cfg.min_response_reacq) and (sp >= self.cfg.min_star_proxy):
                self._init_ref(gray_u8)
                self.state = "LOCKED"

    def get_stack_u8(self, raw: bool = False) -> Optional[np.ndarray]:
        if self.acc is None or self.wacc <= 0:
            return None
        img = (self.acc / self.wacc)
        return np.clip(img, 0, 255).astype(np.uint8)

    def overlay_lines(self) -> List[str]:
        resp, sharp = self.last_scores
        dx, dy = self.last_dxdy
        return [
            f"STATE: {self.state}   {'PAUSED' if self.paused else 'RUNNING'}",
            f"Frames: seen={self.seen}   accepted={self.accepted}",
            f"Score: resp={resp:.3f}   sharp={sharp:.2f}   stars~{self.last_star_proxy}",
            f"Shift: dx={dx:.2f}   dy={dy:.2f}",
        ]


# =========================
# UI widgets
# =========================
FONT_FACE = cv2.FONT_HERSHEY_PLAIN
THICK_UI = 1
THICK_TITLE = 2
THICK_INFO = 1


@dataclass
class UIFonts:
    title: float
    ui: float
    info: float
    small: float
    line_h: int


def compute_ui_fonts(canvas_w: int) -> UIFonts:
    base = float(np.clip(canvas_w / 1400.0, 0.85, 1.45))
    return UIFonts(
        title=2.0 * base,
        ui=1.4 * base,
        info=1.25 * base,
        small=1.10 * base,
        line_h=int(22 * base),
    )


@dataclass
class Button:
    key: str
    label: str
    rect: Tuple[int, int, int, int]
    on_click: Callable[[], None]
    tooltip: str = ""
    toggled: Callable[[], bool] = lambda: False


class UIControl:
    key: str
    label: str
    tooltip: str
    rect: Tuple[int, int, int, int]

    def draw(self, canvas: np.ndarray, fonts: UIFonts, hovered: bool, focused: bool):
        raise NotImplementedError

    def hit(self, x: int, y: int) -> bool:
        rx, ry, rw, rh = self.rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def on_click(self, x: int, y: int, flags: int):
        pass

    def on_key(self, key: int):
        pass


@dataclass
class ToggleControl(UIControl):
    key: str
    label: str
    tooltip: str
    rect: Tuple[int, int, int, int]
    getter: Callable[[], bool]
    setter: Callable[[bool], None]

    def draw(self, canvas, fonts, hovered, focused):
        x, y, w, h = self.rect
        base = (75, 75, 75) if not self.getter() else (95, 95, 95)
        border = (190, 190, 190) if hovered else (120, 120, 120)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), base, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

        TR.put(canvas, self.label, (x + 8, y + int(h * 0.70)),
               FONT_FACE, fonts.ui, (235, 235, 235), THICK_UI, height_px=int(18 * fonts.ui))

        pill_w, pill_h = 56, h - 10
        px = x + w - pill_w - 6
        py = y + 5
        col = (50, 150, 50) if self.getter() else (90, 90, 90)
        cv2.rectangle(canvas, (px, py), (px + pill_w, py + pill_h), col, -1)
        cv2.rectangle(canvas, (px, py), (px + pill_w, py + pill_h), (200, 200, 200), 1)
        txt = "ON" if self.getter() else "OFF"
        (tw, th), _ = TR.get_text_size(txt, FONT_FACE, fonts.ui, THICK_UI, height_px=int(18 * fonts.ui))
        TR.put(canvas, txt, (px + (pill_w - tw) // 2, py + (pill_h + th) // 2 - 3),
               FONT_FACE, fonts.ui, (245, 245, 245), THICK_UI, height_px=int(18 * fonts.ui))

    def on_click(self, x: int, y: int, flags: int):
        self.setter(not self.getter())


@dataclass
class SpinnerControl(UIControl):
    key: str
    label: str
    tooltip: str
    rect: Tuple[int, int, int, int]
    getter: Callable[[], Union[int, float]]
    setter: Callable[[Union[int, float]], None]
    vmin: float
    vmax: float
    step: float
    step_fast: float
    fmt: str
    unit: str = ""

    _editing: bool = False
    _edit_text: str = ""

    def arrow_dir(self, x: int, y: int) -> Optional[int]:
        rx, ry, rw, rh = self.rect
        ax_w = 34
        ax_x0 = rx + rw - ax_w
        if x < ax_x0:
            return None
        return +1 if y < ry + rh // 2 else -1

    def draw(self, canvas, fonts, hovered, focused):
        x, y, w, h = self.rect
        base = (70, 70, 70)
        border = (195, 195, 195) if focused else ((170, 170, 170) if hovered else (120, 120, 120))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), base, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

        TR.put(canvas, self.label, (x + 8, y + int(h * 0.42)),
               FONT_FACE, fonts.small, (220, 220, 220), THICK_INFO, height_px=int(14 * fonts.small))

        val = self._edit_text if self._editing else (self.fmt.format(self.getter()) + (self.unit or ""))
        TR.put(canvas, val, (x + 8, y + int(h * 0.85)),
               FONT_FACE, fonts.ui, (245, 245, 245), THICK_UI, height_px=int(18 * fonts.ui))

        ax_w = 34
        ax_x0 = x + w - ax_w
        cv2.rectangle(canvas, (ax_x0, y), (x + w, y + h), (60, 60, 60), -1)
        cv2.rectangle(canvas, (ax_x0, y), (x + w, y + h), (110, 110, 110), 1)

        up = np.array([[ax_x0 + ax_w // 2, y + 8], [ax_x0 + 8, y + h // 2 - 2], [ax_x0 + ax_w - 8, y + h // 2 - 2]], dtype=np.int32)
        dn = np.array([[ax_x0 + 8, y + h // 2 + 2], [ax_x0 + ax_w - 8, y + h // 2 + 2], [ax_x0 + ax_w // 2, y + h - 8]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, up, (200, 200, 200))
        cv2.fillConvexPoly(canvas, dn, (200, 200, 200))
        cv2.line(canvas, (ax_x0, y + h // 2), (x + w, y + h // 2), (110, 110, 110), 1)

    def _set_clamped(self, v: float):
        v = float(v)
        v = clamp(v, self.vmin, self.vmax)
        if "{:.0f}" in self.fmt:
            v = int(round(v))
        self.setter(v)

    def inc(self, delta: float):
        cur = float(self.getter())
        self._set_clamped(cur + delta)

    def on_click(self, x: int, y: int, flags: int):
        self._editing = True
        self._edit_text = str(self.fmt.format(self.getter()))

    def cancel_edit(self):
        self._editing = False
        self._edit_text = ""

    def commit_edit(self):
        if not self._editing:
            return
        s = self._edit_text.strip()
        if s in ("", "-", "."):
            self.cancel_edit()
            return
        try:
            v = float(s)
        except Exception:
            self.cancel_edit()
            return
        self._set_clamped(v)
        self.cancel_edit()

    def on_key(self, key: int):
        if not self._editing:
            return
        if key in (13, 10):
            self.commit_edit()
            return
        if key == 27:
            self.cancel_edit()
            return
        if key in (8, 127):
            self._edit_text = self._edit_text[:-1]
            return
        ch = chr(key) if 32 <= key <= 126 else ""
        if ch in "0123456789.-":
            if ch == "." and "." in self._edit_text:
                return
            if ch == "-" and len(self._edit_text) > 0:
                return
            self._edit_text += ch


@dataclass
class EnumControl(UIControl):
    """
    Control tipo spinner para opciones discretas (p.ej. Bayer pattern).
    Flechas: ciclo; click en valor: ciclo.
    """
    key: str
    label: str
    tooltip: str
    rect: Tuple[int, int, int, int]
    options: List[str]
    getter_idx: Callable[[], int]
    setter_idx: Callable[[int], None]

    def arrow_dir(self, x: int, y: int) -> Optional[int]:
        rx, ry, rw, rh = self.rect
        ax_w = 34
        ax_x0 = rx + rw - ax_w
        if x < ax_x0:
            return None
        return +1 if y < ry + rh // 2 else -1

    def _set_idx(self, i: int):
        n = len(self.options)
        if n <= 0:
            return
        i = i % n
        self.setter_idx(i)

    def inc(self, d: int):
        self._set_idx(self.getter_idx() + d)

    def draw(self, canvas, fonts, hovered, focused):
        x, y, w, h = self.rect
        base = (70, 70, 70)
        border = (170, 170, 170) if hovered else (120, 120, 120)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), base, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

        TR.put(canvas, self.label, (x + 8, y + int(h * 0.42)),
               FONT_FACE, fonts.small, (220, 220, 220), THICK_INFO, height_px=int(14 * fonts.small))

        idx = int(self.getter_idx())
        val = self.options[idx % len(self.options)] if self.options else "-"
        TR.put(canvas, val, (x + 8, y + int(h * 0.85)),
               FONT_FACE, fonts.ui, (245, 245, 245), THICK_UI, height_px=int(18 * fonts.ui))

        ax_w = 34
        ax_x0 = x + w - ax_w
        cv2.rectangle(canvas, (ax_x0, y), (x + w, y + h), (60, 60, 60), -1)
        cv2.rectangle(canvas, (ax_x0, y), (x + w, y + h), (110, 110, 110), 1)

        up = np.array([[ax_x0 + ax_w // 2, y + 8], [ax_x0 + 8, y + h // 2 - 2], [ax_x0 + ax_w - 8, y + h // 2 - 2]], dtype=np.int32)
        dn = np.array([[ax_x0 + 8, y + h // 2 + 2], [ax_x0 + ax_w - 8, y + h // 2 + 2], [ax_x0 + ax_w // 2, y + h - 8]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, up, (200, 200, 200))
        cv2.fillConvexPoly(canvas, dn, (200, 200, 200))
        cv2.line(canvas, (ax_x0, y + h // 2), (x + w, y + h // 2), (110, 110, 110), 1)

    def on_click(self, x: int, y: int, flags: int):
        # click on value area cycles forward
        self.inc(+1)


# =========================
# UI manager
# =========================
class SingleWindowUI:
    def __init__(self, win_name: str, canvas_w: int, canvas_h: int):
        self.win = win_name
        self.W = int(canvas_w)
        self.H = int(canvas_h)

        self.buttons: List[Button] = []
        self.controls: List[UIControl] = []

        self.mouse = {"x": 0, "y": 0, "flags": 0}

        self.hover_key: Optional[str] = None
        self.hover_since: float = 0.0
        self.dwell_s: float = 0.40

        self.focus_key: Optional[str] = None

        # press-hold repeat
        self.press = {
            "active": False,
            "key": None,
            "dir": 0,
            "shift": False,
            "t0": 0.0,
            "t_last": 0.0,
        }
        self.repeat_delay = 0.28
        self.repeat_interval = 0.055
        self.accel_after = 0.80

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.W, self.H)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def add_button(self, b: Button):
        self.buttons.append(b)

    def add_control(self, c: UIControl):
        self.controls.append(c)

    def _get_control(self, key: str) -> Optional[UIControl]:
        for c in self.controls:
            if c.key == key:
                return c
        return None

    def _get_button(self, key: str) -> Optional[Button]:
        for b in self.buttons:
            if b.key == key:
                return b
        return None

    def _set_focus(self, key: Optional[str]):
        if self.focus_key is not None and self.focus_key != key:
            prev = self._get_control(self.focus_key)
            if isinstance(prev, SpinnerControl):
                prev.commit_edit()
        self.focus_key = key

    def _hit_any(self, x: int, y: int) -> Tuple[Optional[str], str]:
        for c in self.controls:
            if c.hit(x, y):
                return c.key, "control"
        for b in self.buttons:
            bx, by, bw, bh = b.rect
            if (bx <= x <= bx + bw) and (by <= y <= by + bh):
                return b.key, "button"
        return None, ""

    def _spinner_increment(self, c: Union[SpinnerControl, EnumControl], direction: int, shift: bool, held_s: float):
        if isinstance(c, EnumControl):
            c.inc(direction)
            return
        step = c.step_fast if (shift or held_s >= self.accel_after) else c.step
        c.inc(direction * step)

    def _on_mouse(self, event, x, y, flags, userdata=None):
        self.mouse["x"], self.mouse["y"], self.mouse["flags"] = x, y, flags

        if event == cv2.EVENT_MOUSEMOVE:
            key, _ = self._hit_any(x, y)
            if key != self.hover_key:
                self.hover_key = key
                self.hover_since = time.time()
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.press["active"] = False
            self.press["key"] = None
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        key, kind = self._hit_any(x, y)
        if kind == "control" and key is not None:
            c = self._get_control(key)
            self._set_focus(key)

            if isinstance(c, (SpinnerControl, EnumControl)):
                d = c.arrow_dir(x, y)
                if d is not None:
                    shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
                    # stop editing if spinner
                    if isinstance(c, SpinnerControl):
                        c._editing = False
                        c._edit_text = ""
                    self._spinner_increment(c, d, shift, held_s=0.0)
                    self.press.update({"active": True, "key": key, "dir": d, "shift": shift, "t0": time.time(), "t_last": time.time()})
                    return

                # click in value area
                c.on_click(x, y, flags)
                return

            if c is not None:
                c.on_click(x, y, flags)
            return

        if kind == "button" and key is not None:
            self._set_focus(None)
            b = self._get_button(key)
            if b is not None:
                b.on_click()
            return

        self._set_focus(None)

    def tick_repeat(self):
        if not self.press["active"] or self.press["key"] is None:
            return
        c = self._get_control(self.press["key"])
        if not isinstance(c, (SpinnerControl, EnumControl)):
            self.press["active"] = False
            self.press["key"] = None
            return

        now = time.time()
        held = now - self.press["t0"]
        if held < self.repeat_delay:
            return
        if now - self.press["t_last"] < self.repeat_interval:
            return
        self.press["t_last"] = now
        self._spinner_increment(c, int(self.press["dir"]), bool(self.press["shift"]), held_s=held)

    @staticmethod
    def _rounded_rect(img, rect, color, radius=10, thickness=-1):
        x, y, w, h = rect
        r = min(radius, w // 2, h // 2)
        if thickness < 0:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
            cv2.circle(img, (x + r, y + r), r, color, -1)
            cv2.circle(img, (x + w - r, y + r), r, color, -1)
            cv2.circle(img, (x + r, y + h - r), r, color, -1)
            cv2.circle(img, (x + w - r, y + h - r), r, color, -1)
        else:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
            cv2.circle(img, (x + r, y + r), r, color, thickness)
            cv2.circle(img, (x + w - r, y + r), r, color, thickness)
            cv2.circle(img, (x + r, y + h - r), r, color, thickness)
            cv2.circle(img, (x + w - r, y + h - r), r, color, thickness)

    def _draw_tooltip(self, canvas: np.ndarray, fonts: UIFonts, text: str, x: int, y: int):
        if text.strip() == "":
            return
        lines = text.split(" | ")
        maxw = 0
        for ln in lines:
            (tw, _), _ = TR.get_text_size(ln, FONT_FACE, fonts.small, THICK_INFO, height_px=int(14 * fonts.small))
            maxw = max(maxw, tw)

        pad = 10
        lh = int(fonts.line_h * 0.90)
        box_w = maxw + 2 * pad
        box_h = len(lines) * lh + 2 * pad

        bx = int(clamp(x + 18, 8, self.W - box_w - 8))
        by = int(clamp(y + 18, 8, self.H - box_h - 8))

        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (200, 200, 200), 1)

        ty = by + pad + lh - 6
        for ln in lines:
            TR.put(canvas, ln, (bx + pad, ty), FONT_FACE, fonts.small, (245, 245, 245), THICK_INFO, height_px=int(14 * fonts.small))
            ty += lh

    def render(self, live_bgr: np.ndarray, stack_disp_u8: Optional[np.ndarray],
               info_lines: List[str], status_right: str, footer_right: str, state: str):
        pad = 12
        fonts = compute_ui_fonts(self.W)

        # ---- Dynamic panel height (fix overflow) ----
        cols = 3
        ctrl_h = 50
        ctrl_gap_y = 10
        ctrl_gap_x = 10
        n_controls = len(self.controls)
        nrows = int(np.ceil(n_controls / cols)) if n_controls > 0 else 0

        top_row_h = 60          # state pill + buttons row block
        bottom_pad = 16
        panel_h = (top_row_h + (nrows * ctrl_h) + max(0, nrows - 1) * ctrl_gap_y + bottom_pad)
        panel_h = int(max(220, panel_h))  # minimum

        # Views
        view_h = self.H - panel_h - 2 * pad
        view_h = int(max(240, view_h))
        view_w = (self.W - 3 * pad) // 2

        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        lx, ly = pad, pad
        rx, ry = lx + view_w + pad, pad

        live_view = cv2.resize(live_bgr, (view_w, view_h), interpolation=cv2.INTER_AREA)
        if stack_disp_u8 is None:
            stack_view = np.zeros((view_h, view_w, 3), dtype=np.uint8)
        else:
            st = cv2.resize(stack_disp_u8, (view_w, view_h), interpolation=cv2.INTER_AREA)
            stack_view = cv2.cvtColor(st, cv2.COLOR_GRAY2BGR) if st.ndim == 2 else st

        canvas[ly:ly + view_h, lx:lx + view_w] = live_view
        canvas[ry:ry + view_h, rx:rx + view_w] = stack_view

        cv2.rectangle(canvas, (lx, ly), (lx + view_w, ly + view_h), (70, 70, 70), 1)
        cv2.rectangle(canvas, (rx, ry), (rx + view_w, ry + view_h), (70, 70, 70), 1)

        TR.put(canvas, "LIVE", (lx + 10, ly + int(28 * fonts.title)),
               FONT_FACE, fonts.title, (235, 235, 235), THICK_TITLE, height_px=int(22 * fonts.title))
        TR.put(canvas, "STACK", (rx + 10, ry + int(28 * fonts.title)),
               FONT_FACE, fonts.title, (235, 235, 235), THICK_TITLE, height_px=int(22 * fonts.title))

        (tw, _), _ = TR.get_text_size(status_right, FONT_FACE, fonts.small, THICK_INFO, height_px=int(14 * fonts.small))
        TR.put(canvas, status_right, (self.W - pad - tw - 10, ly + view_h - 10),
               FONT_FACE, fonts.small, (235, 235, 235), THICK_INFO, height_px=int(14 * fonts.small))

        # Bottom panel
        py = pad + view_h + pad
        cv2.rectangle(canvas, (pad, py), (self.W - pad, self.H - pad), (25, 25, 25), -1)
        cv2.rectangle(canvas, (pad, py), (self.W - pad, self.H - pad), (70, 70, 70), 1)

        # State pill
        pill_w, pill_h = 118, 34
        pill_x, pill_y = pad + 14, py + 12
        col = (45, 130, 45) if state == "LOCKED" else (40, 95, 165)
        txt = "LOCKED" if state == "LOCKED" else "LOST"
        self._rounded_rect(canvas, (pill_x, pill_y, pill_w, pill_h), col, radius=12, thickness=-1)
        self._rounded_rect(canvas, (pill_x, pill_y, pill_w, pill_h), (150, 150, 150), radius=12, thickness=1)
        (tw, th), _ = TR.get_text_size(txt, FONT_FACE, fonts.ui, THICK_UI, height_px=int(18 * fonts.ui))
        TR.put(canvas, txt, (pill_x + (pill_w - tw) // 2, pill_y + (pill_h + th) // 2 - 3),
               FONT_FACE, fonts.ui, (245, 245, 245), THICK_UI, height_px=int(18 * fonts.ui))

        panel_x0 = pad + 14
        panel_x1 = self.W - pad - 14
        panel_w = panel_x1 - panel_x0
        left_w = int(panel_w * 0.68)
        right_x0 = panel_x0 + left_w + 18

        # Buttons row
        btn_y = py + 12
        btn_x0 = pill_x + pill_w + 14
        btn_area_w = left_w - (btn_x0 - panel_x0)
        gap = 10
        bw = max(92, (btn_area_w - gap * (len(self.buttons) - 1)) // max(1, len(self.buttons)))
        bh = 34

        for i, b in enumerate(self.buttons):
            rect = (btn_x0 + i * (bw + gap), btn_y, bw, bh)
            b.rect = rect
            hovered = (self.hover_key == b.key)
            toggled = bool(b.toggled())
            base = (75, 75, 75) if not toggled else (95, 95, 95)
            border = (190, 190, 190) if hovered else (130, 130, 130)
            self._rounded_rect(canvas, rect, base, radius=10, thickness=-1)
            self._rounded_rect(canvas, rect, border, radius=10, thickness=1)
            (tw, _), _ = TR.get_text_size(b.label, FONT_FACE, fonts.ui, THICK_UI, height_px=int(18 * fonts.ui))
            TR.put(canvas, b.label, (rect[0] + (bw - tw) // 2, rect[1] + int(bh * 0.70)),
                   FONT_FACE, fonts.ui, (245, 245, 245), THICK_UI, height_px=int(18 * fonts.ui))

        # Controls grid
        ctrl_top = py + 60
        col_w = (left_w - (cols - 1) * ctrl_gap_x) // cols

        for idx, c in enumerate(self.controls):
            r = idx // cols
            cc = idx % cols
            x = panel_x0 + cc * (col_w + ctrl_gap_x)
            y = ctrl_top + r * (ctrl_h + ctrl_gap_y)
            c.rect = (x, y, col_w, ctrl_h)
            hovered = (self.hover_key == c.key)
            focused = (self.focus_key == c.key)
            c.draw(canvas, fonts, hovered=hovered, focused=focused)

        # Info right
        info_y0 = py + 16
        for i, line in enumerate(info_lines[:5]):
            y = info_y0 + i * fonts.line_h
            TR.put(canvas, line, (right_x0, y), FONT_FACE, fonts.info, (230, 230, 230), THICK_INFO, height_px=int(16 * fonts.info))

        # Footer right
        (tw, _), _ = TR.get_text_size(footer_right, FONT_FACE, fonts.small, THICK_INFO, height_px=int(14 * fonts.small))
        TR.put(canvas, footer_right, (self.W - pad - tw - 10, self.H - pad - 18),
               FONT_FACE, fonts.small, (230, 230, 230), THICK_INFO, height_px=int(14 * fonts.small))

        # Tooltip
        if self.hover_key is not None and (time.time() - self.hover_since) >= self.dwell_s:
            tip = ""
            c = self._get_control(self.hover_key)
            if c is not None:
                tip = getattr(c, "tooltip", "")
            else:
                b = self._get_button(self.hover_key)
                if b is not None:
                    tip = getattr(b, "tooltip", "")
            if tip:
                self._draw_tooltip(canvas, fonts, tip, self.mouse["x"], self.mouse["y"])

        cv2.imshow(self.win, canvas)

    def poll_key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    def route_key_to_focus(self, key: int):
        if self.focus_key is None:
            return
        c = self._get_control(self.focus_key)
        if c is None:
            return
        if isinstance(c, SpinnerControl):
            if not c._editing:
                if key in (ord("+"), ord("=")):
                    c.inc(c.step)
                    return
                if key in (ord("-"), ord("_")):
                    c.inc(-c.step)
                    return
            c.on_key(key)


# =========================
# Main params
# =========================
@dataclass
class Params:
    exp_ms: int = 200
    gain: int = 200
    gain_auto: bool = False

    live_stretch: bool = True
    auto_contrast: bool = True
    gamma: float = 1.40

    min_sharpness: float = 2.0
    resp_lock: float = 0.020
    resp_reacq: float = 0.030
    max_shift_pct: int = 35
    ref_update_every: int = 50
    min_stars_proxy: int = 2
    weight: bool = True

    # Debayer
    debayer: bool = True
    debayer_hq: bool = False
    bayer_idx: int = 0  # 0..3 over ["RGGB","BGGR","GBRG","GRBG"]


# =========================
# Main
# =========================
def main(outdir: str = "captures", roi: int = 1024, binning: int = 1):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cam_id = None
    props = None
    model = "Sin camara"
    is_color = False
    started = False
    iw = 0
    ih = 0
    fmt = None
    buf = None

    cfg = StackConfig(roi=roi)
    stacker = LiveStacker(cfg)
    p = Params()

    running = {"quit": False}
    saved = {"stack": 0, "live": 0}
    last_live_holder = {"bgr": np.zeros((roi, roi, 3), dtype=np.uint8)}

    # Window size (safe default)
    pad = 12
    view_w, view_h = 640, 640
    canvas_w = pad * 3 + view_w * 2
    canvas_h = pad * 2 + view_h + pad + 420  # generous; panel is dynamic anyway

    ui = SingleWindowUI("Live Stacking (Mars-C)", canvas_w, canvas_h)

    # Camera control
    def teardown_camera(reason: str | None = None):
        nonlocal cam_id, props, model, is_color, started, iw, ih, fmt, buf
        if cam_id is None:
            return
        if started:
            try:
                ensure_ok(pyPOACamera.StopExposure(cam_id), "StopExposure")
            except Exception:
                pass
        try:
            err = pyPOACamera.CloseCamera(cam_id)
            if err != pyPOACamera.POAErrors.POA_OK:
                print("Warning: CloseCamera returned error:", pyPOACamera.GetErrorString(err))
        except Exception:
            pass
        if reason:
            print(reason)
        cam_id = None
        props = None
        model = "Sin camara"
        is_color = False
        started = False
        iw = 0
        ih = 0
        fmt = None
        buf = None

    def connect_camera():
        nonlocal cam_id, props, model, is_color, started, iw, ih, fmt, buf
        if cam_id is not None:
            return
        cam_id, props = pick_first_camera()
        model = props.cameraModelName.decode(errors="ignore").strip("\x00") if isinstance(props.cameraModelName, (bytes, bytearray)) else str(props.cameraModelName)
        is_color = bool(props.isColorCamera)
        ensure_ok(pyPOACamera.OpenCamera(cam_id), "OpenCamera")
        try:
            ensure_ok(pyPOACamera.InitCamera(cam_id), "InitCamera")
            ensure_ok(pyPOACamera.SetImageBin(cam_id, int(binning)), "SetImageBin")
            set_centered_roi(cam_id, props, roi, roi)

            fmt = pyPOACamera.POAImgFormat.POA_RAW8
            ensure_ok(pyPOACamera.SetImageFormat(cam_id, fmt), "SetImageFormat(RAW8)")

            err, iw, ih = pyPOACamera.GetImageSize(cam_id)
            ensure_ok(err, "GetImageSize")
            err, fmt2 = pyPOACamera.GetImageFormat(cam_id)
            ensure_ok(err, "GetImageFormat")
            fmt = fmt2

            buf_size = pyPOACamera.ImageCalcSize(ih, iw, fmt)
            buf = np.zeros(buf_size, dtype=np.uint8)

            apply_camera_settings(force=True)
            ensure_ok(pyPOACamera.StartExposure(cam_id, False), "StartExposure(Video)")
            started = True
            print(f"Camera connected: {model}")
        except Exception as e:
            teardown_camera(f"Warning: camera init failed: {e!r}")
            raise

    # Buttons
    def act_reset(): stacker.reset()
    def act_pause(): stacker.paused = not stacker.paused
    def act_save_stack():
        st = stacker.get_stack_u8(raw=True)
        if st is None:
            return
        saved["stack"] += 1
        fname = outdir / f"stack_{saved['stack']:06d}.png"
        cv2.imwrite(str(fname), st)
        print("Saved", fname)

    def act_save_live():
        saved["live"] += 1
        fname = outdir / f"live_{saved['live']:06d}.png"
        cv2.imwrite(str(fname), last_live_holder["bgr"])
        print("Saved", fname)

    def act_camera():
        if cam_id is None:
            try:
                connect_camera()
                stacker.reset()
            except Exception as e:
                print("Warning: connect failed:", repr(e))
        else:
            teardown_camera("Camera disconnected by user.")

    def act_quit(): running["quit"] = True

    ui.add_button(Button("camera", "Camera", (0, 0, 1, 1), act_camera, tooltip="Conectar/desconectar camara.", toggled=lambda: cam_id is not None))
    ui.add_button(Button("reset", "Reset", (0, 0, 1, 1), act_reset, tooltip="Reinicia referencia y acumulador."))
    ui.add_button(Button("pause", "Pause", (0, 0, 1, 1), act_pause, tooltip="Pausa/continua el acumulado.", toggled=lambda: stacker.paused))
    ui.add_button(Button("savestack", "SaveStack", (0, 0, 1, 1), act_save_stack, tooltip="Guarda el stack actual (PNG 8-bit)."))
    ui.add_button(Button("savelive", "SaveLive", (0, 0, 1, 1), act_save_live, tooltip="Guarda la vista LIVE actual (PNG)."))
    ui.add_button(Button("quit", "Quit", (0, 0, 1, 1), act_quit, tooltip="Salir."))

    # Controls
    ui.add_control(SpinnerControl(
        key="exp", label="Exposure (ms)",
        tooltip="Tiempo de exposicion. | Mas alto = mas senal, menos FPS. | Shift acelera y aumenta el paso.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.exp_ms, setter=lambda v: setattr(p, "exp_ms", int(v)),
        vmin=1, vmax=2000, step=1, step_fast=10, fmt="{:.0f}", unit="ms"
    ))
    ui.add_control(SpinnerControl(
        key="gain", label="Gain",
        tooltip="Ganancia del sensor. | Mas alto = mas senal y mas ruido. | Shift acelera y aumenta el paso.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.gain, setter=lambda v: setattr(p, "gain", int(v)),
        vmin=0, vmax=500, step=1, step_fast=10, fmt="{:.0f}"
    ))
    ui.add_control(ToggleControl(
        key="gainauto", label="Gain Auto",
        tooltip="Ganancia automatica (si el driver lo soporta bien). | Si ves inestabilidad, OFF.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.gain_auto, setter=lambda b: setattr(p, "gain_auto", bool(b)),
    ))

    # Debayer controls (still shown even if mono; they just won't affect capture)
    bayer_opts = ["RGGB", "BGGR", "GBRG", "GRBG"]
    ui.add_control(ToggleControl(
        key="debayer", label="Debayer",
        tooltip="Demosaicing para camaras color (Bayer). | ON: muestra color real. | OFF: preview/stack en gris.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.debayer, setter=lambda b: setattr(p, "debayer", bool(b)),
    ))
    ui.add_control(EnumControl(
        key="bayer", label="Bayer",
        tooltip="Patron Bayer del sensor. | Si el color se ve raro, cambia esta opcion.",
        rect=(0, 0, 1, 1),
        options=bayer_opts,
        getter_idx=lambda: p.bayer_idx,
        setter_idx=lambda i: setattr(p, "bayer_idx", int(i)),
    ))
    ui.add_control(ToggleControl(
        key="debayerhq", label="Debayer HQ",
        tooltip="Debayer de mayor calidad (VNG) si OpenCV lo soporta. | Mejor color, mas lento.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.debayer_hq, setter=lambda b: setattr(p, "debayer_hq", bool(b)),
    ))

    ui.add_control(SpinnerControl(
        key="minsharp", label="Min Sharp",
        tooltip="Umbral de nitidez (varianza Laplaciano). | Filtra frames borrosos. | Baja si no acumula.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.min_sharpness, setter=lambda v: setattr(p, "min_sharpness", float(v)),
        vmin=0.0, vmax=50.0, step=0.1, step_fast=1.0, fmt="{:.1f}"
    ))
    ui.add_control(SpinnerControl(
        key="resplock", label="Resp Lock",
        tooltip="Umbral de correlacion para permanecer LOCKED. | Mas alto = mas estricto. | Baja si resp es baja.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.resp_lock, setter=lambda v: setattr(p, "resp_lock", float(v)),
        vmin=0.0, vmax=0.20, step=0.001, step_fast=0.010, fmt="{:.3f}"
    ))
    ui.add_control(SpinnerControl(
        key="respreacq", label="Resp Reacq",
        tooltip="Umbral de correlacion para salir de LOST. | Normalmente >= RespLock.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.resp_reacq, setter=lambda v: setattr(p, "resp_reacq", float(v)),
        vmin=0.0, vmax=0.25, step=0.001, step_fast=0.010, fmt="{:.3f}"
    ))
    ui.add_control(SpinnerControl(
        key="maxshift", label="Max Shift (%)",
        tooltip="Maximo shift antes de LOST. | Sube si deriva mucho. | Baja si hay aligns falsos.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.max_shift_pct, setter=lambda v: setattr(p, "max_shift_pct", int(v)),
        vmin=5, vmax=60, step=1, step_fast=5, fmt="{:.0f}", unit="%"
    ))
    ui.add_control(SpinnerControl(
        key="refupd", label="Ref Update",
        tooltip="Cada N frames aceptados, actualiza referencia con el stack. | Ayuda con deriva lenta.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.ref_update_every, setter=lambda v: setattr(p, "ref_update_every", int(v)),
        vmin=5, vmax=300, step=1, step_fast=10, fmt="{:.0f}"
    ))
    ui.add_control(SpinnerControl(
        key="minstars", label="Min Stars",
        tooltip="Proxy minimo de estrellas para LOCKED/reacquire. | Baja si el campo tiene pocas estrellas.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.min_stars_proxy, setter=lambda v: setattr(p, "min_stars_proxy", int(v)),
        vmin=0, vmax=30, step=1, step_fast=5, fmt="{:.0f}"
    ))
    ui.add_control(ToggleControl(
        key="weight", label="Quality Weight",
        tooltip="Pesa frames por (resp * sharp). | Puede mejorar stack si hay frames muy malos.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.weight, setter=lambda b: setattr(p, "weight", bool(b)),
    ))
    ui.add_control(ToggleControl(
        key="autoc", label="Auto Contrast",
        tooltip="Stretch por percentiles (solo display). | ON ayuda a ver senal debil sin afectar el stack.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.auto_contrast, setter=lambda b: setattr(p, "auto_contrast", bool(b)),
    ))
    ui.add_control(ToggleControl(
        key="livestretch", label="Live Stretch",
        tooltip="Stretch/gamma en LIVE (solo display). | Util para enfoque/encuadre en campos tenues.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.live_stretch, setter=lambda b: setattr(p, "live_stretch", bool(b)),
    ))
    ui.add_control(SpinnerControl(
        key="gamma", label="Gamma",
        tooltip="Gamma de visualizacion (solo display). | >1 levanta sombras; <1 oscurece. | No altera el stack.",
        rect=(0, 0, 1, 1),
        getter=lambda: p.gamma, setter=lambda v: setattr(p, "gamma", float(v)),
        vmin=0.5, vmax=3.0, step=0.05, step_fast=0.20, fmt="{:.2f}"
    ))

    # Debounce apply
    last_apply_t = 0.0
    last_applied = {"exp_ms": None, "gain": None, "gain_auto": None}

    def apply_camera_settings(force: bool = False):
        nonlocal last_apply_t
        if cam_id is None:
            return
        now = time.time()
        if not force and (now - last_apply_t) < 0.20:
            return

        exp_ms = int(clamp(p.exp_ms, 1, 2000))
        gain = int(clamp(p.gain, 0, 500))
        gain_auto = bool(p.gain_auto)

        changed = force or (
            exp_ms != last_applied["exp_ms"] or
            gain != last_applied["gain"] or
            gain_auto != last_applied["gain_auto"]
        )
        if not changed:
            return

        try:
            exp_us = int(exp_ms * 1000)
            ensure_ok(pyPOACamera.SetExp(cam_id, exp_us, False), "SetExp")
            ensure_ok(pyPOACamera.SetGain(cam_id, int(gain), bool(gain_auto)), "SetGain")
            last_applied["exp_ms"] = exp_ms
            last_applied["gain"] = gain
            last_applied["gain_auto"] = gain_auto
            last_apply_t = now
        except Exception as e:
            print("Warning: apply_camera_settings failed:", repr(e))

    try:
        t0 = time.time()
        last_report = time.time()
        fps_in = 0.0
        fps_acc = 0.0

        if TR.use_freetype:
            print(f"[UI] Unicode ON (FreeType). Font: {TR.ttf_path}")
        else:
            print("[UI] Unicode OFF (Hershey). Para tildes: pip install opencv-contrib-python")

        print("Modo rapido spinners: mantener presionado ▲/▼. Shift acelera (step_fast).")
        print("Editar valores: click en el numero, escribe, Enter aplica, Esc cancela.")
        print("Teclas: q=quit, r=reset, p=pause, s=save stack, l=save live")

        while not running["quit"]:
            ui.tick_repeat()

            # mirror params -> cfg
            cfg.min_sharpness = float(p.min_sharpness)
            cfg.min_response_lock = float(p.resp_lock)
            cfg.min_response_reacq = float(p.resp_reacq)
            cfg.max_shift_frac = float(p.max_shift_pct) / 100.0
            cfg.update_ref_every_accepted = int(p.ref_update_every)
            cfg.min_star_proxy = int(p.min_stars_proxy)
            cfg.use_quality_weight = bool(p.weight)

            apply_camera_settings(force=False)

            if cam_id is None:
                live_disp = np.zeros((view_h, view_w, 3), dtype=np.uint8)
                st_disp = stacker.get_stack_u8(raw=True)
                if st_disp is not None:
                    st_disp = percentile_stretch_u8(st_disp, 1.0, 99.5) if p.auto_contrast else st_disp
                    st_disp = apply_gamma_u8(st_disp, float(p.gamma))
                ui.render(live_disp, st_disp, stacker.overlay_lines(), "Sin camara conectada", "FPS in: 0.0   FPS acc: 0.0", "IDLE")
            else:
                try:
                    err, ready = pyPOACamera.ImageReady(cam_id)
                    ensure_ok(err, "ImageReady")
                except Exception as e:
                    teardown_camera(f"Warning: camera disconnected: {e!r}")
                    continue

                if not ready:
                    time.sleep(0.001)
                else:
                    timeout_ms = int(p.exp_ms) + 2500
                    try:
                        ensure_ok(pyPOACamera.GetImageData(cam_id, buf, timeout_ms), "GetImageData")
                        img = pyPOACamera.ImageDataConvert(buf, ih, iw, fmt)  # RAW8 -> (H,W) u8 typically
                    except Exception as e:
                        teardown_camera(f"Warning: frame grab failed: {e!r}")
                        continue

                    # Build gray + live_bgr using debayer pipeline if color camera
                    if is_color:
                        pattern = bayer_opts[int(p.bayer_idx) % len(bayer_opts)]
                        gray_u8, live_bgr = demosaic_raw8(
                            raw=img,
                            pattern=pattern,
                            want_color=bool(p.debayer),
                            hq=bool(p.debayer_hq),
                        )
                    else:
                        gray_u8 = ensure_gray_u8(img)
                        live_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

                    last_live_holder["bgr"] = live_bgr

                    try:
                        stacker.process(gray_u8)
                    except Exception as e:
                        print("Warning: stacker.process failed:", repr(e))
                        stacker.state = "LOST"

                    gamma = float(p.gamma)

                    # LIVE display
                    if p.live_stretch:
                        lg = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY)
                        lg = percentile_stretch_u8(lg, 2.0, 99.8)
                        lg = apply_gamma_u8(lg, gamma)
                        live_disp = cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR)
                    else:
                        live_disp = live_bgr.copy()

                    # STACK display
                    st_raw = stacker.get_stack_u8(raw=True)
                    if st_raw is None:
                        st_disp = None
                    else:
                        st_disp = percentile_stretch_u8(st_raw, 1.0, 99.5) if p.auto_contrast else st_raw
                        st_disp = apply_gamma_u8(st_disp, gamma)

                    now = time.time()
                    if now - last_report > 1.0:
                        dt = now - t0
                        fps_in = stacker.seen / dt if dt > 0 else 0.0
                        fps_acc = stacker.accepted / dt if dt > 0 else 0.0
                        last_report = now

                    status_right = (
                        f"{model}   ROI={iw}x{ih}   Exp={p.exp_ms}ms   Gain={p.gain}"
                        + ("(A)" if p.gain_auto else "")
                        + f"   Gamma={gamma:.2f}"
                        + (f"   Debayer={p.debayer}({bayer_opts[p.bayer_idx % 4]})" if is_color else "")
                    )
                    footer_right = f"FPS in: {fps_in:.1f}   FPS acc: {fps_acc:.1f}"

                    info = stacker.overlay_lines()
                    info.append(f"Gates: sharp>={cfg.min_sharpness:.1f}  lock>={cfg.min_response_lock:.3f}  reacq>={cfg.min_response_reacq:.3f}")
                    info.append(f"MaxShift={cfg.max_shift_frac:.2f}  RefUpd={cfg.update_ref_every_accepted}  StarsMin={cfg.min_star_proxy}  Weight={cfg.use_quality_weight}")

                    ui.render(live_disp, st_disp, info[:5], status_right, footer_right, stacker.state)

            key = ui.poll_key()
            if key != 255:
                ui.route_key_to_focus(key)

            if key == ord("q"):
                running["quit"] = True
            elif key == ord("r"):
                act_reset()
            elif key == ord("p"):
                act_pause()
            elif key == ord("s"):
                act_save_stack()
            elif key == ord("l"):
                act_save_live()

        if cam_id is not None:
            try:
                err, dropped = pyPOACamera.GetDroppedImagesCount(cam_id)
                if err == pyPOACamera.POAErrors.POA_OK:
                    print(f"Dropped frames (SDK): {dropped}")
            except Exception:
                pass

        cv2.destroyAllWindows()

    finally:
        teardown_camera("Camera closed.")


if __name__ == "__main__":
    main(outdir="captures", roi=1024, binning=1)
