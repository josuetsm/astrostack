from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .preprocessing import preprocess_for_phasecorr, phasecorr_delta, pyramid_phasecorr_delta


@dataclass
class TrackingConfig:
    subtract_bg_ema: bool = True
    bg_ema_alpha: float = 0.03
    sigma_hp: float = 10.0
    sigma_smooth: float = 2.0
    bright_percentile: float = 99.3
    resp_min: float = 0.06
    max_shift_per_frame_px: float = 25.0


@dataclass
class TrackingState:
    bg_ema: Optional[np.ndarray] = None
    prev_reg: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    x_hat: float = 0.0
    y_hat: float = 0.0
    last_resp: float = 0.0
    fail_count: int = 0


class TrackingEngine:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.state = TrackingState()

    def reset(self) -> None:
        self.state = TrackingState()

    def preprocess(self, frame_u16: np.ndarray, update_bg: bool = True) -> np.ndarray:
        reg, bg = preprocess_for_phasecorr(
            frame_u16,
            self.state.bg_ema,
            sigma_hp=self.config.sigma_hp,
            sigma_smooth=self.config.sigma_smooth,
            bright_percentile=self.config.bright_percentile,
            bg_ema_alpha=self.config.bg_ema_alpha,
            subtract_bg_ema=self.config.subtract_bg_ema,
            update_bg=update_bg,
        )
        self.state.bg_ema = bg
        return reg

    def update(self, reg: np.ndarray, timestamp: float) -> Tuple[float, float, float]:
        state = self.state
        if state.prev_reg is None:
            state.prev_reg = reg
            state.prev_t = timestamp
            return 0.0, 0.0, 0.0

        dt = timestamp - (state.prev_t or timestamp)
        if dt <= 1e-6:
            return 0.0, 0.0, 0.0

        dx, dy, resp = phasecorr_delta(state.prev_reg, reg)
        mag = float(np.hypot(dx, dy))
        good = (
            resp >= self.config.resp_min
            and mag <= self.config.max_shift_per_frame_px
            and np.isfinite(mag)
        )
        state.last_resp = float(resp)

        if good:
            state.fail_count = 0
            state.x_hat += dx
            state.y_hat += dy
        else:
            state.fail_count += 1

        state.prev_reg = reg
        state.prev_t = timestamp
        return float(dx), float(dy), float(resp)

    def absolute_correction(self, ref_reg: np.ndarray, cur_reg: np.ndarray, levels: int = 3) -> Tuple[float, float, float]:
        return pyramid_phasecorr_delta(ref_reg, cur_reg, levels)
