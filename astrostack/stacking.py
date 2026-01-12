from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .preprocessing import (
    local_zscore_u16,
    make_sparse_reg,
    pyramid_phasecorr_delta,
    remove_hot_pixels,
    warp_translate,
)


@dataclass
class StackingConfig:
    sigma_bg: float = 35.0
    sigma_floor_p: float = 10.0
    z_clip: float = 6.0
    peak_p: float = 99.75
    peak_blur: float = 1.0
    resp_min: float = 0.05
    max_rad: float = 400.0
    hot_z: float = 12.0
    hot_max: int = 200


@dataclass
class StackingState:
    active: bool = False
    ref_reg: Optional[np.ndarray] = None
    stack_sum: Optional[np.ndarray] = None
    stack_w: Optional[np.ndarray] = None
    ones: Optional[np.ndarray] = None
    frames_used: int = 0
    last_dx: float = 0.0
    last_dy: float = 0.0
    last_resp: float = 0.0
    last_used: int = 0


class StackingEngine:
    def __init__(self, config: StackingConfig | None = None) -> None:
        self.config = config or StackingConfig()
        self.state = StackingState()

    def start(self, height: int, width: int) -> None:
        self.state = StackingState(
            active=True,
            ref_reg=None,
            stack_sum=np.zeros((height, width), dtype=np.float32),
            stack_w=np.zeros((height, width), dtype=np.float32),
            ones=np.ones((height, width), dtype=np.float32),
        )

    def stop(self) -> None:
        self.state.active = False

    def reset(self) -> None:
        self.state = StackingState()

    def step(self, frame_u16: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.state.active:
            return False, None

        cfg = self.config
        state = self.state

        frame_fix = remove_hot_pixels(frame_u16, hot_z=cfg.hot_z, hot_max=cfg.hot_max)

        z = local_zscore_u16(
            frame_fix,
            sigma_bg=float(cfg.sigma_bg),
            floor_p=float(cfg.sigma_floor_p),
            z_clip=float(cfg.z_clip),
        )
        reg = make_sparse_reg(
            z,
            peak_p=float(cfg.peak_p),
            blur_sigma=float(cfg.peak_blur),
            hot_mask=None,
        )

        if state.ref_reg is None:
            state.ref_reg = reg.astype(np.float32).copy()
            dx = dy = 0.0
            resp = 1.0
            used = True
            warped = frame_fix.astype(np.float32)
            wmask = state.ones.copy()
        else:
            dx, dy, resp = pyramid_phasecorr_delta(
                state.ref_reg.astype(np.float32),
                reg.astype(np.float32),
                levels=3,
            )
            rad = float(np.hypot(dx, dy))
            used = True
            if (resp < float(cfg.resp_min)) or (rad > float(cfg.max_rad)) or (not np.isfinite(rad)):
                used = False
                warped = None
                wmask = None
            else:
                warped = warp_translate(frame_fix.astype(np.float32), dx, dy, is_mask=False)
                wmask = warp_translate(state.ones, dx, dy, is_mask=True)

        state.last_dx = float(dx)
        state.last_dy = float(dy)
        state.last_resp = float(resp)
        state.last_used = int(used)

        if used and warped is not None and wmask is not None:
            state.frames_used += 1
            state.stack_sum += warped
            state.stack_w += wmask

        return used, self.stack_image()

    def stack_image(self) -> Optional[np.ndarray]:
        if self.state.stack_sum is None or self.state.stack_w is None:
            return None
        denom = np.maximum(self.state.stack_w, 1e-6)
        return (self.state.stack_sum / denom).astype(np.float32)

    def save_png(self, path: str | Path) -> None:
        import cv2

        stack = self.stack_image()
        if stack is None:
            raise RuntimeError("No stack available.")
        stack_u16 = np.clip(stack, 0, 65535).astype(np.uint16)
        ok = cv2.imwrite(str(path), stack_u16)
        if not ok:
            raise RuntimeError(f"Failed to write stack to {path}.")
