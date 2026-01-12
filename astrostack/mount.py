from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


AZ_DISK_R_MM = 240.0
GT2_PITCH_MM = 2.0
AZ_PULLEY_T = 20
AZ_PULLEY_C_MM = AZ_PULLEY_T * GT2_PITCH_MM
AZ_DISK_C_MM = 2.0 * math.pi * AZ_DISK_R_MM
AZ_GEAR_RATIO = AZ_DISK_C_MM / AZ_PULLEY_C_MM
AZ_DEG_PER_MOTOR_REV = 360.0 / AZ_GEAR_RATIO

ALT_A_CM = 39.5
ALT_B_CM = 44.5
ALT_C0_CM = 65.5
ALT_THETA0_DEG = 5.8
ALT_PITCH_MM_PER_REV = 10.0 / 7.0


def gamma_from_c_cm(c_cm: float) -> float:
    a = ALT_A_CM
    b = ALT_B_CM
    c = float(c_cm)
    num = a * a + b * b - c * c
    den = 2.0 * a * b
    x = num / den
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))


GAMMA0_DEG = gamma_from_c_cm(ALT_C0_CM)
PHI_B_DEG = ALT_THETA0_DEG + GAMMA0_DEG


def theta_from_c_cm(c_cm: float) -> float:
    return PHI_B_DEG - gamma_from_c_cm(c_cm)


def az_micro_to_deg(delta_micro: float, ms_az: int) -> float:
    motor_rev = delta_micro / (200.0 * ms_az)
    return motor_rev * AZ_DEG_PER_MOTOR_REV


def alt_micro_to_c_cm(delta_micro: float, ms_alt: int) -> float:
    motor_rev = delta_micro / (200.0 * ms_alt)
    delta_mm = motor_rev * ALT_PITCH_MM_PER_REV
    return delta_mm / 10.0


@dataclass
class MountState:
    cal_ok: bool = False
    az_micro: float = 0.0
    alt_micro: float = 0.0
    ms_az: int = 64
    ms_alt: int = 64
    c_cm_cal: float = ALT_C0_CM
    az0_sky_deg: Optional[float] = None
    alt0_sky_deg: Optional[float] = None
    az0_micro: Optional[float] = None
    alt0_micro: Optional[float] = None
    sign_az: float = 1.0
    sign_alt: float = 1.0


class MountModel:
    def __init__(self, state: Optional[MountState] = None) -> None:
        self.state = state or MountState()

    def calibrate(
        self,
        *,
        az_center_deg: float,
        alt_center_deg: float,
        az_micro_now: float,
        alt_micro_now: float,
        ms_az: int,
        ms_alt: int,
        c_cm_now: float,
    ) -> None:
        self.state.cal_ok = True
        self.state.az0_sky_deg = float(az_center_deg)
        self.state.alt0_sky_deg = float(alt_center_deg)
        self.state.az0_micro = float(az_micro_now)
        self.state.alt0_micro = float(alt_micro_now)
        self.state.ms_az = int(ms_az)
        self.state.ms_alt = int(ms_alt)
        self.state.c_cm_cal = float(c_cm_now)

    def predict_altaz(self) -> Optional[Tuple[float, float, float]]:
        state = self.state
        if not state.cal_ok:
            return None
        if state.az0_micro is None or state.alt0_micro is None:
            return None
        if state.az0_sky_deg is None or state.alt0_sky_deg is None:
            return None

        d_az_micro = state.az_micro - state.az0_micro
        d_alt_micro = state.alt_micro - state.alt0_micro

        az_deg = float(state.az0_sky_deg) + state.sign_az * az_micro_to_deg(d_az_micro, state.ms_az)
        c_cm = float(state.c_cm_cal) + state.sign_alt * alt_micro_to_c_cm(d_alt_micro, state.ms_alt)
        c_cm = float(np.clip(c_cm, 20.0, 200.0))
        alt_deg = theta_from_c_cm(c_cm)

        az_deg = az_deg % 360.0
        return az_deg, alt_deg, c_cm

    def compute_goto(
        self,
        *,
        target_az_deg: float,
        target_alt_deg: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        cur = self.predict_altaz()
        if cur is None:
            return None
        cur_az, _cur_alt, cur_c = cur
        da = (float(target_az_deg) - float(cur_az) + 540.0) % 360.0 - 180.0
        deg_per_micro = AZ_DEG_PER_MOTOR_REV / (200.0 * self.state.ms_az)
        d_micro_az = (da / deg_per_micro) * (1.0 / self.state.sign_az)

        target_alt = float(target_alt_deg)
        lo, hi = 20.0, 200.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            th = theta_from_c_cm(mid)
            if th < target_alt:
                hi = mid
            else:
                lo = mid
        c_target = 0.5 * (lo + hi)

        dc = c_target - cur_c
        cm_per_micro = (ALT_PITCH_MM_PER_REV / 10.0) / (200.0 * self.state.ms_alt)
        d_micro_alt = (dc / cm_per_micro) * (1.0 / self.state.sign_alt)
        return d_micro_az, d_micro_alt, da, c_target, cur_c

    def apply_move(self, delta_micro_az: float, delta_micro_alt: float) -> None:
        self.state.az_micro += float(delta_micro_az)
        self.state.alt_micro += float(delta_micro_alt)
