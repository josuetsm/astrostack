from __future__ import annotations

import math

import numpy as np

# ---- AZ: GT2 belt on disk radius 24cm, pulley 20T, pitch 2mm
AZ_DISK_R_MM = 240.0
GT2_PITCH_MM = 2.0
AZ_PULLEY_T = 20
AZ_PULLEY_C_MM = AZ_PULLEY_T * GT2_PITCH_MM
AZ_DISK_C_MM = 2.0 * math.pi * AZ_DISK_R_MM
AZ_GEAR_RATIO = AZ_DISK_C_MM / AZ_PULLEY_C_MM
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


def mount_predict_altaz_from_micro(mount_state, az_micro: float, alt_micro: float):
    if not mount_state["cal_ok"]:
        return None
    ms_az = int(mount_state["ms_az"])
    ms_alt = int(mount_state["ms_alt"])

    d_az_micro = az_micro - float(mount_state["az0_micro"])
    d_alt_micro = alt_micro - float(mount_state["alt0_micro"])

    az_deg = float(mount_state["az0_sky_deg"]) + mount_state["sign_az"] * az_micro_to_deg(d_az_micro, ms_az)

    c_cm = float(mount_state["alt_c_cm"]) + mount_state["sign_alt"] * alt_micro_to_c_cm(d_alt_micro, ms_alt)
    c_cm = float(np.clip(c_cm, 20.0, 200.0))
    alt_deg = theta_from_c_cm(c_cm)

    az_deg = az_deg % 360.0
    return az_deg, alt_deg, c_cm


def mount_set_calibration(
    mount_state,
    az_center_deg: float,
    alt_center_deg: float,
    az_micro_now: float,
    alt_micro_now: float,
    ms_az: int,
    ms_alt: int,
    c_cm_now: float,
):
    mount_state["cal_ok"] = True
    mount_state["az0_sky_deg"] = float(az_center_deg)
    mount_state["alt0_sky_deg"] = float(alt_center_deg)
    mount_state["az0_micro"] = float(az_micro_now)
    mount_state["alt0_micro"] = float(alt_micro_now)
    mount_state["ms_az"] = int(ms_az)
    mount_state["ms_alt"] = int(ms_alt)
    mount_state["alt_c_cm"] = float(c_cm_now)


def mount_goto_altaz(
    mount_state,
    target_az_deg: float,
    target_alt_deg: float,
    arduino_rate,
    mover_motor,
    log_fn,
    flush_fn,
    motor_ser,
    rate_az_uS: float = 220.0,
    rate_alt_uS: float = 220.0,
    tol_deg: float = 0.05,
):
    if motor_ser is None:
        log_fn("GOTO: Arduino no conectado")
        flush_fn()
        return
    if not mount_state["cal_ok"]:
        log_fn("GOTO: no calibrado aún (haz plate-solve y Accept primero)")
        flush_fn()
        return

    ms_az = int(mount_state["ms_az"])
    ms_alt = int(mount_state["ms_alt"])

    cur = mount_predict_altaz_from_micro(mount_state, mount_state["az_micro"], mount_state["alt_micro"])
    if cur is None:
        return
    cur_az, cur_alt, cur_c = cur

    da = (float(target_az_deg) - float(cur_az) + 540.0) % 360.0 - 180.0
    deg_per_micro = AZ_DEG_PER_MOTOR_REV / (200.0 * ms_az)
    d_micro_az = (da / deg_per_micro) * (1.0 / mount_state["sign_az"])

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
    cm_per_micro = (ALT_PITCH_MM_PER_REV / 10.0) / (200.0 * ms_alt)
    d_micro_alt = (dc / cm_per_micro) * (1.0 / mount_state["sign_alt"])

    delay_az = int(max(200, min(5000, rate_az_uS)))
    delay_alt = int(max(200, min(5000, rate_alt_uS)))

    if abs(d_micro_az) >= 1.0:
        axis = "A"
        direction = "FWD" if d_micro_az >= 0 else "REV"
        steps = int(abs(d_micro_az))
        arduino_rate(0, 0)
        mover_motor(axis, direction, steps, delay_az)
        mount_state["az_micro"] += float(np.sign(d_micro_az) * steps)

    if abs(d_micro_alt) >= 1.0:
        axis = "B"
        direction = "FWD" if d_micro_alt >= 0 else "REV"
        steps = int(abs(d_micro_alt))
        arduino_rate(0, 0)
        mover_motor(axis, direction, steps, delay_alt)
        mount_state["alt_micro"] += float(np.sign(d_micro_alt) * steps)

    cur2 = mount_predict_altaz_from_micro(mount_state, mount_state["az_micro"], mount_state["alt_micro"])
    if cur2 is not None:
        _, _, c2 = cur2
        mount_state["alt_c_cm"] = float(c2)

    log_fn(
        f"GOTO: target(AZ,ALT)=({target_az_deg:.2f},{target_alt_deg:.2f}) "
        f"da={da:+.2f}deg dµ_az={d_micro_az:+.0f} dµ_alt={d_micro_alt:+.0f} -> done"
    )
    flush_fn()
