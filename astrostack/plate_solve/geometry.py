from __future__ import annotations

from math import sqrt

import numpy as np


def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def chord_radius_from_deg(theta_deg):
    theta = np.deg2rad(theta_deg)
    return 2.0 * np.sin(theta / 2.0)


def compute_scale_arcsec_per_px(pixel_size_m, focal_m):
    return (206265.0 * float(pixel_size_m)) / float(focal_m)


def estimate_theta_max_deg_from_stars(stars, scale_arcsec_per_px, min_cap_deg=0.30):
    st = np.asarray(stars, dtype=np.float64)
    xmin, xmax = st[:, 0].min(), st[:, 0].max()
    ymin, ymax = st[:, 1].min(), st[:, 1].max()
    diag_px = sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    theta = (diag_px * scale_arcsec_per_px) / 3600.0
    return max(theta, float(min_cap_deg))


def mag_to_size(
    mag: np.ndarray,
    *,
    m_ref: float = 8.0,
    size_ref: float = 120.0,
    min_size: float = 6.0,
    max_size: float = 160.0,
) -> np.ndarray:
    """
    Escala fotométrica simple: size = size_ref * 10^(-0.4*(mag - m_ref)).
    Clampea a [min_size, max_size].
    """
    m = np.asarray(mag, dtype=float)
    size = size_ref * (10.0 ** (-0.4 * (m - m_ref)))
    return np.clip(size, min_size, max_size)


def _circular_mean_deg(ra_deg: np.ndarray) -> float:
    """Media circular de ángulos en grados (retorna en [0,360))."""
    a = np.deg2rad(np.asarray(ra_deg, dtype=float))
    s = np.sin(a).mean()
    c = np.cos(a).mean()
    ang = np.arctan2(s, c)
    deg = np.rad2deg(ang) % 360.0
    return float(deg)


def unwrap_ra_for_plot(ra_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """
    Devuelve RA continua alrededor de center_deg.
    Implementación: (ra - center) envuelto a (-180,180] y luego re-sumado al centro.
    Resultado: puede exceder 360 o ser negativo, pero es continuo y evita saltos.
    """
    ra = np.asarray(ra_deg, dtype=float)
    x = ((ra - center_deg + 180.0) % 360.0) - 180.0
    return x + center_deg
