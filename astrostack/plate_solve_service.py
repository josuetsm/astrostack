from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

from .plate_solve.pipeline import run_pipeline
from .preprocessing import StretchConfig, stretch_to_u8
from .star_detection import StarDetectionConfig, detect_stars


@dataclass(frozen=True)
class PlateSolveSettings:
    target: str = "M42"
    radius_deg: float = 1.0
    gmax: float = 12.0
    pixel_size_um: float = 2.9
    focal_length_mm: float = 900.0
    tol_rel: float = 0.05
    max_per_pair: int = 200
    arcsec_err_cap: float = 0.05
    nside: int = 32
    row_limit: int = -1
    max_gaia_sources: int = 8000
    label_brightest: int = 20
    simbad_radius_arcsec: float = 1.0


@dataclass(frozen=True)
class PlateSolveResult:
    stars: list[tuple[float, float, float]]
    solution: dict[str, Any]
    center_radec: tuple[float, float] | None


def _circular_mean_deg(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 0.0
    rad = np.deg2rad(values)
    mean = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    deg = np.rad2deg(mean)
    if deg < 0:
        deg += 360.0
    return float(deg)


def _solution_center_radec(solution: dict[str, Any]) -> tuple[float, float] | None:
    df_gaia = solution.get("gaia_df")
    best_map = solution.get("best_map")
    if df_gaia is None or not best_map:
        return None
    idxs = sorted(set(best_map.values()))
    if not idxs:
        return None
    ra = df_gaia.iloc[idxs]["ra"].to_numpy(np.float64)
    dec = df_gaia.iloc[idxs]["dec"].to_numpy(np.float64)
    if ra.size == 0:
        return None
    return _circular_mean_deg(ra), float(np.mean(dec))


def solve_from_stack(
    stack_img: np.ndarray,
    *,
    settings: PlateSolveSettings | None = None,
    stretch: StretchConfig | None = None,
    detect_cfg: StarDetectionConfig | None = None,
) -> PlateSolveResult:
    if stack_img is None:
        raise ValueError("stack_img is required for plate solving.")

    settings = settings or PlateSolveSettings()
    stretch_cfg = stretch or StretchConfig()

    preview_u8 = stretch_to_u8(stack_img, stretch_cfg)
    candidates = detect_stars(preview_u8, config=detect_cfg)
    stars = [(float(x), float(y), float(score)) for (y, x, score) in candidates]
    if len(stars) < 4:
        return PlateSolveResult(stars=stars, solution={}, center_radec=None)

    solution = run_pipeline(
        stars=stars,
        target=settings.target,
        radius_deg=settings.radius_deg,
        gmax=settings.gmax,
        pixel_size_m=float(settings.pixel_size_um) * 1e-6,
        focal_m=float(settings.focal_length_mm) * 1e-3,
        tol_rel=settings.tol_rel,
        max_per_pair=settings.max_per_pair,
        arcsec_err_cap=settings.arcsec_err_cap,
        nside=settings.nside,
        row_limit=settings.row_limit,
        plot=False,
        verbose=False,
        label_brightest=settings.label_brightest,
        simbad_radius_arcsec=settings.simbad_radius_arcsec,
        max_gaia_sources=settings.max_gaia_sources,
    )
    center = _solution_center_radec(solution)
    return PlateSolveResult(stars=stars, solution=solution, center_radec=center)
