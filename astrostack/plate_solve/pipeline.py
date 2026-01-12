from __future__ import annotations

from typing import Optional

import pandas as pd

from ..gaia_cache import gaia_healpix_cone_with_mag, normalize_input
from .geometry import compute_scale_arcsec_per_px, estimate_theta_max_deg_from_stars
from .matching import build_per_pair_tables, solve_global_consensus


def query_gaia_healpix_df(
    target,
    radius_deg,
    gmax,
    *,
    nside=32,
    auth=None,
    row_limit=-1,
    verbose=True,
    max_gaia_sources: Optional[int] = None,
):
    """
    Envuelve `gaia_healpix_cone_with_mag` y retorna pandas.DataFrame con columnas mínimas.
    max_gaia_sources: si no es None, recorta a las más brillantes (phot_g_mean_mag menor).
    """
    tab = gaia_healpix_cone_with_mag(
        target,
        float(radius_deg),
        gmax=float(gmax),
        nside=int(nside),
        auth=auth,
        row_limit=int(row_limit),
        verbose=verbose,
    )
    df = tab.to_pandas()
    keep = [c for c in ["source_id", "ra", "dec", "phot_g_mean_mag"] if c in df.columns]
    df = df.loc[:, keep].dropna(subset=["ra", "dec"]).reset_index(drop=True)

    if (max_gaia_sources is not None) and ("phot_g_mean_mag" in df.columns) and (len(df) > int(max_gaia_sources)):
        df = df.nsmallest(int(max_gaia_sources), "phot_g_mean_mag").reset_index(drop=True)

    return df


def run_pipeline(
    stars,
    target,
    radius_deg,
    gmax,
    pixel_size_m,
    focal_m,
    tol_rel=0.05,
    max_per_pair=200,
    arcsec_err_cap=0.05,
    nside=32,
    auth=None,
    row_limit=-1,
    plot=True,
    verbose=True,
    label_brightest: int = 20,
    simbad_radius_arcsec: float = 1.0,
    max_gaia_sources: Optional[int] = 8000,
):
    """Orquesta el pipeline completo usando HEALPix + caché para Gaia y SIMBAD para nombres."""
    scale_arcsec_per_px = compute_scale_arcsec_per_px(pixel_size_m, focal_m)
    theta_max_deg = estimate_theta_max_deg_from_stars(stars, scale_arcsec_per_px)
    if verbose:
        print(f"[INFO] Escala = {scale_arcsec_per_px:.3f}\"/px | FOV diag estimada ~ {theta_max_deg:.3f}°")

    c = normalize_input(target)
    if verbose:
        print(f"[INFO] Centro = RA {c.ra.deg:.6f}°, Dec {c.dec.deg:.6f}° | radio = {radius_deg}° | G <= {gmax}")

    if verbose:
        print("[INFO] Descargando Gaia vía HEALPix (caché por tesela)…")
    df_gaia = query_gaia_healpix_df(
        target,
        radius_deg,
        gmax,
        nside=nside,
        auth=auth,
        row_limit=row_limit,
        verbose=verbose,
        max_gaia_sources=max_gaia_sources,
    )
    if verbose:
        print(f"[INFO] Gaia: {len(df_gaia)} fuentes (tras HEALPix + filtro G)")

    per_pair_tables = build_per_pair_tables(
        stars=stars,
        df_gaia=df_gaia,
        scale_arcsec_per_px=scale_arcsec_per_px,
        theta_max_deg=theta_max_deg,
        tol_rel=tol_rel,
        verbose=verbose,
    )

    best_map, best_edges_df, metrics = solve_global_consensus(
        per_pair_tables=per_pair_tables,
        df_gaia=df_gaia,
        max_per_pair=max_per_pair,
        arcsec_err_cap=arcsec_err_cap,
        plot=plot,
        invert_ra=True,
        label_brightest=label_brightest,
        simbad_radius_arcsec=simbad_radius_arcsec,
    )

    return {
        "gaia_df": df_gaia,
        "per_pair_tables": per_pair_tables,
        "best_map": best_map,
        "best_edges_df": best_edges_df,
        "metrics": metrics,
    }
