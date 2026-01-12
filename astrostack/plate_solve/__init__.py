"""Subpaquete de plate solving (Gaia + SIMBAD)."""

from .pipeline import run_pipeline, query_gaia_healpix_df
from .simbad_cache import resolve_bright_star_names_simbad, set_name_cache_dir
from .geometry import (
    radec_to_unitvec,
    chord_radius_from_deg,
    compute_scale_arcsec_per_px,
    estimate_theta_max_deg_from_stars,
    mag_to_size,
    unwrap_ra_for_plot,
)
from .matching import (
    GaiaIndex,
    build_gaia_index,
    candidate_pairs_by_annulus,
    build_per_pair_tables,
    consolidate_candidates,
    build_edge_index,
    candidate_nodes_per_img_star,
    search_assignment,
    solve_global_consensus,
)

__all__ = [
    "run_pipeline",
    "query_gaia_healpix_df",
    "resolve_bright_star_names_simbad",
    "set_name_cache_dir",
    "radec_to_unitvec",
    "chord_radius_from_deg",
    "compute_scale_arcsec_per_px",
    "estimate_theta_max_deg_from_stars",
    "mag_to_size",
    "unwrap_ra_for_plot",
    "GaiaIndex",
    "build_gaia_index",
    "candidate_pairs_by_annulus",
    "build_per_pair_tables",
    "consolidate_candidates",
    "build_edge_index",
    "candidate_nodes_per_img_star",
    "search_assignment",
    "solve_global_consensus",
]
