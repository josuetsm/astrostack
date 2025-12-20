# -*- coding: utf-8 -*-
"""
gaia_cache.py — Consultas Gaia (astroquery) con caché y mosaico HEALPix (async, login único).

Cambios (para tu caso):
- Evita loggear en cache-hit por defecto (log_cache_hits=False).
- En HEALPix: si TODAS las teselas están en caché, NO imprime nada y NO hace login.
- Mantiene verbose para cuando hay misses (consultas reales).

Requisitos
----------
- astroquery, astropy
- astropy-healpix (para función HEALPix)
- (opcional) pyarrow para parquet

Licencia: MIT
"""
from __future__ import annotations

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.table import Table, vstack

# HEALPix puede venir como paquete aparte
try:
    from astropy_healpix import HEALPix
except Exception:
    HEALPix = None  # validaremos cuando se use

# parquet opcional
try:
    import pyarrow  # type: ignore  # noqa: F401
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

# -------------------------
# Config caché y defaults
# -------------------------
_DEFAULT_CACHE_DIR = Path(os.environ.get("GAIA_CONE_CACHE_DIR", "~/.cache/gaia_cones")).expanduser()
DEFAULT_TABLE = "gaiadr3.gaia_source"
DEFAULT_COLUMNS = ("source_id", "ra", "dec", "phot_g_mean_mag")


# -------------------------
# Normalización de inputs
# -------------------------
def normalize_input(target) -> SkyCoord:
    """
    Acepta:
      - 'Ankaa' → SIMBAD (SkyCoord.from_name)
      - '00:26:14.8 -39:39:00.7' → sexagesimal
      - (ra_deg, dec_deg) o [ra_deg, dec_deg] → grados
      - {'ra': 6.5, 'dec': -39.6} → grados
    Devuelve SkyCoord(ICRS).
    """
    if isinstance(target, str):
        if any(ch.isalpha() for ch in target):
            return SkyCoord.from_name(target)
        ra_str, dec_str = target.split()
        if ":" in ra_str:
            return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        return SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg, frame="icrs")

    if isinstance(target, (tuple, list)) and len(target) == 2:
        ra, dec = target
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")

    if isinstance(target, dict) and {"ra", "dec"} <= target.keys():
        return SkyCoord(float(target["ra"]) * u.deg, float(target["dec"]) * u.deg, frame="icrs")

    raise ValueError(f"Formato de target no reconocido: {target}")


# -------------------------
# Utilidades de caché
# -------------------------
def set_cache_dir(path: Union[str, Path]) -> None:
    """Cambia el directorio base de caché en tiempo de ejecución."""
    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = Path(path).expanduser().resolve()
    _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(*, kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _path_for(hexkey: str, prefer_parquet: bool) -> Path:
    ext = "parquet" if (prefer_parquet and _HAS_PARQUET) else "ecsv"
    return _DEFAULT_CACHE_DIR.joinpath(hexkey[:2], hexkey[2:4], f"{hexkey}.{ext}")


def _save_table(tab: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        import pandas as pd
        tab.to_pandas().to_parquet(path, index=False)
    else:
        tab.write(path, format="ascii.ecsv", overwrite=True, fast_writer=False)


def _load_table(path: Path) -> Table:
    if path.suffix.lower() == ".parquet":
        import pandas as pd
        return Table.from_pandas(pd.read_parquet(path))
    return Table.read(path, format="ascii.ecsv")


# -------------------------
# Cono único (async)
# -------------------------
def gaia_cone_with_mag(
    target,
    radius: Union[float, u.Quantity],
    *,
    gmax: float = 15.0,
    table_name: str = DEFAULT_TABLE,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    auth: Optional[Tuple[str, str]] = None,
    row_limit: int = -1,
    prefer_parquet: bool = True,
    retries: int = 3,
    backoff_s: float = 3.0,
    verbose: bool = True,
    log_cache_hits: bool = False,
    log_cache_misses: bool = True,
) -> Table:
    """
    Cone search con filtro 'phot_g_mean_mag <= gmax' en el servidor (ADQL),
    caché por parámetros. Usa launch_job_async(background=False).

    Notas:
    - Por defecto NO imprime hits (log_cache_hits=False).
    - Solo imprime miss si log_cache_misses=True.
    """
    center = normalize_input(target)
    ra_deg, dec_deg = center.ra.deg, center.dec.deg
    radius_deg = (radius.to_value(u.deg) if isinstance(radius, u.Quantity) else float(radius))

    hexkey = _cache_key(kind="cone", payload={
        "table": table_name,
        "ra": round(ra_deg, 8),
        "dec": round(dec_deg, 8),
        "radius": round(radius_deg, 8),
        "gmax": float(gmax),
        "columns": list(columns),
    })
    path = _path_for(hexkey, prefer_parquet)
    if path.exists():
        if verbose and log_cache_hits:
            print(f"[gaia_cache] HIT {path}")
        return _load_table(path)

    Gaia.ROW_LIMIT = row_limit
    cols_sql = ", ".join(columns)
    query = f"""
    SELECT {cols_sql}
    FROM {table_name}
    WHERE phot_g_mean_mag <= {gmax}
      AND 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
          )
    """

    did_login = False
    try:
        if auth:
            if verbose:
                print("[gaia_cache] Login al Gaia Archive…")
            Gaia.login(user=auth[0], password=auth[1])
            did_login = True

        for attempt in range(1, retries + 1):
            try:
                job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=verbose)
                tab = job.get_results()
                break
            except Exception as e:
                if attempt == retries:
                    raise
                if verbose:
                    print(f"[gaia_cache] retry {attempt}: {type(e).__name__} -> {e}")
                time.sleep(backoff_s * attempt)

    finally:
        if did_login:
            if verbose:
                print("[gaia_cache] Logout del Gaia Archive.")
            try:
                Gaia.logout()
            except Exception:
                pass

    # dedup por source_id
    if "source_id" in tab.colnames:
        try:
            import pandas as pd
            tab = Table.from_pandas(tab.to_pandas().drop_duplicates(subset=["source_id"]))
        except Exception:
            seen = set()
            keep = []
            for i, sid in enumerate(tab["source_id"]):
                sid = int(sid)
                if sid not in seen:
                    seen.add(sid)
                    keep.append(i)
            tab = tab[keep]

    _save_table(tab, path)
    if verbose and log_cache_misses:
        print(f"[gaia_cache] MISS -> saved {len(tab)} rows to {path}")
    return tab


# -------------------------
# HEALPix helpers
# -------------------------
def _ensure_healpix_available():
    if HEALPix is None:
        raise ImportError("astropy-healpix no está disponible. Instala 'astropy-healpix'.")


def _adql_polygon_from_skycoord(poly: SkyCoord) -> str:
    """
    Convierte vértices SkyCoord a ADQL POLYGON('ICRS', lon1,lat1, ..., lonN,latN).
    Acepta arrays con cualquier forma; se aplana.
    """
    import numpy as np
    lon = np.asarray(poly.ra.deg).ravel()
    lat = np.asarray(poly.dec.deg).ravel()
    pairs = ", ".join(f"{float(lon_i):.10f},{float(lat_i):.10f}" for lon_i, lat_i in zip(lon, lat))
    return f"POLYGON('ICRS', {pairs})"


def _query_healpix_tile_async(
    *,
    table_name: str,
    columns: Sequence[str],
    gmax: float,
    poly_sky: SkyCoord,
    row_limit: int,
    retries: int,
    backoff_s: float,
    verbose: bool,
) -> Table:
    cols_sql = ", ".join(columns)
    poly_adql = _adql_polygon_from_skycoord(poly_sky)
    query = f"""
    SELECT {cols_sql}
    FROM {table_name}
    WHERE phot_g_mean_mag <= {gmax}
      AND 1=CONTAINS(POINT('ICRS', ra, dec), {poly_adql})
    """
    Gaia.ROW_LIMIT = row_limit

    for attempt in range(1, retries + 1):
        try:
            job = Gaia.launch_job_async(query, background=False, dump_to_file=False, verbose=verbose)
            return job.get_results()
        except Exception as e:
            if attempt == retries:
                raise
            if verbose:
                print(f"[gaia_healpix] retry {attempt}: {type(e).__name__} -> {e}")
            time.sleep(backoff_s * attempt)


# -------------------------
# Mosaico HEALPix (async, login único)
# -------------------------
def gaia_healpix_cone_with_mag(
    target,
    radius: Union[float, u.Quantity],
    *,
    gmax: float = 15.0,
    nside: int = 64,
    order: str = "ring",
    table_name: str = DEFAULT_TABLE,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    auth: Optional[Tuple[str, str]] = None,
    row_limit: int = -1,
    prefer_parquet: bool = True,
    retries: int = 3,
    backoff_s: float = 3.0,
    verbose: bool = True,
    log_cache_hits: bool = False,
    log_cache_misses: bool = True,
    quiet_if_fully_cached: bool = True,
) -> Table:
    """
    HEALPix mosaico con filtro 'phot_g_mean_mag <= gmax'.

    Cambios clave:
    - No imprime nada si todas las teselas ya están cacheadas (quiet_if_fully_cached=True).
    - No hace login si no hay misses.
    - log_cache_hits=False por defecto.
    """
    _ensure_healpix_available()

    center = normalize_input(target)
    radius_deg = (radius.to_value(u.deg) if isinstance(radius, u.Quantity) else float(radius))

    hp = HEALPix(nside=nside, order=order, frame=center.frame)
    pix_indices = hp.cone_search_skycoord(center, Angle(radius_deg, u.deg))

    # Precomputar paths para saber si está todo en caché
    tile_meta: List[Tuple[int, Path]] = []
    for pix in pix_indices:
        hexkey = _cache_key(kind="healpix_tile", payload={
            "table": table_name,
            "nside": int(nside),
            "order": str(order),
            "pix": int(pix),
            "gmax": float(gmax),
            "columns": list(columns),
        })
        path = _path_for(hexkey, prefer_parquet)
        tile_meta.append((int(pix), path))

    all_cached = all(p.exists() for _, p in tile_meta)
    v = bool(verbose) and not (quiet_if_fully_cached and all_cached)

    if v:
        print(f"[gaia_healpix] nside={nside}, tiles={len(pix_indices)}")

    did_login = False
    parts: List[Table] = []

    try:
        if auth and not all_cached:
            if v:
                print("[gaia_healpix] Login único al Gaia Archive…")
            Gaia.login(user=auth[0], password=auth[1])
            did_login = True

        for i, (pix, path) in enumerate(tile_meta, 1):
            if path.exists():
                if v and log_cache_hits:
                    # opcional: log hit (desactivado por defecto)
                    print(f"[gaia_healpix] HIT tile {i}/{len(tile_meta)} (pix={pix})")
                tab = _load_table(path)
            else:
                if v and log_cache_misses:
                    print(f"[gaia_healpix] Query tile {i}/{len(tile_meta)} (pix={pix})")
                poly = hp.boundaries_skycoord(pix, step=1)
                tab = _query_healpix_tile_async(
                    table_name=table_name,
                    columns=columns,
                    gmax=gmax,
                    poly_sky=poly,
                    row_limit=row_limit,
                    retries=retries,
                    backoff_s=backoff_s,
                    verbose=False,  # no spamear por tesela
                )
                _save_table(tab, path)
            parts.append(tab)

    finally:
        if did_login:
            if v:
                print("[gaia_healpix] Logout del Gaia Archive.")
            try:
                Gaia.logout()
            except Exception:
                pass

    if not parts:
        return Table(names=list(columns), dtype=[float] * len(columns))

    full = vstack(parts, join_type="outer", metadata_conflicts="silent")

    # Deduplicación por source_id
    if "source_id" in full.colnames:
        try:
            import pandas as pd
            full = Table.from_pandas(full.to_pandas().drop_duplicates(subset=["source_id"]))
        except Exception:
            seen = set()
            keep = []
            for i, sid in enumerate(full["source_id"]):
                sid = int(sid)
                if sid not in seen:
                    seen.add(sid)
                    keep.append(i)
            full = full[keep]

    # Recorte fino al círculo exacto
    sc = SkyCoord(full["ra"] * u.deg, full["dec"] * u.deg, frame="icrs")
    sep = sc.separation(center).deg
    mask = sep <= radius_deg
    full = full[mask]

    if v and log_cache_misses:
        print(f"[gaia_healpix] Final rows (G<={gmax}): {len(full)}")

    return full


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gaia queries with caching and HEALPix tiling (async).")
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--radius", type=float, required=True, help="en grados")
    p.add_argument("--gmax", type=float, default=15.0)
    p.add_argument("--healpix", action="store_true")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    tgt = (args.ra, args.dec)
    if args.healpix:
        out = gaia_healpix_cone_with_mag(tgt, args.radius, gmax=args.gmax, nside=args.nside, verbose=args.verbose)
    else:
        out = gaia_cone_with_mag(tgt, args.radius, gmax=args.gmax, verbose=args.verbose)
    print(out[:10])
