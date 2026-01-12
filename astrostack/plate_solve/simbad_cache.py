from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

_NAME_CACHE_DIR = Path(os.environ.get("GAIA_NAME_CACHE_DIR", "~/.cache/gaia_names")).expanduser()
_NAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_NAME_CACHE_FILE = _NAME_CACHE_DIR / "simbad_names.json"


def set_name_cache_dir(path: str) -> None:
    """Cambia el directorio de caché de nombres SIMBAD."""
    global _NAME_CACHE_DIR, _NAME_CACHE_FILE
    _NAME_CACHE_DIR = Path(path).expanduser().resolve()
    _NAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _NAME_CACHE_FILE = _NAME_CACHE_DIR / "simbad_names.json"


def _load_name_cache() -> Dict[str, str]:
    if _NAME_CACHE_FILE.exists():
        try:
            return json.loads(_NAME_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_name_cache(d: Dict[str, str]) -> None:
    try:
        _NAME_CACHE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _pick_preferred_alias(
    ids_list: Iterable[str],
    prefer_prefixes=("HD", "HR", "HIP", "TYC", "Bayer", "Flamsteed"),
) -> Optional[str]:
    for pref in prefer_prefixes:
        for s in ids_list:
            if s.upper().startswith(pref.upper() + " "):
                return s
    return None


def _first_col_case_insensitive(tab, candidates: Iterable[str]) -> Optional[str]:
    """Retorna el nombre real de la columna que matchee candidates (case-insensitive)."""
    if tab is None:
        return None
    colnames = list(getattr(tab, "colnames", []))
    if not colnames:
        return None

    for c in candidates:
        if c in colnames:
            return c

    low = {c.lower(): c for c in colnames}
    for c in candidates:
        k = c.lower()
        if k in low:
            return low[k]

    return None


def _as_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def resolve_bright_star_names_simbad(
    df_gaia: pd.DataFrame,
    top_n: int = 20,
    search_radius_arcsec: float = 1.0,
    sleep_sec: float = 0.2,
    prefer_prefixes=("HD", "HR", "HIP", "TYC", "Bayer", "Flamsteed"),
    verbose: bool = True,
) -> Dict[int, str]:
    """
    Dada una tabla Gaia (con columnas 'ra','dec','phot_g_mean_mag','source_id'),
    resuelve nombres SIMBAD para las N estrellas más brillantes (menor magnitud).
    Devuelve dict: índice_de_df_gaia -> nombre_preferido.
    Usa caché simple por source_id.
    """
    if not len(df_gaia):
        return {}
    req_cols = {"ra", "dec", "phot_g_mean_mag"}
    if not req_cols.issubset(set(df_gaia.columns)):
        raise ValueError("df_gaia debe contener columnas: 'ra','dec','phot_g_mean_mag'")

    cache = _load_name_cache()
    out: Dict[int, str] = {}

    brightest = df_gaia.nsmallest(int(top_n), "phot_g_mean_mag").reset_index()

    sim = Simbad()
    sim.add_votable_fields("ids", "ra", "dec")

    for _, row in brightest.iterrows():
        idx = int(row["index"])
        sid = (
            str(int(row["source_id"]))
            if "source_id" in df_gaia.columns
            else f"{float(row['ra']):.6f}_{float(row['dec']):.6f}"
        )
        if sid in cache:
            out[idx] = cache[sid]
            continue

        coord = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg, frame="icrs")
        try:
            res = sim.query_region(coord, radius=float(search_radius_arcsec) * u.arcsec)
        except Exception as e:
            if verbose:
                print(f"[SIMBAD] Error cerca de ({float(row['ra']):.6f},{float(row['dec']):.6f}): {e}")
            continue

        if res is None or len(res) == 0:
            continue

        col_main = _first_col_case_insensitive(res, ["MAIN_ID", "main_id"])
        col_ids = _first_col_case_insensitive(res, ["IDS", "ids"])

        if col_main is not None:
            main_id = _as_str(res[col_main][0]).strip()
        else:
            try:
                main_id = _as_str(res[0][0]).strip()
            except Exception:
                main_id = "UNKNOWN"

        ids: list[str] = []
        if col_ids is not None:
            raw = _as_str(res[col_ids][0])
            ids = [s.strip() for s in raw.split("|") if s.strip()]

        alias = _pick_preferred_alias(ids, prefer_prefixes=prefer_prefixes)
        name = alias or main_id

        cache[sid] = name
        out[idx] = name
        time.sleep(max(0.0, float(sleep_sec)))

    _save_name_cache(cache)
    return out
