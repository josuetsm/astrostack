from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import (
    _circular_mean_deg,
    chord_radius_from_deg,
    mag_to_size,
    radec_to_unitvec,
    unwrap_ra_for_plot,
)
from .simbad_cache import resolve_bright_star_names_simbad


@dataclass
class GaiaIndex:
    df: pd.DataFrame
    U: np.ndarray
    tree: cKDTree


def build_gaia_index(df_gaia: pd.DataFrame) -> GaiaIndex:
    ra = df_gaia["ra"].to_numpy(np.float64)
    dec = df_gaia["dec"].to_numpy(np.float64)
    U = radec_to_unitvec(ra, dec)
    tree = cKDTree(U)
    return GaiaIndex(df=df_gaia, U=U, tree=tree)


def candidate_pairs_by_annulus(
    df_or_index,
    target_arcsec,
    theta_max_deg,
    tol_rel=0.05,
):
    """
    Busca pares (i,j) en Gaia cuya separación esté en [d*(1-tol), d*(1+tol)], limitado por 'theta_max_deg'.
    Retorna lista de dicts con: i, j, sep_arcsec, G_sum, ra_i, dec_i, ra_j, dec_j, diff_arcsec

    Optimización:
    - df_or_index puede ser pd.DataFrame o GaiaIndex (recomendado).
    """
    if isinstance(df_or_index, GaiaIndex):
        df = df_or_index.df
        U = df_or_index.U
        tree = df_or_index.tree
    else:
        df = df_or_index
        idx = build_gaia_index(df)
        U = idx.U
        tree = idx.tree

    sep_deg = float(target_arcsec) / 3600.0
    r_hi = chord_radius_from_deg(min(sep_deg * (1.0 + tol_rel), float(theta_max_deg)))
    r_lo = chord_radius_from_deg(max(sep_deg * (1.0 - tol_rel), 0.0))

    out = []
    N = len(df)

    for i in range(N):
        nbrs_hi = tree.query_ball_point(U[i], r=r_hi)
        nbrs_lo = set(tree.query_ball_point(U[i], r=r_lo)) if r_lo > 0 else set()
        for j in nbrs_hi:
            if j <= i or j in nbrs_lo:
                continue
            cosang = np.clip(np.dot(U[i], U[j]), -1.0, 1.0)
            ang = np.arccos(cosang)
            sep_as = np.rad2deg(ang) * 3600.0
            if abs(sep_as - target_arcsec) <= tol_rel * target_arcsec:
                out.append(
                    {
                        "i": i,
                        "j": j,
                        "sep_arcsec": sep_as,
                        "G_sum": float(df.at[i, "phot_g_mean_mag"] + df.at[j, "phot_g_mean_mag"]),
                        "ra_i": float(df.at[i, "ra"]),
                        "dec_i": float(df.at[i, "dec"]),
                        "ra_j": float(df.at[j, "ra"]),
                        "dec_j": float(df.at[j, "dec"]),
                        "diff_arcsec": abs(sep_as - target_arcsec),
                    }
                )

    out.sort(key=lambda d: (d["diff_arcsec"], d["G_sum"]))
    return out


def build_per_pair_tables(
    stars,
    df_gaia,
    scale_arcsec_per_px,
    theta_max_deg,
    tol_rel=0.05,
    verbose=True,
):
    st = np.asarray(stars, dtype=np.float64)
    idxs = list(range(len(st)))
    img_pairs = list(combinations(idxs, 2))

    gaia_index = build_gaia_index(df_gaia)

    per_pair_tables = []
    for (p, q) in img_pairs:
        dpx = np.linalg.norm(st[q, :2] - st[p, :2])
        d_arc = dpx * scale_arcsec_per_px
        cand = candidate_pairs_by_annulus(gaia_index, d_arc, theta_max_deg, tol_rel=tol_rel)
        if verbose:
            print(f"[INFO] Par imagen {p}-{q}: d={d_arc:.2f}\" -> candidatos={len(cand)}")
        if len(cand):
            tab = pd.DataFrame(cand)
            tab.insert(0, "pair", f"{min(p,q)}-{max(p,q)}")
        else:
            tab = pd.DataFrame(
                columns=["pair", "i", "j", "sep_arcsec", "diff_arcsec", "G_sum", "ra_i", "dec_i", "ra_j", "dec_j"]
            )
        per_pair_tables.append(((p, q), d_arc, tab))
    return per_pair_tables


def _infer_n_img_from_per_pair_tables(per_pair_tables):
    M = len(per_pair_tables)
    if M <= 0:
        return 0
    n_est = int(np.rint((1.0 + np.sqrt(1.0 + 8.0 * M)) / 2.0))
    return max(n_est, 0)


def _parse_pair_str(pair_str):
    a, b = pair_str.split("-")
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def consolidate_candidates(per_pair_tables, max_per_pair=200, arcsec_err_cap=0.05):
    rows = []
    pairs_order = []

    for k, item in enumerate(per_pair_tables):
        (p, q), d_arc, tab = item
        pairs_order.append((min(p, q), max(p, q)))
        if tab is None or not len(tab):
            continue

        tmp = tab.copy()
        tmp["pair_k"] = k
        tmp["sep_true"] = float(d_arc)

        with np.errstate(divide="ignore", invalid="ignore"):
            arcsec_err = (tmp["sep_arcsec"] - tmp["sep_true"]).abs()
        tmp["arcsec_err"] = arcsec_err.fillna(np.inf)

        tmp = tmp[(tmp["arcsec_err"] <= arcsec_err_cap)].copy()
        if not len(tmp):
            continue

        key = tmp.apply(lambda r: tuple(sorted((int(r["i"]), int(r["j"])))), axis=1)
        tmp["_edge"] = key
        tmp = tmp.sort_values(["arcsec_err", "G_sum"]).groupby("_edge", as_index=False).first()
        tmp = tmp.sort_values(["arcsec_err", "G_sum"]).head(int(max_per_pair))
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(), [], 0

    C = pd.concat(rows, ignore_index=True)
    if "pair" in C.columns:
        C["pair_tuple"] = C["pair"].apply(_parse_pair_str)
    else:
        mapping = dict(enumerate(pairs_order))
        C["pair_tuple"] = C["pair_k"].map(mapping)

    N_img = _infer_n_img_from_per_pair_tables(per_pair_tables)
    if "diff_arcsec" not in C.columns:
        C["diff_arcsec"] = (C["sep_arcsec"] - C["sep_true"]).abs()
    return C.reset_index(drop=True), pairs_order, N_img


def build_edge_index(C):
    edge_by_pair = defaultdict(list)
    edge_err = defaultdict(dict)

    for idx, r in C.iterrows():
        a, b = r["pair_tuple"]
        gi, gj = int(r["i"]), int(r["j"])
        x, y = (gi, gj) if gi < gj else (gj, gi)

        if (x, y) not in {(e[0], e[1]) for e in edge_by_pair[(a, b)]}:
            edge_by_pair[(a, b)].append((x, y, float(r["arcsec_err"]), int(idx)))

        edge_err[(a, b)][(gi, gj)] = min(edge_err[(a, b)].get((gi, gj), np.inf), float(r["arcsec_err"]))
        edge_err[(a, b)][(gj, gi)] = min(edge_err[(a, b)].get((gj, gi), np.inf), float(r["arcsec_err"]))

    for k in edge_by_pair:
        edge_by_pair[k].sort(key=lambda t: t[2])
    return edge_by_pair, edge_err


def candidate_nodes_per_img_star(edge_by_pair, N_img):
    pairs_by_star = defaultdict(list)
    for (a, b) in edge_by_pair.keys():
        pairs_by_star[a].append((a, b))
        pairs_by_star[b].append((a, b))

    node_stats = [defaultdict(lambda: {"count": 0, "errsum": 0.0}) for _ in range(N_img)]

    for s in range(N_img):
        for (a, b) in pairs_by_star[s]:
            lst = edge_by_pair[(a, b)]
            for (gi, gj, err, _) in lst:
                if s == a:
                    node_stats[s][gi]["count"] += 1
                    node_stats[s][gi]["errsum"] += err
                    node_stats[s][gj]
                else:
                    node_stats[s][gj]["count"] += 1
                    node_stats[s][gj]["errsum"] += err
                    node_stats[s][gi]

    cand_list = []
    for s in range(N_img):
        items = []
        for g, st in node_stats[s].items():
            items.append((g, st["count"], st["errsum"]))
        items.sort(key=lambda t: (-t[1], t[2], t[0]))
        cand_list.append([g for (g, _, __) in items])
    return cand_list


def search_assignment(edge_err, N_img, cand_nodes, pairs_order):
    order = list(range(N_img))
    order.sort(key=lambda s: len(cand_nodes[s]) if len(cand_nodes[s]) > 0 else 1e9)

    best = {"err_sum": np.inf, "assign": None}
    assign = {}
    used = set()

    def partial_err():
        err_sum = 0.0
        for (a, b) in pairs_order:
            if a in assign and b in assign:
                gi, gj = assign[a], assign[b]
                val = edge_err[(a, b)].get((gi, gj), np.inf)
                if not np.isfinite(val):
                    return np.inf
                err_sum += val
        return err_sum

    def dfs(t):
        if t == N_img:
            err_sum = partial_err()
            if err_sum < best["err_sum"]:
                best["err_sum"] = err_sum
                best["assign"] = dict(assign)
            return
        s = order[t]
        if len(cand_nodes[s]) == 0:
            return
        for g in cand_nodes[s]:
            if g in used:
                continue
            assign[s] = g
            used.add(g)
            err_now = partial_err()
            if err_now < best["err_sum"]:
                dfs(t + 1)
            used.remove(g)
            del assign[s]

    dfs(0)
    return best


def solve_global_consensus(
    per_pair_tables,
    df_gaia,
    max_per_pair=200,
    arcsec_err_cap=0.05,
    plot=True,
    invert_ra=True,
    label_brightest: int = 20,
    simbad_radius_arcsec: float = 1.0,
):
    """
    Consolida candidatos y encuentra la asignación global. Si `plot=True`,
    grafica con nombres de SIMBAD para las `label_brightest` estrellas más brillantes.
    """
    C, pairs_order, N_img = consolidate_candidates(
        per_pair_tables,
        max_per_pair=max_per_pair,
        arcsec_err_cap=arcsec_err_cap,
    )
    if not len(C) or N_img <= 1:
        print("[WARN] Candidatos insuficientes o N_img inválido.")
        return None, pd.DataFrame(), {}

    edge_by_pair, edge_err = build_edge_index(C)
    cand_nodes = candidate_nodes_per_img_star(edge_by_pair, N_img)
    best = search_assignment(edge_err, N_img, cand_nodes, pairs_order)
    if best["assign"] is None:
        print("[WARN] No se halló asignación completa.")
        return None, pd.DataFrame(), {}

    best_map = best["assign"]

    rows = []
    for (a, b) in pairs_order:
        gi, gj = best_map[a], best_map[b]
        mask = (
            (C["pair_tuple"] == (a, b))
            & (((C["i"] == gi) & (C["j"] == gj)) | ((C["i"] == gj) & (C["j"] == gi)))
        )
        sub = C[mask].sort_values("arcsec_err").head(1)
        if len(sub) == 0:
            rows.append({"pair": f"{a}-{b}", "i": gi, "j": gj, "arcsec_err": edge_err[(a, b)].get((gi, gj), np.inf)})
        else:
            r = sub.iloc[0].to_dict()
            r.update({"pair": f"{a}-{b}"})
            rows.append(r)
    best_edges_df = pd.DataFrame(rows)

    err_arr = best_edges_df["arcsec_err"].to_numpy()
    metrics = {
        "err_sum": float(np.sum(err_arr)),
        "err_median": float(np.median(err_arr)),
        "err_max": float(np.max(err_arr)),
        "n_pairs": len(err_arr),
        "n_img": N_img,
        "gaia_nodes": sorted(set(best_map.values())),
    }

    name_map: dict[int, str] = {}
    if plot and label_brightest and isinstance(df_gaia, pd.DataFrame) and len(df_gaia):
        try:
            name_map = resolve_bright_star_names_simbad(
                df_gaia,
                top_n=int(label_brightest),
                search_radius_arcsec=float(simbad_radius_arcsec),
                verbose=False,
            )
        except Exception as e:
            print(f"[WARN] SIMBAD: {e}")

    if plot and isinstance(df_gaia, pd.DataFrame) and len(df_gaia):
        dfp = df_gaia.copy()

        idxs = metrics["gaia_nodes"]
        if len(idxs) > 0:
            ra_center = _circular_mean_deg(dfp.iloc[idxs]["ra"].to_numpy())
        else:
            ra_center = _circular_mean_deg(dfp["ra"].to_numpy())

        dfp["ra_plot"] = unwrap_ra_for_plot(dfp["ra"].to_numpy(), center_deg=ra_center)

        if "phot_g_mean_mag" in dfp.columns:
            sizes_gaia = mag_to_size(dfp["phot_g_mean_mag"].to_numpy())
        else:
            sizes_gaia = np.full(len(dfp), 10.0, dtype=float)

        ra_sel = dfp.iloc[idxs]["ra_plot"].to_numpy()
        de_sel = dfp.iloc[idxs]["dec"].to_numpy()
        if "phot_g_mean_mag" in dfp.columns and len(idxs) > 0:
            g_sel = dfp.iloc[idxs]["phot_g_mean_mag"].to_numpy()
        else:
            g_sel = np.full(len(idxs), 12.0)
        sizes_sel = mag_to_size(g_sel) if len(idxs) else np.array([])

        fig = plt.figure(figsize=(9, 9))
        ax = plt.gca()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.scatter(
            dfp["ra_plot"],
            dfp["dec"],
            s=sizes_gaia * 0.06,
            alpha=0.3,
            color="white",
            label="Gaia",
        )
        if len(idxs):
            ax.scatter(
                ra_sel,
                de_sel,
                s=sizes_sel * 0.15,
                c="C1",
                marker="o",
                label="Nodos elegidos",
            )

        for idx_df, name in name_map.items():
            ra_b = float(dfp.at[idx_df, "ra_plot"])
            de_b = float(dfp.at[idx_df, "dec"])
            ax.text(ra_b, de_b, name, fontsize=9, color="C1", ha="left", va="bottom")

        if len(idxs):
            ra_min, ra_max = float(np.min(ra_sel)), float(np.max(ra_sel))
            de_min, de_max = float(np.min(de_sel)), float(np.max(de_sel))
        else:
            ra_min, ra_max = float(dfp["ra_plot"].min()), float(dfp["ra_plot"].max())
            de_min, de_max = float(dfp["dec"].min()), float(dfp["dec"].max())

        pad_ra = max(0.2, 0.15 * (ra_max - ra_min + 1e-6))
        pad_de = max(0.2, 0.15 * (de_max - de_min + 1e-6))
        rect_x0, rect_x1 = ra_min - pad_ra, ra_max + pad_ra
        rect_y0, rect_y1 = de_min - pad_de, de_max + pad_de

        ax.add_patch(
            plt.Rectangle(
                (rect_x0, rect_y0),
                (rect_x1 - rect_x0),
                (rect_y1 - rect_y0),
                fill=False,
                lw=1.4,
                ls="--",
                alpha=0.9,
                color="C1",
                label="Zona de zoom",
            )
        )

        if invert_ra:
            ax.invert_xaxis()

        ax.set_title(f"Campo completo | N={metrics['n_img']} | err_med={metrics['err_median']:.3f}\"", color="white")
        ax.set_xlabel("RA [deg] (continua)", color="white")
        ax.set_ylabel("Dec [deg]", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.grid(True, color="0.25")
        leg = ax.legend(facecolor="black", edgecolor="0.4")
        for t in leg.get_texts():
            t.set_color("white")

        sub = dfp[
            (dfp["ra_plot"] >= min(rect_x0, rect_x1))
            & (dfp["ra_plot"] <= max(rect_x0, rect_x1))
            & (dfp["dec"] >= min(rect_y0, rect_y1))
            & (dfp["dec"] <= max(rect_y0, rect_y1))
        ].copy()

        fig = plt.figure(figsize=(9, 9))
        ax = plt.gca()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        sizes_sub = mag_to_size(sub["phot_g_mean_mag"].to_numpy()) if "phot_g_mean_mag" in sub.columns else np.full(len(sub), 10.0)
        ax.scatter(sub["ra_plot"], sub["dec"], s=sizes_sub * 0.12, alpha=0.4, color="white", label="Gaia (zoom)")

        for _, r in best_edges_df.iterrows():
            gi, gj = int(r["i"]), int(r["j"])
            ra_i, de_i = float(dfp.iloc[gi]["ra_plot"]), float(dfp.iloc[gi]["dec"])
            ra_j, de_j = float(dfp.iloc[gj]["ra_plot"]), float(dfp.iloc[gj]["dec"])
            ax.plot([ra_i, ra_j], [de_i, de_j], lw=1.0, alpha=0.95, color="white")

        if len(idxs):
            ax.scatter(ra_sel, de_sel, s=sizes_sel * 0.22, c="C1", marker="o", label="Nodos elegidos")
            for img_idx, gaia_idx in best_map.items():
                ax.text(
                    float(dfp.iloc[gaia_idx]["ra_plot"]),
                    float(dfp.iloc[gaia_idx]["dec"]),
                    f"{img_idx}",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    color="C1",
                )

        for idx_df, name in name_map.items():
            ra_b = float(dfp.at[idx_df, "ra_plot"])
            de_b = float(dfp.at[idx_df, "dec"])
            if (min(rect_x0, rect_x1) <= ra_b <= max(rect_x0, rect_x1)) and (min(rect_y0, rect_y1) <= de_b <= max(rect_y0, rect_y1)):
                ax.text(ra_b, de_b, name, fontsize=9, color="C1", ha="left", va="bottom")

        if invert_ra:
            ax.invert_xaxis()

        ax.set_title("Zoom de la solución — segmentos, nodos y nombres SIMBAD", color="white")
        ax.set_xlabel("RA [deg] (continua)", color="white")
        ax.set_ylabel("Dec [deg]", color="white")
        ax.set_xlim(min(rect_x0, rect_x1), max(rect_x0, rect_x1))
        ax.set_ylim(min(rect_y0, rect_y1), max(rect_y0, rect_y1))
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.grid(True, color="0.25")
        leg = ax.legend(facecolor="black", edgecolor="0.4")
        for t in leg.get_texts():
            t.set_color("white")

        plt.figure(figsize=(9, 4))
        order = np.argsort(best_edges_df["arcsec_err"].to_numpy())
        plt.bar(np.arange(len(order)), best_edges_df.iloc[order]["arcsec_err"].to_numpy())
        plt.xticks(np.arange(len(order)), best_edges_df.iloc[order]["pair"].to_list(), rotation=45)
        plt.ylabel("error absoluto (arcsec)")
        plt.title("Error absoluto por par (asignación elegida)")
        plt.tight_layout()
        plt.show()

    return best_map, best_edges_df, metrics
