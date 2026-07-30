"""
IPI (Intervention Priority Index) visualisation for Swindon.

Driven by the ranked index file `SHAP/TabPFN/ipi_tabpfn.csv` (produced by the
TabPFN IPI pipeline: ipi_shap.py -> ipi_build.py). This script consolidates the
four things the council asked for into one folder (SHAP/IPI/):

    1. Top-5 priority areas            -> 01_top5_ranking.png   (+ ipi_top5.csv)
    2. Map of Swindon by IPI           -> 02_ipi_map.png
    3. Each area's bottleneck lever     -> shown on (1) and (3)
    4. "Investment" to reach +30% GVA   -> 03_investment_top5.png (+ ipi_top5_investment.csv)

On (4): there is NO pound-cost model in this project, so "investment" is
expressed the only honest way the model supports — how far the area's bottleneck
lever must move (within Swindon's observed range) for model-predicted GVA to
reach current x 1.30 (the +30% 2036 ambition). This is ASSOCIATIONAL, not a
guaranteed return on spend, mirroring the disclaimer in SHAP/TabPFN/app.py.

The lever response curves are generated with the same TabPFN model that produced
ipi_tabpfn.csv, so the whole figure set is internally consistent with that file.

Run:  python SHAP/IPI/ipi_plots.py
Env:  N_TOP (default 5), GRID (curve resolution, default 25).
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import geopandas as gpd

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths / config                                                               #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IPI_CSV = ROOT / "SHAP" / "TabPFN" / "ipi_tabpfn.csv"
DATA_CSV = ROOT / "commuting-regression" / "data_swindon_with_commute.csv"
GEO = ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"

N_TOP = int(os.environ.get("N_TOP", "5"))
GRID = int(os.environ.get("GRID", "25"))
UPLIFT = 1.30  # +30% ambition

TARGET = "log_total_GVA_2023"
ENG = [
    "log_voa_rv_2023", "rv_per_working_age", "sme_density", "qualification_index",
    "firm_size_diversity", "rv_per_employee", "sme_qual_interaction",
    "employment_quality", "modern_sector_leverage", "asset_growth_diversity",
]
COM = [
    "msoa_out_commute_share", "msoa_same_msoa_work_share", "msoa_wfh_share",
    "msoa_workplace_commuters", "msoa_total_employed", "msoa_in_commute_share",
    "msoa_local_worker_share", "msoa_workers_at_workplace", "msoa_inbound_worker_count",
]
FEATS = ENG + COM

LABELS = {
    "log_voa_rv_2023": "log VOA rateable value", "rv_per_working_age": "RV per working-age adult",
    "sme_density": "SME density", "qualification_index": "Qualification index",
    "firm_size_diversity": "Firm-size diversity", "rv_per_employee": "RV per employee",
    "sme_qual_interaction": "SME x qualification", "employment_quality": "Employment quality",
    "modern_sector_leverage": "Modern-sector leverage", "asset_growth_diversity": "Asset-growth diversity",
    "msoa_out_commute_share": "Out-commute share", "msoa_same_msoa_work_share": "Same-MSOA work share",
    "msoa_wfh_share": "Work-from-home share", "msoa_workplace_commuters": "Workplace commuters",
    "msoa_total_employed": "Total employed", "msoa_in_commute_share": "In-commute share",
    "msoa_local_worker_share": "Local worker share", "msoa_workers_at_workplace": "Workers at workplace",
    "msoa_inbound_worker_count": "Inbound workers",
}

# Areas whose bottleneck is an association, not a council lever (carried through
# from SHAP/figures/make_scenario_maps.py).
CAVEAT = {
    "E01015473": "Higher out-commute co-occurs with higher GVA (affluent commuter belt) — "
                 "read as association, not a lever the council should pull.",
}

# --------------------------------------------------------------------------- #
# Aesthetics                                                                   #
# --------------------------------------------------------------------------- #
NAVY, CORAL, TEAL, GOLD = "#1f3b57", "#ff5a5f", "#2ec4b6", "#f2a900"
GREEN_BAND, GREEN_LINE, RED, GREY = "#DDF3DE", "#3E9E52", "#B64342", "#8a97a5"
MAP_GREY, NBLACK, NDARK, NMID = "#E4E4E4", "#272727", "#4D4D4D", "#767676"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#c9d1d9", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#eceff3", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": "#43515e", "ytick.color": "#43515e",
    "text.color": "#22303c", "axes.labelcolor": "#22303c", "figure.facecolor": "white",
})


def is_pct(feat: str) -> bool:
    return "share" in feat


def fmt_v(v: float, feat: str) -> str:
    return f"{v*100:.0f}%" if is_pct(feat) else f"{v:.2f}"


# --------------------------------------------------------------------------- #
# Model + lever curves (same recipe as ipi_slider_data.py / app.py)            #
# --------------------------------------------------------------------------- #
def fit_model(df: pd.DataFrame) -> TabPFNRegressor:
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
    reg.fit(df[FEATS].to_numpy(float), df[TARGET].to_numpy(float))
    return reg


def lever_curve(df, reg, code, feat):
    """ICE-style response of predicted GVA (£m) as one area's bottleneck lever
    sweeps Swindon's observed range, holding its other features fixed."""
    i = int(df.index[df["LSOA21CD"] == code][0])
    X = df[FEATS].to_numpy(float)
    j = FEATS.index(feat)
    lo, hi = float(df[feat].min()), float(df[feat].max())
    grid = np.linspace(lo, hi, GRID)
    rows = np.repeat(X[i:i + 1], GRID, axis=0)
    rows[:, j] = grid
    gva = np.exp(reg.predict(rows))
    cur = float(X[i, j])
    actual = float(np.exp(df[TARGET].to_numpy(float)[i]))
    return dict(grid=grid, gva=gva, cur=cur, actual=actual, lo=lo, hi=hi, feat=feat)


def nearest_cross(grid, gva, cur, target):
    xs = np.linspace(grid[0], grid[-1], 500)
    ys = np.interp(xs, grid, gva)
    idx = np.where(ys >= target)[0]
    if len(idx) == 0:
        return None
    return float(xs[idx[np.argmin(np.abs(xs[idx] - cur))]])


# --------------------------------------------------------------------------- #
# Figure 1 — top-N ranking + bottleneck                                        #
# --------------------------------------------------------------------------- #
def fig_top_ranking(top, path):
    ranks = top["rank"].to_numpy()
    ipi = top["IPI"].to_numpy()
    labels = [f"#{r}  {c}" for r, c in zip(ranks, top["LSOA21CD"])]
    bott = [LABELS.get(b, b) for b in top["bottleneck"]]
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(top)))[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(top) + 1.6))
    y = np.arange(len(top))[::-1]
    bars = ax.barh(y, ipi, color=colors, edgecolor="white", linewidth=0.8)
    for yi, v, b, gap in zip(y, ipi, bott, top["gva_unrealised_m"]):
        ax.text(v + 0.012, yi, f"IPI {v:.2f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=NBLACK)
        ax.text(0.012, yi, f"lever: {b}   ·   unrealised GVA = £{gap:.0f}m",
                va="center", ha="left", fontsize=9.5, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, max(ipi) * 1.22)
    ax.set_xlabel("Intervention Priority Index  (rank-need × rank-leverable-SHAP)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — choropleth map by IPI, top-N highlighted                          #
# --------------------------------------------------------------------------- #
def fig_ipi_map(gdf_sw, ipi_df, top, path):
    g = gdf_sw.merge(ipi_df[["LSOA21CD", "IPI", "rank"]], on="LSOA21CD", how="left")

    fig, ax = plt.subplots(figsize=(9.2, 9.4))
    # Every LSOA gets a clear mid-grey boundary so low-IPI (pale) areas stay
    # visually distinct instead of merging into one another / the background.
    g.plot(ax=ax, column="IPI", cmap="magma_r", edgecolor="#4d4d4d", linewidth=0.6,
           legend=True, legend_kwds={"label": "Intervention Priority Index (0–1)",
                                     "shrink": 0.5, "orientation": "vertical"})
    top_codes = set(top["LSOA21CD"])
    tg = g[g["LSOA21CD"].isin(top_codes)]
    tg.plot(ax=ax, facecolor="none", edgecolor=TEAL, linewidth=2.4, zorder=5)

    # Leader lines: label each top area with a rank badge + LSOA code, placed
    # around the map margin and connected back to the polygon with a thin line.
    minx, miny, maxx, maxy = gdf_sw.total_bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    R = max(maxx - minx, maxy - miny)
    for _, row in tg.sort_values("rank").iterrows():
        pt = row.geometry.representative_point()
        ang = np.arctan2(pt.y - cy, pt.x - cx)
        lx, ly = cx + np.cos(ang) * R * 0.66, cy + np.sin(ang) * R * 0.66
        ha = "left" if lx >= cx else "right"
        ax.annotate(
            f"#{int(row['rank'])}  {row['LSOA21CD']}", xy=(pt.x, pt.y), xytext=(lx, ly),
            ha=ha, va="center", fontsize=10.5, fontweight="bold", color=NBLACK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NAVY, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.2,
                            shrinkA=4, shrinkB=2, connectionstyle="arc3,rad=0.0"),
            annotation_clip=False, zorder=7)
        ax.plot(pt.x, pt.y, "o", ms=4, mfc=NAVY, mec="white", mew=0.8, zorder=6)

    pad = R * 0.30
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — "+30% investment" lever curves for the top-N                      #
# --------------------------------------------------------------------------- #
def fig_investment(curves, top, path):
    n = len(curves)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, top.iterrows()):
        code = r["LSOA21CD"]
        c = curves[code]
        grid, gva, cur, feat = c["grid"], c["gva"], c["cur"], c["feat"]
        base = float(np.interp(cur, grid, gva))
        target = base * UPLIFT
        xc = nearest_cross(grid, gva, cur, target)

        ax.axhspan(base, target, color=GREEN_BAND, zorder=0)
        ax.axhline(target, color=GREEN_LINE, ls="--", lw=1.1, zorder=1)
        ax.axvline(cur, color=NMID, ls=":", lw=0.9, zorder=1)
        ax.plot(grid, gva, color=NAVY, lw=2.0, marker="o", ms=3, mfc=NAVY,
                mec="white", mew=0.4, zorder=3)
        ax.plot(cur, base, "o", ms=8, mfc=GOLD, mec=NBLACK, mew=0.9, zorder=5)

        if xc is not None:
            ax.plot(xc, target, marker="v", ms=9, mfc=RED, mec="white", mew=0.6, zorder=6)
            direc = "↑" if xc > cur else "↓"
            verdict = f"+30% at {fmt_v(xc, feat)} ({direc} from {fmt_v(cur, feat)})"
            vcol = NDARK
        else:
            verdict = "+30% beyond observed range"
            vcol = RED

        if is_pct(feat):
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title(f"#{int(r['rank'])}  {code}", loc="left", fontsize=11.5)
        ax.set_xlabel(LABELS.get(feat, feat) + (" (share)" if is_pct(feat) else " (index)"),
                      fontsize=9.5)
        ax.set_ylabel("pred. GVA (£m)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.text(0.5, 1.15, f"now £{base:.0f}m → target £{target:.0f}m",
                transform=ax.transAxes, ha="center", fontsize=9, color=NBLACK, fontweight="bold")
        ax.text(0.5, -0.30, verdict, transform=ax.transAxes, ha="center",
                fontsize=8.8, color=vcol)

    for ax in axes[n:]:
        ax.set_axis_off()

    key = [
        Line2D([0], [0], color=NAVY, lw=2, marker="o", ms=3.5, mfc=NAVY, mec="white", label="model response"),
        Line2D([0], [0], marker="o", color="none", mfc=GOLD, mec=NBLACK, ms=8, label="current level"),
        Patch(fc=GREEN_BAND, ec=GREEN_LINE, label="current → +30% zone"),
        Line2D([0], [0], marker="v", color="none", mfc=RED, mec="white", ms=8, label="lever reaches +30%"),
    ]
    fig.legend(handles=key, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ipi_df = pd.read_csv(IPI_CSV)
    ipi_df.to_csv(HERE / "ipi_index_all_areas.csv", index=False)  # per-region index (137)

    top = ipi_df[ipi_df["IPI"] > 0].sort_values("rank").head(N_TOP).reset_index(drop=True)
    print(f"Top-{N_TOP} areas:\n",
          top[["rank", "LSOA21CD", "IPI", "bottleneck"]].to_string(index=False), flush=True)

    df = pd.read_csv(DATA_CSV)
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    df = df.reset_index(drop=True)

    print("fitting TabPFN + building lever curves ...", flush=True)
    reg = fit_model(df)
    curves = {r["LSOA21CD"]: lever_curve(df, reg, r["LSOA21CD"], r["bottleneck"])
              for _, r in top.iterrows()}
    # cache the curves so the figures are reproducible without a refit
    (HERE / "slider_data.json").write_text(json.dumps(
        {k: {"feature": v["feat"], "grid": v["grid"].tolist(), "gva": v["gva"].tolist(),
             "current_value": v["cur"], "actual_gva": v["actual"],
             "vmin": v["lo"], "vmax": v["hi"]} for k, v in curves.items()}, indent=1))

    # ---- investment summary table -------------------------------------------- #
    # NOTE: the `need` column in ipi_tabpfn.csv is a LOG-GVA gap (rank input for
    # IPI), NOT pounds. The £ unrealised potential is model-now minus observed in
    # exp space (as app.py reports it) — kept as a separate, correctly-named field.
    rows = []
    for _, r in top.iterrows():
        code = r["LSOA21CD"]
        c = curves[code]
        base = float(np.interp(c["cur"], c["grid"], c["gva"]))   # model GVA now (£m)
        target = base * UPLIFT
        gap = base - c["actual"]                                 # £m unrealised (model vs observed)
        xc = nearest_cross(c["grid"], c["gva"], c["cur"], target)
        rows.append({
            "rank": int(r["rank"]), "LSOA21CD": code, "MSOA21CD": r["MSOA21CD"],
            "IPI": round(float(r["IPI"]), 4), "need_score_logscale": round(float(r["need"]), 3),
            "bottleneck": r["bottleneck"], "bottleneck_label": LABELS.get(r["bottleneck"], r["bottleneck"]),
            "gva_observed_m": round(c["actual"], 1), "gva_now_model_m": round(base, 1),
            "gva_unrealised_m": round(gap, 1), "gva_target_+30pct_m": round(target, 1),
            "lever_now": round(c["cur"], 4),
            "lever_at_+30pct": None if xc is None else round(xc, 4),
            "lever_change": None if xc is None else round(xc - c["cur"], 4),
            "reachable_within_observed_range": xc is not None,
            "caveat": CAVEAT.get(code, ""),
        })
    inv = pd.DataFrame(rows)
    inv.to_csv(HERE / "ipi_top5_investment.csv", index=False)
    # carry the £ unrealised gap onto `top` so the ranking figure can label it
    top = top.merge(inv[["LSOA21CD", "gva_unrealised_m"]], on="LSOA21CD", how="left")
    top.to_csv(HERE / "ipi_top5.csv", index=False)

    # ---- geometry ------------------------------------------------------------ #
    print("loading boundaries ...", flush=True)
    gdf = gpd.read_file(GEO)
    gdf_sw = gdf[gdf["LSOA21CD"].isin(set(df["LSOA21CD"]))].to_crs(27700).reset_index(drop=True)

    # ---- figures ------------------------------------------------------------- #
    fig_top_ranking(top, HERE / "01_top5_ranking.png")
    fig_ipi_map(gdf_sw, ipi_df, top, HERE / "02_ipi_map.png")
    fig_investment(curves, top, HERE / "03_investment_top5.png")

    print("\nInvestment summary (lever movement to +30% GVA):")
    show = inv[["rank", "LSOA21CD", "bottleneck_label", "gva_now_model_m",
                "gva_target_+30pct_m", "lever_now", "lever_at_+30pct",
                "reachable_within_observed_range"]]
    print(show.to_string(index=False))
    print("\nDone. Outputs ->", HERE)


if __name__ == "__main__":
    main()
