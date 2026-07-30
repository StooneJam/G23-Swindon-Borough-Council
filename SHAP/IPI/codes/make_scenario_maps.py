import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, ConnectionPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8.5, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

BLUE = "#0F4D92"; GREEN_BAND = "#DDF3DE"; GREEN_LINE = "#3E9E52"
GOLD = "#E8A400"; RED = "#B64342"; NMID = "#767676"; NDARK = "#4D4D4D"
NBLACK = "#272727"; MAP_GREY = "#E4E4E4"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SLIDER = HERE.parent / "TabPFN" / "slider_data.json"
GEO = ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
SWCSV = ROOT / "commuting-regression" / "data_swindon_with_commute.csv"

order = ["E01015473", "E01033845", "E01033861", "E01015569", "E01015524"]
NOTES = {
    "E01015473": "Higher out-commute co-occurs with higher GVA (affluent commuter belt). "
                 "Read this as association, not a lever the council should pull.",
}


def is_pct(feat):
    return "share" in feat


def fmt_v(v, feat):
    return f"{v*100:.0f}%" if is_pct(feat) else f"{v:.2f}"


def nearest_cross(grid, gva, cur, target):
    xs = np.linspace(grid[0], grid[-1], 500)
    ys = np.interp(xs, grid, gva)
    idx = np.where(ys >= target)[0]
    if len(idx) == 0:
        return None
    return xs[idx[np.argmin(np.abs(xs[idx] - cur))]]


def draw_curve(ax, e):
    grid = np.array(e["grid"]); gva = np.array(e["gva"])
    cur, feat, lab = e["current_value"], e["feature"], e["label"]
    base = float(np.interp(cur, grid, gva)); target = base * 1.3
    xc = nearest_cross(grid, gva, cur, target)

    ax.axhspan(base, target, color=GREEN_BAND, zorder=0)
    ax.axhline(target, color=GREEN_LINE, ls="--", lw=1.1, zorder=1)
    ax.axvline(cur, color=NMID, ls=":", lw=0.9, zorder=1)
    ax.plot(grid, gva, color=BLUE, lw=2.0, marker="o", ms=3, mfc=BLUE,
            mec="white", mew=0.4, zorder=3)
    ax.plot(cur, base, "o", ms=8, mfc=GOLD, mec=NBLACK, mew=0.9, zorder=5)

    ymin = min(gva.min(), base); ymax = max(gva.max(), target)
    pad = (ymax - ymin) * 0.12 + 1e-9
    lo, hi = ymin - pad, ymax + pad * 1.4
    ax.set_ylim(lo, hi); ax.set_xlim(grid[0], grid[-1])
    if is_pct(feat):
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.tick_params(labelsize=8, length=3)

    if xc is not None:
        ax.plot(xc, target, marker="v", ms=8, mfc=RED, mec="white", mew=0.6, zorder=6)
        direc = "higher" if xc > cur else "lower"
        verdict = f"reaches +30% at {fmt_v(xc, feat)} ({direc})"; vcol = NDARK
    else:
        verdict = "+30% beyond the observed range"; vcol = RED

    ax.text(0.99, target, "+30% target", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=GREEN_LINE, va="bottom", ha="right")
    ax.set_ylabel("predicted total GVA (£m)", fontsize=8.5)
    ax.set_xlabel(lab + (" (share of workers)" if is_pct(feat) else " (index)"), fontsize=8.5)
    return verdict, vcol, base


def build(gdf_sw, e, code, rank, out):
    fig = plt.figure(figsize=(8.6, 4.4))
    gs = GridSpec(1, 2, width_ratios=[1, 1.32], wspace=0.02,
                  left=0.02, right=0.965, top=0.85, bottom=0.135)
    mapax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    gdf_sw.plot(ax=mapax, color=MAP_GREY, edgecolor="white", linewidth=0.35)
    tgt = gdf_sw[gdf_sw["LSOA21CD"] == code]
    tgt.plot(ax=mapax, color=BLUE, edgecolor=NBLACK, linewidth=0.6, zorder=4)
    mapax.set_axis_off(); mapax.set_aspect("equal")
    mapax.set_title("this area within Swindon", fontsize=8.5, color=NDARK, pad=4)

    verdict, vcol, base = draw_curve(cax, e)
    cax.text(0, 1.03, f"now £{base:.0f}m", transform=cax.transAxes,
             fontsize=9, fontweight="bold", color=NBLACK, va="bottom", ha="left")
    cax.text(1.0, 1.03, verdict, transform=cax.transAxes,
             fontsize=9, color=vcol, va="bottom", ha="right")

    cen = tgt.geometry.representative_point().iloc[0]
    con = ConnectionPatch(xyA=(cen.x, cen.y), coordsA=mapax.transData,
                          xyB=(0, 0.5), coordsB=cax.transAxes,
                          arrowstyle="-|>", color=BLUE, lw=1.4, zorder=20,
                          mutation_scale=14)
    fig.add_artist(con)

    nm = str(tgt["LSOA21NM"].iloc[0])
    fig.text(0.02, 0.945, f"#{rank}  {code}  {nm}    |    lever: {e['label']}", fontsize=12,
             fontweight="bold", color=NBLACK, ha="left")

    key = [
        Line2D([0], [0], color=BLUE, lw=2, marker="o", ms=3.5, mfc=BLUE, mec="white", label="model response"),
        Line2D([0], [0], marker="o", color="none", mfc=GOLD, mec=NBLACK, ms=8, label="current level"),
        Patch(fc=GREEN_BAND, ec=GREEN_LINE, label="current to +30% zone"),
        Line2D([0], [0], color=GREEN_LINE, ls="--", lw=1.1, label="+30% target"),
        Line2D([0], [0], marker="v", color="none", mfc=RED, mec="white", ms=8, label="lever hits +30%"),
    ]
    fig.legend(handles=key, loc="lower center", bbox_to_anchor=(0.55, 0.005),
               ncol=5, fontsize=7.5, handlelength=1.6, columnspacing=1.5, handletextpad=0.5)

    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    d = json.loads(SLIDER.read_text(encoding="utf-8"))
    sw_codes = set(pd.read_csv(SWCSV)["LSOA21CD"])
    print("loading boundaries ...", flush=True)
    gdf = gpd.read_file(GEO)
    gdf_sw = gdf[gdf["LSOA21CD"].isin(sw_codes)].to_crs(27700).reset_index(drop=True)
    print(f"Swindon polygons: {len(gdf_sw)}", flush=True)
    for rank, code in enumerate(order, 1):
        out = HERE / f"scenario_{rank}_{code}"
        build(gdf_sw, d[code], code, rank, out)
        print("saved", out.name, flush=True)


if __name__ == "__main__":
    main()
