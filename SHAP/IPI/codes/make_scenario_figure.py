import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

BLUE = "#0F4D92"; GREEN_BAND = "#DDF3DE"; GREEN_LINE = "#3E9E52"
GOLD = "#E8A400"; RED = "#B64342"; NMID = "#767676"; NDARK = "#4D4D4D"; NBLACK = "#272727"

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "TabPFN" / "slider_data.json"
d = json.loads(DATA.read_text(encoding="utf-8"))
order = ["E01015473", "E01033845", "E01033861", "E01015569", "E01015524"]


def is_pct(feat):
    return "share" in feat


def nearest_cross(grid, gva, cur, target):
    xs = np.linspace(grid[0], grid[-1], 500)
    ys = np.interp(xs, grid, gva)
    idx = np.where(ys >= target)[0]
    if len(idx) == 0:
        return None
    return xs[idx[np.argmin(np.abs(xs[idx] - cur))]]


def fmt_v(v, feat):
    return f"{v*100:.0f}%" if is_pct(feat) else f"{v:.2f}"


def panel(ax, code, rank, hero=False):
    e = d[code]
    grid = np.array(e["grid"]); gva = np.array(e["gva"])
    cur, feat, lab = e["current_value"], e["feature"], e["label"]
    base = float(np.interp(cur, grid, gva)); target = base * 1.3
    xc = nearest_cross(grid, gva, cur, target)

    ax.axhspan(base, target, color=GREEN_BAND, zorder=0)
    ax.axhline(target, color=GREEN_LINE, ls="--", lw=1.0, zorder=1)
    ax.axvline(cur, color=NMID, ls=":", lw=0.9, zorder=1)
    ax.plot(grid, gva, color=BLUE, lw=1.9, marker="o", ms=2.6, mfc=BLUE,
            mec="white", mew=0.3, zorder=3, clip_on=True)
    ax.plot(cur, base, "o", ms=7, mfc=GOLD, mec=NBLACK, mew=0.8, zorder=5)

    ymin = min(gva.min(), base); ymax = max(gva.max(), target)
    pad = (ymax - ymin) * 0.12 + 1e-9
    ax.set_ylim(ymin - pad, ymax + pad * 1.4)
    ax.set_xlim(grid[0], grid[-1])
    if is_pct(feat):
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.tick_params(labelsize=7, length=3)

    if xc is not None:
        ax.plot(xc, target, marker="v", ms=6, mfc=RED, mec="white", mew=0.5, zorder=6)
        direc = "higher" if xc > cur else "lower"
        verdict = f"reaches +30% at {fmt_v(xc, feat)} ({direc})"
        vcol = NDARK
    else:
        verdict = "+30% beyond observed range"
        vcol = RED

    tfs = 9 if hero else 7.8
    ty, vy = (1.045, 1.018) if hero else (1.135, 1.03)
    ax.text(0, ty, f"#{rank}   {lab}", transform=ax.transAxes,
            fontsize=tfs, fontweight="bold", color=NBLACK, va="bottom")
    ax.text(0, vy, verdict, transform=ax.transAxes,
            fontsize=7 if hero else 6.5, color=vcol, va="bottom")
    ax.set_ylabel("predicted total GVA (£m)", fontsize=7.2)
    ax.set_xlabel(lab + (" (share of workers)" if is_pct(feat) else " (index)"), fontsize=7.2)

    lo, hi = ymin - pad, ymax + pad * 1.4
    dy, va = (11, "bottom") if (base - lo) / (hi - lo) < 0.30 else (-3, "top")
    ax.annotate(f"now £{base:.0f}m", xy=(cur, base), xytext=(8, dy),
                textcoords="offset points", fontsize=6.4, color=NDARK, va=va, ha="left")
    if hero:
        need = base - e["actual_gva"]
        ax.text(0.03, 0.97,
                "The top-priority area's bottleneck is out-commuting — yet\n"
                "higher out-commute co-occurs with higher GVA (affluent\n"
                "commuter belt). Read as association, not a lever to pull.",
                transform=ax.transAxes, fontsize=6.6, color=NDARK, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F7F3E8", ec="#E0D8C0", lw=0.7))
        ax.text(0.03, 0.55, f"unrealised potential\n(model − observed) = £{need:.0f}m",
                transform=ax.transAxes, fontsize=6.6, color=BLUE, va="top")


fig = plt.figure(figsize=(7.6, 4.9))
gs = GridSpec(2, 3, width_ratios=[1.42, 1, 1], height_ratios=[1, 1],
              hspace=0.92, wspace=0.44, left=0.07, right=0.985, top=0.775, bottom=0.12)
panel(fig.add_subplot(gs[:, 0]), order[0], 1, hero=True)
panel(fig.add_subplot(gs[0, 1]), order[1], 2)
panel(fig.add_subplot(gs[0, 2]), order[2], 3)
panel(fig.add_subplot(gs[1, 1]), order[3], 4)
panel(fig.add_subplot(gs[1, 2]), order[4], 5)

fig.text(0.07, 0.965, "Scenario simulation — can each priority area's bottleneck lever reach Swindon's +30% GVA ambition?",
         fontsize=11.5, fontweight="bold", color=NBLACK, ha="left")
fig.text(0.07, 0.925,
         "TabPFN model-implied response as each LSOA's SHAP-identified bottleneck varies over its observed Swindon range.  "
         "Associational, not causal — pilot before acting.",
         fontsize=7.6, color=NMID, ha="left")

key = [
    Line2D([0], [0], color=BLUE, lw=1.9, marker="o", ms=3, mfc=BLUE, mec="white", label="model response"),
    Line2D([0], [0], marker="o", color="none", mfc=GOLD, mec=NBLACK, ms=7, label="current level"),
    Patch(fc=GREEN_BAND, ec=GREEN_LINE, label="current → +30% zone"),
    Line2D([0], [0], color=GREEN_LINE, ls="--", lw=1, label="+30% target"),
    Line2D([0], [0], marker="v", color="none", mfc=RED, mec="white", ms=7, label="lever hits +30%"),
]
fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.07, 0.895),
           ncol=5, fontsize=6.8, handlelength=1.5, columnspacing=1.3,
           handletextpad=0.5, borderaxespad=0)

fig.text(0.07, 0.025,
         "The +30% line mirrors Swindon's 2036 borough ambition for reference only; it is not a per-area, single-lever causal target.  "
         "Levers = per-area bottleneck (most-negative intervenable SHAP feature).  n = 5 top-IPI LSOAs "
         "(#1 E01015473, #2 E01033845, #3 E01033861, #4 E01015569, #5 E01015524).",
         fontsize=6.2, color=NMID, ha="left")

fig.savefig(HERE / "ipi_scenario_lever.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("saved ipi_scenario_lever.png ->", HERE)
