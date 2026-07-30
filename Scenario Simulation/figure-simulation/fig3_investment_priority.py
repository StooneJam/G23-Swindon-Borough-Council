"""Figure 3: LSOA-wide scenario ranking and equal-budget strategies."""
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parent
AAS = SIM / "all areas simulation"
sys.path.insert(0, str(SIM))
if not (AAS / "all_zone_ranking.csv").exists():
    from lsoa_scenario_core import build_all_outputs
    build_all_outputs()

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
INK, BLUE, ORANGE, GREEN, GREY = "#26323C", "#436F99", "#C66A3D", "#3C8D82", "#B9C2CB"


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=600, bbox_inches="tight")


ranking = pd.read_csv(AAS / "all_zone_ranking.csv")
strategies = pd.read_csv(AAS / "optimisation_result.csv")
boundaries = gpd.read_file(
    ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
)
poly = boundaries[boundaries["LSOA21CD"].isin(ranking["LSOA21CD"])].copy()
poly = poly.merge(
    ranking[["LSOA21CD", "rank", "allocation_pct", "label"]],
    on="LSOA21CD",
).to_crs(27700)

fig = plt.figure(figsize=(7.08, 4.7))
gs = GridSpec(2, 2, width_ratios=[1.24, 1], height_ratios=[1, 1],
              wspace=0.31, hspace=0.56)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# a: all-zone return map
norm = TwoSlopeNorm(
    vmin=poly["allocation_pct"].min(), vcenter=0,
    vmax=poly["allocation_pct"].max(),
)
poly.plot(column="allocation_pct", cmap="PuOr", norm=norm, ax=ax_a,
          edgecolor="white", linewidth=0.35)
top5 = set(ranking.head(5)["LSOA21CD"])
top3 = set(ranking.head(3)["LSOA21CD"])
halo = [pe.withStroke(linewidth=2.0, foreground="white")]
for _, row in poly[poly["LSOA21CD"].isin(top3)].iterrows():
    point = row.geometry.representative_point()
    ax_a.annotate(
        f"#{int(row['rank'])}", (point.x, point.y),
        ha="center", va="center", fontsize=6.5, fontweight="bold",
        color=INK, path_effects=halo,
    )
poly[poly["LSOA21CD"].isin(top5)].boundary.plot(
    ax=ax_a, color=GREEN, linewidth=1.3
)
ax_a.axis("off")
ax_a.set_title("a  Return to the same +30% intervention",
               loc="left", fontsize=8, fontweight="bold")
mapper = plt.cm.ScalarMappable(cmap="PuOr", norm=norm)
cbar = fig.colorbar(mapper, ax=ax_a, shrink=0.46, pad=0.01, aspect=17)
cbar.set_label("GVA allocation effect (% of Swindon total)", fontsize=6.1)
cbar.ax.tick_params(labelsize=6, length=2)

# b: show both opportunities and risks
extremes = pd.concat([ranking.head(5), ranking.tail(5)]).drop_duplicates()
extremes = extremes.sort_values("allocation_pct")
y = np.arange(len(extremes))
bar_colors = [GREEN if value >= 0 else ORANGE
              for value in extremes["allocation_pct"]]
labels = [label.replace("Swindon ", "") for label in extremes["label"]]
ax_b.barh(y, extremes["allocation_pct"], color=bar_colors, height=0.70,
          edgecolor="white", linewidth=0.4)
ax_b.axvline(0, color=INK, lw=0.8)
ax_b.set_yticks(y, labels, fontsize=5.9)
for yy, value in zip(y, extremes["allocation_pct"]):
    if value >= 0:
        ax_b.text(value + 0.035, yy, f"{value:+.2f}%", va="center",
                  ha="left", fontsize=5.6, color=INK)
    else:
        ax_b.text(value / 2, yy, f"{value:+.2f}%", va="center",
                  ha="center", fontsize=4.9, color="white", fontweight="bold")
ax_b.set_xlabel("GVA allocation effect (%)", fontsize=6.6)
ax_b.set_title("b  Highest and lowest-return LSOAs",
               loc="left", fontsize=8, fontweight="bold")
ax_b.tick_params(length=2)

# c: equal total attractiveness budget (+0.60)
name_map = {
    "LSOA-optimised (top 2 returns)": "Top 2\nreturns",
    "2 top adjusted-productivity LSOAs": "Top 2\nproductivity",
    "uniform across 137 LSOAs": "Uniform\n(137)",
    "2 biggest workplace hubs": "2 biggest\nhubs",
}
order = list(name_map)
plot_strategies = strategies.set_index("strategy").loc[order].reset_index()
values = plot_strategies["dGVA_allocation_pct"].to_numpy(float)
x = np.arange(len(values))
colors = [GREEN if value > 0.01 else (ORANGE if value < -0.01 else GREY)
          for value in values]
ax_c.bar(x, values, color=colors, width=0.66, edgecolor="white", linewidth=0.5)
ax_c.axhline(0, color=INK, lw=0.8)
for xx, value in zip(x, values):
    label = "0.00%" if abs(value) < 0.005 else f"{value:+.2f}%"
    ax_c.text(
        xx, value + (0.07 if value >= 0 else -0.07), label,
        ha="center", va="bottom" if value >= 0 else "top",
        fontsize=5.9, fontweight="bold", color=INK,
    )
ax_c.set_xticks(x, [name_map[name] for name in order], fontsize=5.8)
ax_c.set_ylabel("GVA allocation effect (%)", fontsize=6.5)
ax_c.set_ylim(values.min() * 1.45 - 0.12, values.max() * 1.20 + 0.12)
ax_c.set_title("c  Equal-budget allocation strategies",
               loc="left", fontsize=8, fontweight="bold")
ax_c.tick_params(length=2)

fig.suptitle(
    "LSOA-level targeting materially changes investment returns",
    x=0.075, y=0.985, ha="left", fontsize=9.2, fontweight="bold",
)
save(fig, "fig3_investment_priority")
print(
    f"saved fig3 | top={ranking.iloc[0]['label']} "
    f"{ranking.iloc[0]['allocation_pct']:+.2f}% | "
    f"top-2={values[0]:+.2f}% | biggest hubs={values[-1]:+.2f}%"
)
