"""Figure 2: LSOA employment sensitivity and scenario contrast."""
from pathlib import Path
import json
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parent
sys.path.insert(0, str(SIM))
if not (SIM / "scenario_lsoa_contrast.csv").exists():
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


elasticity = pd.read_csv(SIM / "lsoa_local_elasticity.csv")
ranking = pd.read_csv(SIM / "all areas simulation" / "all_zone_ranking.csv")
contrast = pd.read_csv(SIM / "scenario_lsoa_contrast.csv")
diagnostics = json.loads((SIM / "lsoa_model_diagnostics.json").read_text(encoding="utf-8"))

boundaries = gpd.read_file(
    ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
)
poly = boundaries[boundaries["LSOA21CD"].isin(elasticity["LSOA21CD"])].copy()
poly = poly.merge(
    elasticity[["LSOA21CD", "elasticity"]], on="LSOA21CD"
).to_crs(27700)

largest_code = contrast.iloc[0]["LSOA21CD"]
top_code = contrast.iloc[1]["LSOA21CD"]
highlight = {
    largest_code: contrast.iloc[0]["label"].replace("Swindon ", ""),
    top_code: contrast.iloc[1]["label"].replace("Swindon ", ""),
}

fig = plt.figure(figsize=(7.08, 4.7))
gs = GridSpec(2, 2, width_ratios=[1.28, 1], height_ratios=[1, 1],
              wspace=0.30, hspace=0.48)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# a: GWR local elasticity
norm = Normalize(vmin=poly["elasticity"].min(), vmax=poly["elasticity"].max())
poly.plot(column="elasticity", cmap="cividis", norm=norm, ax=ax_a,
          edgecolor="white", linewidth=0.35)
halo = [pe.withStroke(linewidth=2.2, foreground="white")]
for code, short in highlight.items():
    row = poly[poly["LSOA21CD"].eq(code)]
    row.boundary.plot(ax=ax_a, color=ORANGE if code == largest_code else GREEN,
                      linewidth=1.6)
    point = row.geometry.representative_point().iloc[0]
    ax_a.annotate(
        short, (point.x, point.y), ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=INK, path_effects=halo,
    )
ax_a.set_title("a  Local GVA–employment elasticity (GWR)",
               loc="left", fontsize=8, fontweight="bold")
ax_a.axis("off")
mapper = plt.cm.ScalarMappable(cmap="cividis", norm=norm)
cbar = fig.colorbar(mapper, ax=ax_a, shrink=0.44, pad=0.01, aspect=17)
cbar.set_label("Local elasticity", fontsize=6.2)
cbar.ax.tick_params(labelsize=6, length=2)
gwr = diagnostics["gwr"]
ax_a.text(
    0.02, 0.015,
    f"adaptive k={gwr['adaptive_neighbours']}  |  "
    f"range {gwr['local_elasticity_min']:.2f}–{gwr['local_elasticity_max']:.2f}",
    transform=ax_a.transAxes, fontsize=5.8, color=INK,
)

# b: same +30% intervention at two different LSOAs
values = contrast["allocation_pct"].to_numpy(float)
labels = [label.replace("Swindon ", "") for label in contrast["label"]]
colors = [ORANGE if value < 0 else GREEN for value in values]
x = np.arange(2)
ax_b.bar(x, values, color=colors, width=0.60, edgecolor="white", linewidth=0.5)
ax_b.axhline(0, color=INK, lw=0.8)
for xx, value in zip(x, values):
    ax_b.text(
        xx, value + (0.055 if value >= 0 else -0.055), f"{value:+.2f}%",
        ha="center", va="bottom" if value >= 0 else "top",
        fontsize=6.5, fontweight="bold",
        color=GREEN if value >= 0 else ORANGE,
    )
ax_b.set_xticks(x, [f"{labels[0]}\n(largest hub)",
                    f"{labels[1]}\n(highest return)"], fontsize=6.2)
ax_b.set_ylabel("Modelled GVA allocation effect (%)", fontsize=6.6)
ax_b.set_ylim(min(values.min() * 1.55, -0.48), values.max() * 1.28)
ax_b.set_title("b  +30% workplace-attractiveness scenario",
               loc="left", fontsize=8, fontweight="bold")
ax_b.tick_params(length=2)

# c: independent descriptive signals
plot_data = elasticity.merge(
    ranking[[
        "LSOA21CD", "adjusted_gva_per_workplace_commuter", "workplace_jobs"
    ]], on="LSOA21CD"
)
sizes = 8 + 34 * np.sqrt(
    plot_data["workplace_jobs"] / plot_data["workplace_jobs"].max()
)
ax_c.scatter(
    plot_data["adjusted_gva_per_workplace_commuter"],
    plot_data["elasticity"], s=sizes, color=BLUE, alpha=0.55,
    edgecolors="white", linewidths=0.35,
)
for code, color in [(largest_code, ORANGE), (top_code, GREEN)]:
    row = plot_data[plot_data["LSOA21CD"].eq(code)].iloc[0]
    ax_c.scatter(
        row["adjusted_gva_per_workplace_commuter"], row["elasticity"],
        s=54, color=color, edgecolors=INK, linewidths=0.7, zorder=3,
    )
    ax_c.annotate(
        highlight[code],
        (row["adjusted_gva_per_workplace_commuter"], row["elasticity"]),
        xytext=(5, 3), textcoords="offset points", fontsize=6.2,
        fontweight="bold", color=color,
    )
ax_c.axhline(
    gwr["global_employment_elasticity"], color=GREY, lw=0.8, ls="--"
)
ax_c.set_xlabel("Adjusted GVA per workplace commuter", fontsize=6.5)
ax_c.set_ylabel("Local elasticity", fontsize=6.5)
ax_c.set_title("c  Value and employment sensitivity differ",
               loc="left", fontsize=8, fontweight="bold")
ax_c.tick_params(length=2)

fig.suptitle(
    "LSOA detail changes the employment-investment signal",
    x=0.075, y=0.985, ha="left", fontsize=9.2, fontweight="bold",
)
save(fig, "fig2_policy_simulation")
print(
    f"saved fig2 | {contrast.iloc[0]['label']}={values[0]:+.2f}% | "
    f"{contrast.iloc[1]['label']}={values[1]:+.2f}%"
)
