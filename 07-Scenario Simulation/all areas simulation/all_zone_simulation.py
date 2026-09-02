"""Run all 137 LSOA scenarios and export the standalone priority map."""
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parent
sys.path.insert(0, str(SIM))
from lsoa_scenario_core import build_all_outputs

if not (HERE / "all_zone_ranking.csv").exists():
    build_all_outputs()
ranking = pd.read_csv(HERE / "all_zone_ranking.csv")
strategies = pd.read_csv(HERE / "optimisation_result.csv")
geo = gpd.read_file(
    ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
)
geo = geo[geo["LSOA21CD"].isin(ranking["LSOA21CD"])].merge(
    ranking[["LSOA21CD", "rank", "allocation_pct"]], on="LSOA21CD"
).to_crs(27700)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
})
fig, ax = plt.subplots(figsize=(7.2, 7.4))
norm = TwoSlopeNorm(
    vmin=geo["allocation_pct"].min(), vcenter=0,
    vmax=geo["allocation_pct"].max(),
)
geo.plot(column="allocation_pct", cmap="PuOr", norm=norm, ax=ax,
         edgecolor="white", linewidth=0.4)
top5 = set(ranking.head(5)["LSOA21CD"])
top3 = set(ranking.head(3)["LSOA21CD"])
halo = [pe.withStroke(linewidth=2.2, foreground="white")]
for _, row in geo[geo["LSOA21CD"].isin(top3)].iterrows():
    point = row.geometry.representative_point()
    ax.annotate(
        f"#{int(row['rank'])}", (point.x, point.y), ha="center", va="center",
        fontsize=7, fontweight="bold", path_effects=halo,
    )
geo[geo["LSOA21CD"].isin(top5)].boundary.plot(
    ax=ax, color="#3C8D82", linewidth=1.4
)
ax.set_title(
    "Swindon LSOA employment-investment priority\n"
    "(GVA allocation effect of a +30% workplace-attractiveness intervention)",
    fontsize=10.5,
)
ax.axis("off")
mapper = plt.cm.ScalarMappable(cmap="PuOr", norm=norm)
cbar = fig.colorbar(mapper, ax=ax, shrink=0.52, pad=0.01)
cbar.set_label("GVA allocation effect (% of Swindon total)")
stem = HERE / "all_zone_priority_map"
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print(ranking.head(10)[[
    "rank", "label", "workplace_jobs", "allocation_pct"
]].to_string(index=False))
print("\nEqual-budget strategies:")
print(strategies[[
    "strategy", "dGVA_allocation_pct", "budget_used"
]].to_string(index=False))
