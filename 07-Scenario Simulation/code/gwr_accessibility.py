from pathlib import Path
import json
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from lsoa_scenario_core import build_all_outputs

if not (HERE / "lsoa_local_elasticity.csv").exists():
    build_all_outputs()
elasticity = pd.read_csv(HERE / "lsoa_local_elasticity.csv")
diagnostics = json.loads(
    (HERE / "lsoa_model_diagnostics.json").read_text(encoding="utf-8")
)["gwr"]
geo = gpd.read_file(
    ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
)
geo = geo[geo["LSOA21CD"].isin(elasticity["LSOA21CD"])].merge(
    elasticity[["LSOA21CD", "elasticity"]], on="LSOA21CD"
).to_crs(27700)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
})
fig, ax = plt.subplots(figsize=(7.2, 7.4))
norm = Normalize(geo["elasticity"].min(), geo["elasticity"].max())
geo.plot(column="elasticity", cmap="cividis", norm=norm, ax=ax,
         edgecolor="white", linewidth=0.4)
ax.set_title(
    "Swindon LSOA local GVA–employment elasticity\n"
    f"(GWR, adaptive k={diagnostics['adaptive_neighbours']}; "
    f"pseudo-$R^2$={diagnostics['pseudo_r2']:.2f})",
    fontsize=11,
)
ax.axis("off")
mapper = plt.cm.ScalarMappable(cmap="cividis", norm=norm)
cbar = fig.colorbar(mapper, ax=ax, shrink=0.55, pad=0.01)
cbar.set_label("Local elasticity of GVA with respect to workplace jobs")
stem = HERE / "gwr_elasticity_map"
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print(
    f"saved gwr_elasticity_map | range "
    f"{diagnostics['local_elasticity_min']:.2f}-"
    f"{diagnostics['local_elasticity_max']:.2f}"
)
