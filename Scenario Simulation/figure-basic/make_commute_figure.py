"""Publication figure: LSOA-level Swindon commuting structure."""
from pathlib import Path
import re
import sys

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parent
sys.path.insert(0, str(SIM))
if not (SIM / "lsoa_internal_od_estimated.csv").exists():
    from lsoa_scenario_core import build_all_outputs
    build_all_outputs()

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
})
C_INTERNAL, C_OUT, C_IN, INK = "#5878A8", "#D07C3E", "#3C8D82", "#26323C"


def authority(label):
    return re.sub(r"\s+\d+[A-Za-z]?$", "", str(label)).strip()


internal = pd.read_csv(SIM / "lsoa_internal_od_estimated.csv")
external = pd.read_csv(SIM / "lsoa_external_flows_geo.csv")
features = pd.read_csv(
    ROOT / "commuting-regression" / "data_swindon_with_lsoa_commute.csv"
)

pivot = internal.pivot(
    index="origin_lsoa", columns="dest_lsoa", values="count"
).fillna(0)
order = pivot.sum(axis=0).sort_values(ascending=False).index
pivot = pivot.reindex(index=order, columns=order)
matrix = pivot.to_numpy(float)

internal_total = int(round(
    features["lsoa_workplace_commuters"].sum()
    - features["lsoa_outbound_from_swindon_count"].sum()
))
outbound_total = int(features["lsoa_outbound_from_swindon_count"].sum())
inbound_total = int(features["lsoa_inbound_to_swindon_count"].sum())
self_containment = 100 * internal_total / (internal_total + outbound_total)

out = external[external["flow_type"].eq("outbound")].copy()
out["authority"] = out["dest_lsoa_name"].map(authority)
out = out.groupby("authority")["count"].sum()
inbound = external[external["flow_type"].eq("inbound")].copy()
inbound["authority"] = inbound["origin_lsoa_name"].map(authority)
inbound = inbound.groupby("authority")["count"].sum()
partners = pd.DataFrame({"outbound": out, "inbound": inbound}).fillna(0)
partners["total"] = partners.sum(axis=1)
partners = partners.nlargest(10, "total").sort_values("total")

fig = plt.figure(figsize=(7.2, 5.5))
gs = gridspec.GridSpec(
    2, 2, figure=fig, width_ratios=[1.28, 1], height_ratios=[1, 1.05],
    left=0.075, right=0.96, top=0.90, bottom=0.10,
    wspace=0.42, hspace=0.54,
)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# a: estimated internal OD matrix
positive = matrix[matrix > 0]
image = ax_a.imshow(
    np.ma.masked_less_equal(matrix, 0), cmap="magma_r",
    norm=LogNorm(vmin=max(0.2, positive.min()), vmax=positive.max()),
    interpolation="nearest", aspect="equal",
)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.set_xlabel("Workplace LSOAs ordered by workplace jobs")
ax_a.set_ylabel("Residence LSOAs in the same order")
for spine in ax_a.spines.values():
    spine.set_visible(True)
ax_a.set_title("a  Estimated within-Swindon LSOA flows", loc="left",
               fontsize=8, fontweight="bold")
ax_a.text(
    0.0, -0.055,
    "Internal cells reconstructed from LSOA marginals; diagonal counts observed",
    transform=ax_a.transAxes, ha="left", va="top", fontsize=5.7, color="#555555",
)
cax = ax_a.inset_axes([0.0, -0.13, 1.0, 0.025])
cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
cbar.set_label("Estimated commuters per LSOA pair (log scale)", fontsize=6.2)
cbar.ax.tick_params(labelsize=5.5, length=2)

# b: boundary balance
labels = ["Within\nSwindon", "Out to\nother areas", "In from\nother areas"]
values = [internal_total, outbound_total, inbound_total]
colors = [C_INTERNAL, C_OUT, C_IN]
y = np.arange(3)[::-1]
ax_b.barh(y, values, color=colors, height=0.62)
ax_b.set_yticks(y, labels)
ax_b.set_xlim(0, max(values) * 1.23)
ax_b.set_xlabel("Commuters (persons)")
ax_b.spines["left"].set_visible(False)
ax_b.tick_params(axis="y", length=0)
for yy, value in zip(y, values):
    ax_b.text(value + max(values) * 0.015, yy, f"{value:,}",
              va="center", fontsize=6.3, fontweight="bold", color=INK)
ax_b.text(
    0.98, 0.04,
    f"Resident self-containment\n{self_containment:.1f}%",
    transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.2,
    color=C_INTERNAL,
    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=C_INTERNAL, lw=0.6),
)
ax_b.set_title("b  Boundary balance (137 modelled LSOAs)", loc="left",
               fontsize=8, fontweight="bold")

# c: observed external partners
yy = np.arange(len(partners))
in_values = partners["inbound"].to_numpy()
out_values = partners["outbound"].to_numpy()
ax_c.barh(yy, -in_values, color=C_IN, height=0.68)
ax_c.barh(yy, out_values, color=C_OUT, height=0.68)
ax_c.axvline(0, color=INK, lw=0.7)
ax_c.set_yticks(yy, partners.index, fontsize=6.1)
ax_c.tick_params(axis="y", length=0)
xmax = max(in_values.max(), out_values.max())
ax_c.set_xlim(-xmax * 1.25, xmax * 1.25)
ticks = np.linspace(-6000, 6000, 5)
ax_c.set_xticks(ticks, [f"{abs(int(t)):,}" if t else "0" for t in ticks])
ax_c.set_xlabel("Observed cross-boundary commuters")
ax_c.spines["left"].set_visible(False)
ax_c.text(-xmax * 0.52, len(partners) - 0.2, "Into Swindon",
          color=C_IN, ha="center", va="bottom", fontsize=6.2, fontweight="bold")
ax_c.text(xmax * 0.52, len(partners) - 0.2, "Out of Swindon",
          color=C_OUT, ha="center", va="bottom", fontsize=6.2, fontweight="bold")
ax_c.set_ylim(-0.7, len(partners) + 0.15)
ax_c.set_title("c  Main external partner authorities", loc="left",
               fontsize=8, fontweight="bold")

fig.suptitle(
    "Swindon commuting structure at LSOA level, Census 2021",
    x=0.075, y=0.97, ha="left", fontsize=9.5, fontweight="bold",
)
stem = HERE / "commute_figure"
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print(
    f"saved commute_figure | internal={internal_total:,}, "
    f"outbound={outbound_total:,}, inbound={inbound_total:,}"
)
