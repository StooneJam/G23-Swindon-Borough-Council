"""Figure 1: LSOA inbound destination-choice gravity model."""
from pathlib import Path
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
if not (SIM / "lsoa_gravity_validation.csv").exists():
    from lsoa_scenario_core import build_all_outputs
    build_all_outputs()

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
INK, BLUE, TEAL, GREY = "#26323C", "#436F99", "#3C8D82", "#B9C2CB"


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=600, bbox_inches="tight")


flows = pd.read_csv(SIM / "lsoa_external_flows_geo.csv")
validation = pd.read_csv(SIM / "lsoa_gravity_validation.csv")
diagnostics = json.loads((SIM / "lsoa_model_diagnostics.json").read_text(encoding="utf-8"))
model = diagnostics["gravity_model"]

inbound = flows[flows["flow_type"].eq("inbound")]
destinations = (
    inbound.groupby(["dest_lsoa", "dest_lsoa_name"])["count"].sum()
    .nlargest(10).sort_values()
)

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.08, 3.0), gridspec_kw={"width_ratios": [1, 1.08]}
)

# a: observed destination concentration
y = np.arange(len(destinations))
colors = [BLUE] * len(destinations)
colors[-1] = TEAL
ax_a.barh(y, destinations.to_numpy(), color=colors, height=0.72,
          edgecolor="white", linewidth=0.4)
labels = [name.replace("Swindon ", "") for _, name in destinations.index]
ax_a.set_yticks(y, labels, fontsize=6.4)
for yy, value in zip(y, destinations.to_numpy()):
    ax_a.text(value + destinations.max() * 0.015, yy, f"{int(value):,}",
              va="center", fontsize=6, color=INK)
ax_a.set_xlim(0, destinations.max() * 1.18)
ax_a.set_xlabel("Observed inbound commuters")
ax_a.set_title("a  External workers concentrate in a few LSOAs",
               loc="left", fontsize=8, fontweight="bold")
ax_a.tick_params(length=2)

# b: origin-grouped 5-fold out-of-fold validation, positive OD cells
x = validation["observed"].to_numpy(float)
yhat = validation["predicted_oof"].to_numpy(float)
limit = max(x.max(), yhat.max()) * 1.35
ax_b.plot([0.08, limit], [0.08, limit], "--", color=GREY, lw=0.9, zorder=1)
ax_b.scatter(x, yhat, s=5, color=BLUE, alpha=0.18, edgecolors="none", zorder=2)
ax_b.set_xscale("log")
ax_b.set_yscale("log")
ax_b.set_xlim(0.7, limit)
ax_b.set_ylim(0.05, limit)
ax_b.set_xlabel("Observed flow (persons)")
ax_b.set_ylabel("Out-of-fold predicted flow")
ax_b.set_title("b  Five-fold gravity-model validation",
               loc="left", fontsize=8, fontweight="bold")
ax_b.set_box_aspect(1)
coef = model["coefficients"]
ax_b.text(
    0.04, 0.96,
    f"$r_{{positive}}={model['oof_correlation_positive_pairs']:.3f}$"
    f"\n$r_{{all}}={model['oof_correlation_all_pairs']:.3f}$"
    f"\n$T_{{ij}}\\propto O_i^{{{coef['origin_mass']:.2f}}}"
    f"D_j^{{{coef['destination_jobs']:.2f}}}"
    f"d_{{ij}}^{{{coef['distance']:.2f}}}$",
    transform=ax_b.transAxes, ha="left", va="top", fontsize=6.2, color=INK,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6),
)
ax_b.tick_params(length=2)

fig.suptitle(
    "LSOA-level inbound commuting model for Swindon",
    x=0.075, y=1.01, ha="left", fontsize=9.2, fontweight="bold",
)
fig.tight_layout(w_pad=1.8)
save(fig, "fig1_gravity_model")
print(
    "saved fig1 | "
    f"OOF r(all)={model['oof_correlation_all_pairs']:.3f}, "
    f"r(positive)={model['oof_correlation_positive_pairs']:.3f}"
)
