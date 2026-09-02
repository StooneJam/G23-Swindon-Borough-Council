from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "commuting-data" / "data_swindon_with_commute.csv"
IPI = HERE / "ipi_tabpfn.csv"

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
GRID = 25

LABELS = {
    "log_voa_rv_2023": "log VOA rateable value", "rv_per_working_age": "RV per working-age adult",
    "sme_density": "SME density", "qualification_index": "Qualification index",
    "firm_size_diversity": "Firm-size diversity", "rv_per_employee": "RV per employee",
    "sme_qual_interaction": "SME x qualification", "employment_quality": "Employment quality",
    "modern_sector_leverage": "Modern-sector leverage", "asset_growth_diversity": "Asset-growth diversity",
    "msoa_out_commute_share": "Out-commute share", "msoa_same_msoa_work_share": "Same-MSOA work share",
    "msoa_wfh_share": "Work-from-home share", "msoa_in_commute_share": "In-commute share",
    "msoa_local_worker_share": "Local worker share",
}


@st.cache_resource
def load_model():
    df = pd.read_csv(DATA)
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    df = df.reset_index(drop=True)
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
    reg.fit(df[FEATS].to_numpy(float), df[TARGET].to_numpy(float))
    return df, reg


@st.cache_data
def curve(code, feat):
    df, reg = load_model()
    i = int(df.index[df["LSOA21CD"] == code][0])
    X = df[FEATS].to_numpy(float)
    j = FEATS.index(feat)
    lo, hi = float(df[feat].min()), float(df[feat].max())
    grid = np.linspace(lo, hi, GRID)
    rows = np.repeat(X[i:i + 1], GRID, axis=0)
    rows[:, j] = grid
    gva = np.exp(reg.predict(rows))
    actual = float(np.exp(df[TARGET].to_numpy(float)[i]))
    return grid, gva, float(X[i, j]), actual, lo, hi


st.set_page_config(page_title="Swindon IPI lever", layout="wide")
st.title("Swindon IPI — bottleneck lever simulator")
st.caption(
    "Model-implied (TabPFN), associational not causal — the curve shows how areas at different "
    "levels differ, not a guaranteed return on intervention. Clamped to the range observed across "
    "Swindon. The +30% line mirrors the 2036 ambition for illustration only."
)

ipi = pd.read_csv(IPI)
pri = ipi[ipi["IPI"] > 0].sort_values("rank")
opts = {f"#{int(r.rank)}   {r.LSOA21CD}   ·   {LABELS.get(r.bottleneck, r.bottleneck)}": (r.LSOA21CD, r.bottleneck)
        for r in pri.itertuples()}
sel = st.selectbox("Priority area (its bottleneck is the lever)", list(opts))
code, feat = opts[sel]

grid, gva, cur, actual, vmin, vmax = curve(code, feat)
base = float(np.interp(cur, grid, gva))
target = base * 1.3

is_pct = "share" in feat
step = (vmax - vmin) / 100
v = st.slider(f"{LABELS.get(feat, feat)}  (observed range)", float(vmin), float(vmax), float(cur), step=float(step))
g = float(np.interp(v, grid, gva))
pct = (g - base) / base * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("observed 2023", f"£{actual:.0f}m")
c2.metric("model at current", f"£{base:.0f}m")
c3.metric("at lever setting", f"£{g:.0f}m", f"{pct:+.0f}% vs now")
c4.metric("progress to +30%", f"{min(pct / 30 * 100, 999):.0f}%")

fig = go.Figure()
fig.add_hrect(y0=base, y1=target, fillcolor="green", opacity=0.08, line_width=0)
fig.add_trace(go.Scatter(x=grid, y=gva, mode="lines+markers", name="model response",
                         line=dict(color="#2f6df6", width=2.5), marker=dict(size=6)))
fig.add_hline(y=base, line_dash="dash", line_color="gray",
              annotation_text=f"now £{base:.0f}m", annotation_position="top left")
fig.add_hline(y=target, line_dash="dash", line_color="green",
              annotation_text=f"+30% target £{target:.0f}m", annotation_position="top right")
fig.add_vline(x=cur, line_dash="dot", line_color="orange",
              annotation_text="now", annotation_position="bottom")
fig.add_trace(go.Scatter(x=[v], y=[g], mode="markers+text", name="lever setting",
                         marker=dict(size=13, color="#ff5a5f", line=dict(color="white", width=2)),
                         text=[f"£{g:.0f}m  {pct:+.0f}%"], textposition="top center", showlegend=False))
fig.update_layout(
    xaxis_title=f"{LABELS.get(feat, feat)}" + (" (share)" if is_pct else ""),
    yaxis_title="predicted total GVA (£m)",
    height=470, margin=dict(l=10, r=10, t=30, b=10), template="simple_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
if is_pct:
    fig.update_xaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Area {code} · bottleneck lever: {LABELS.get(feat, feat)}. "
    f"Unrealised potential (need) = model-now minus observed = £{base - actual:.0f}m."
)
