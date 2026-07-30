"""
SHAP analysis for the TabPFN (v3) GVA model with commuting features.

Explains the TabPFN regressor from `commuting-regression/tabpfn_commute.ipynb`
on the Swindon-only dataset `data_swindon_with_commute.csv`, which merges the
LLM-engineered features (`total_gva_engineered_features.csv`) with MSOA-level
commuting features.

Because TabPFN is a transformer (in-context) model, not a tree, TreeExplainer
does not apply. We use a model-agnostic PermutationExplainer over the fitted
`reg.predict` function, mirroring the standard SHAP walkthrough
(zhuanlan.zhihu.com/p/456843338) but with the explainer swapped for a
non-tree-compatible one.

Following the notebook, we explain TWO feature sets and compare them:
    * "engineered only"      -> 10 LLM-engineered features
    * "engineered + commute" -> those 10 + 9 commuting features
This isolates what the commuting signal adds.

Target: log_total_GVA_2023 (log scale -> SHAP additive in log-GVA).
`is_swindon` is control-only and excluded from the predictors.

Outputs -> figures/ , shap_feature_importance_tabpfn.csv
Run:  python SHAP/TabPFN/shap_analysis_tabpfn.py
Env knobs: MAX_BG (background masker samples, default 60),
           MAX_EVALS (permutation evals/row, default 2*n_feat+1).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import shap
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root (SHAP/TabPFN/ -> ../../)
DATA_CSV = ROOT / "commuting-regression" / "data_swindon_with_commute.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_BG = int(os.environ.get("MAX_BG", "60"))       # background samples for masker
MAX_EVALS_ENV = os.environ.get("MAX_EVALS", "")    # permutation evals per row

TARGET = "log_total_GVA_2023"
IDS = ["LSOA21CD", "MSOA21CD"]
DROP = IDS + ["is_swindon", TARGET]

ENG_FEATS = [
    "log_voa_rv_2023",
    "rv_per_working_age",
    "sme_density",
    "qualification_index",
    "firm_size_diversity",
    "rv_per_employee",
    "sme_qual_interaction",
    "employment_quality",
    "modern_sector_leverage",
    "asset_growth_diversity",
]
COMMUTE_FEATS = [
    "msoa_out_commute_share",
    "msoa_same_msoa_work_share",
    "msoa_wfh_share",
    "msoa_workplace_commuters",
    "msoa_total_employed",
    "msoa_in_commute_share",
    "msoa_local_worker_share",
    "msoa_workers_at_workplace",
    "msoa_inbound_worker_count",
]

FEATURE_LABELS = {
    # engineered (LLM)
    "log_voa_rv_2023": "log VOA rateable value",
    "rv_per_working_age": "RV per working-age adult",
    "sme_density": "SME density",
    "qualification_index": "Qualification index",
    "firm_size_diversity": "Firm-size diversity",
    "rv_per_employee": "RV per employee",
    "sme_qual_interaction": "SME x qualification",
    "employment_quality": "Employment quality",
    "modern_sector_leverage": "Modern-sector leverage",
    "asset_growth_diversity": "Asset-growth diversity",
    # commute
    "msoa_out_commute_share": "Out-commute share",
    "msoa_same_msoa_work_share": "Same-MSOA work share",
    "msoa_wfh_share": "Work-from-home share",
    "msoa_workplace_commuters": "Workplace commuters (n)",
    "msoa_total_employed": "Total employed (n)",
    "msoa_in_commute_share": "In-commute share",
    "msoa_local_worker_share": "Local worker share",
    "msoa_workers_at_workplace": "Workers at workplace (n)",
    "msoa_inbound_worker_count": "Inbound workers (n)",
}
TARGET_LABEL = "log Total GVA (2023)"

FEATURE_SETS = {
    "engineered_only": ENG_FEATS,
    "engineered_commute": ENG_FEATS + COMMUTE_FEATS,
}

# --------------------------------------------------------------------------- #
# Aesthetics (shared with the XGBoost analysis)                                #
# --------------------------------------------------------------------------- #
NAVY, CORAL, TEAL = "#1f3b57", "#ff5a5f", "#2ec4b6"
SHAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "sbc", ["#2f6df6", "#8a97a5", "#ff5a5f"]
)
plt.rcParams.update(
    {
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.edgecolor": "#c9d1d9", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#eceff3", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.color": "#43515e", "ytick.color": "#43515e",
        "text.color": "#22303c", "axes.labelcolor": "#22303c", "figure.facecolor": "white",
    }
)


def _labels(cols):
    return [FEATURE_LABELS.get(c, c) for c in cols]


def is_commute(col):
    return col in COMMUTE_FEATS


# --------------------------------------------------------------------------- #
# Plot helpers                                                                 #
# --------------------------------------------------------------------------- #
def plot_bar_importance(mean_abs, cols, title, path):
    order = np.argsort(mean_abs)
    labels = np.array(_labels(cols))[order]
    vals = mean_abs[order]
    # colour commute features distinctly so their contribution stands out
    colors = [CORAL if is_commute(cols[i]) else NAVY for i in order]

    fig, ax = plt.subplots(figsize=(8.6, max(5.2, 0.42 * len(cols))))
    bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=8.5, color="#43515e")
    ax.set_xlabel("mean(|SHAP value|)  —  average impact on log-GVA")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, vals.max() * 1.13)
    handles = [mpl.patches.Patch(color=NAVY, label="engineered (LLM)"),
               mpl.patches.Patch(color=CORAL, label="commuting")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_beeswarm(shap_values, title, path):
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=len(shap_values.feature_names),
                        show=False, color=SHAP_CMAP)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    plt.title(title, loc="left", fontweight="bold")
    plt.xlabel("SHAP value (impact on log-GVA)")
    fig.savefig(path)
    plt.close(fig)


def plot_dependence(shap_values, feature_label, title, path):
    plt.figure()
    shap.plots.scatter(shap_values[:, feature_label], color=shap_values,
                       cmap=SHAP_CMAP, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.2)
    plt.title(title, loc="left", fontweight="bold")
    plt.xlabel(feature_label)
    plt.ylabel("SHAP value (impact on log-GVA)")
    fig.savefig(path)
    plt.close(fig)


def plot_waterfall(shap_values, idx, area_name, title, path):
    plt.figure()
    shap.plots.waterfall(shap_values[idx],
                         max_display=len(shap_values.feature_names), show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    plt.title(f"{title}\nLocal explanation: {area_name}", loc="left", fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def plot_shared_delta(imp_eng, imp_full, path):
    """How the 10 engineered features' importance changes once commute is added."""
    shared = ENG_FEATS
    e = np.array([imp_eng[f] for f in shared])
    f = np.array([imp_full[f] for f in shared])
    order = np.argsort(f)
    labels = np.array(_labels(shared))[order]
    y = np.arange(len(shared))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + h / 2, e[order], height=h, color=NAVY, edgecolor="white",
            linewidth=0.5, label="engineered only")
    ax.barh(y - h / 2, f[order], height=h, color=TEAL, edgecolor="white",
            linewidth=0.5, label="engineered + commute")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Do commuting features displace the engineered signal?\n"
                 "engineered-feature importance, before vs after adding commute",
                 loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# SHAP driver                                                                  #
# --------------------------------------------------------------------------- #
def explain_set(name, feats, df):
    print(f"\n=== TabPFN | {name} | {len(feats)} features ===", flush=True)
    X = df[feats].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
    reg.fit(X, y)

    # In-sample fit quality (sanity check; not a generalisation estimate).
    pred = reg.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    print(f"  in-sample R2={1 - ss_res / ss_tot:.4f}", flush=True)

    bg = shap.maskers.Independent(X, max_samples=min(MAX_BG, len(X)))
    explainer = shap.PermutationExplainer(reg.predict, bg, feature_names=_labels(feats))
    kwargs = {}
    if MAX_EVALS_ENV:
        kwargs["max_evals"] = int(MAX_EVALS_ENV)
    sv = explainer(X, **kwargs)
    sv.feature_names = _labels(feats)

    mean_abs = np.abs(sv.values).mean(axis=0)
    importance = dict(zip(feats, mean_abs))

    tag = name
    plot_bar_importance(mean_abs, feats,
                        f"Global feature importance — {name.replace('_', ' ')}\n{TARGET_LABEL}",
                        FIG_DIR / f"01_importance_{tag}.png")
    plot_beeswarm(sv, f"SHAP beeswarm — {name.replace('_', ' ')} | {TARGET_LABEL}",
                  FIG_DIR / f"02_beeswarm_{tag}.png")

    top3 = [feats[i] for i in np.argsort(mean_abs)[::-1][:3]]
    for rank, feat in enumerate(top3, 1):
        lbl = FEATURE_LABELS.get(feat, feat)
        plot_dependence(sv, lbl, f"{lbl} — {TARGET_LABEL}",
                        FIG_DIR / f"03_dependence_{tag}_{rank}_{feat[:20]}.png")

    idx = int(np.argmax(np.abs(sv.values).sum(axis=1)))
    area = str(df.iloc[idx]["LSOA21CD"])
    plot_waterfall(sv, idx, area, f"{name.replace('_', ' ')} | {TARGET_LABEL}",
                   FIG_DIR / f"04_waterfall_{tag}.png")

    return importance


def main():
    print(f"[config] MAX_BG={MAX_BG}  MAX_EVALS={MAX_EVALS_ENV or 'auto'}", flush=True)
    df = pd.read_csv(DATA_CSV)
    # data_swindon_with_commute.csv is already Swindon-only; guard anyway.
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    print(f"rows: {len(df)} | MSOAs: {df['MSOA21CD'].nunique()}", flush=True)

    importances = {}
    for name, feats in FEATURE_SETS.items():
        importances[name] = explain_set(name, feats, df)

    # comparison: engineered importance before/after commute
    plot_shared_delta(importances["engineered_only"],
                      importances["engineered_commute"],
                      FIG_DIR / "05_engineered_shift_with_commute.png")

    # tidy importance table (long format)
    rows = []
    for setname, imp in importances.items():
        for feat, val in imp.items():
            rows.append({"feature_set": setname,
                         "feature": FEATURE_LABELS.get(feat, feat),
                         "group": "commuting" if is_commute(feat) else "engineered",
                         "mean_abs_shap": val})
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(HERE / "shap_feature_importance_tabpfn.csv", index=False)

    # headline: how much of the full model's importance comes from commute
    full = importances["engineered_commute"]
    tot = sum(full.values())
    com = sum(v for k, v in full.items() if is_commute(k))
    print(f"\nCommuting features account for {com / tot * 100:.1f}% of total "
          f"mean|SHAP| in the full model.", flush=True)
    print("Done. Figures ->", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
