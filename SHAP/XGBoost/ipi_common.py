"""
Shared configuration and helpers for the XGBoost IPI pipeline.

This is the XGBoost counterpart to the TabPFN IPI scripts in SHAP/TabPFN/. It runs
on the Swindon-only LSOA-commute dataset (`data_swindon_with_lsoa_commute.csv`),
which merges the 10 LLM-engineered features with 15 LSOA-level commuting features.
Target: log_total_GVA_2023.

The three pipeline stages import from here so that feature lists, actionability
weights, and the IPI definition stay identical across stages:
    ipi_shap.py        -> out-of-fold `need` + local SHAP (TreeExplainer)
    ipi_build.py       -> IPI, bottlenecks, sensitivity analysis
    ipi_investment.py  -> bottleneck lever curves + "+30% GVA" table (data only)

Design mirrors SHAP/TabPFN so an XGBoost-vs-TabPFN comparison is like-for-like,
EXCEPT the commuting geography (LSOA here vs MSOA in the current TabPFN run). For a
clean model comparison the TabPFN pipeline should be re-run on this same dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_CSV = ROOT / "commuting-regression" / "data_swindon_with_lsoa_commute.csv"

TARGET = "log_total_GVA_2023"
RANDOM_STATE = 42

ENG = [
    "log_voa_rv_2023", "rv_per_working_age", "sme_density", "qualification_index",
    "firm_size_diversity", "rv_per_employee", "sme_qual_interaction",
    "employment_quality", "modern_sector_leverage", "asset_growth_diversity",
]
COMMUTE = [
    "lsoa_total_employed", "lsoa_home_or_no_fixed_count", "lsoa_home_or_no_fixed_share",
    "lsoa_workplace_commuters", "lsoa_same_lsoa_worker_count", "lsoa_same_lsoa_work_share",
    "lsoa_out_commute_share", "lsoa_workers_at_workplace", "lsoa_inbound_worker_count",
    "lsoa_local_worker_share", "lsoa_in_commute_share", "lsoa_outbound_from_swindon_count",
    "lsoa_outbound_from_swindon_share", "lsoa_inbound_to_swindon_count",
    "lsoa_inbound_to_swindon_share",
]
FEATS = ENG + COMMUTE

# Actionability weights (policy judgement, NOT learned). Three tiers:
#   0    structural / scale / headcount counts the Council cannot directly move
#   0.5  semi-actionable, influenced slowly through the market
#   1    direct commuting-share levers
W = {
    # engineered — structural
    "log_voa_rv_2023": 0, "rv_per_working_age": 0, "rv_per_employee": 0,
    # engineered — semi-actionable
    "sme_density": 0.5, "qualification_index": 0.5, "firm_size_diversity": 0.5,
    "sme_qual_interaction": 0.5, "employment_quality": 0.5,
    "modern_sector_leverage": 0.5, "asset_growth_diversity": 0.5,
    # commute — headcount / absolute counts (scale proxies)
    "lsoa_total_employed": 0, "lsoa_home_or_no_fixed_count": 0,
    "lsoa_workplace_commuters": 0, "lsoa_same_lsoa_worker_count": 0,
    "lsoa_workers_at_workplace": 0, "lsoa_inbound_worker_count": 0,
    "lsoa_outbound_from_swindon_count": 0, "lsoa_inbound_to_swindon_count": 0,
    # commute — semi-actionable (home/no-fixed working)
    "lsoa_home_or_no_fixed_share": 0.5,
    # commute — direct share levers
    "lsoa_out_commute_share": 1, "lsoa_in_commute_share": 1,
    "lsoa_same_lsoa_work_share": 1, "lsoa_local_worker_share": 1,
    "lsoa_outbound_from_swindon_share": 1, "lsoa_inbound_to_swindon_share": 1,
}

LABELS = {
    "log_voa_rv_2023": "log VOA rateable value", "rv_per_working_age": "RV per working-age adult",
    "sme_density": "SME density", "qualification_index": "Qualification index",
    "firm_size_diversity": "Firm-size diversity", "rv_per_employee": "RV per employee",
    "sme_qual_interaction": "SME x qualification", "employment_quality": "Employment quality",
    "modern_sector_leverage": "Modern-sector leverage", "asset_growth_diversity": "Asset-growth diversity",
    "lsoa_total_employed": "Total employed (n)", "lsoa_home_or_no_fixed_count": "Home/no-fixed workers (n)",
    "lsoa_home_or_no_fixed_share": "Home/no-fixed share", "lsoa_workplace_commuters": "Workplace commuters (n)",
    "lsoa_same_lsoa_worker_count": "Same-LSOA workers (n)", "lsoa_same_lsoa_work_share": "Same-LSOA work share",
    "lsoa_out_commute_share": "Out-commute share", "lsoa_workers_at_workplace": "Workers at workplace (n)",
    "lsoa_inbound_worker_count": "Inbound workers (n)", "lsoa_local_worker_share": "Local worker share",
    "lsoa_in_commute_share": "In-commute share", "lsoa_outbound_from_swindon_count": "Outbound from Swindon (n)",
    "lsoa_outbound_from_swindon_share": "Outbound from Swindon share",
    "lsoa_inbound_to_swindon_count": "Inbound to Swindon (n)",
    "lsoa_inbound_to_swindon_share": "Inbound to Swindon share",
}

# XGBoost search space (same as SHAP/XGBoost/shap_analysis_commute.py).
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500, 700, 1000],
    "max_depth": [2, 3, 4, 5, 6, 7],
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10, 15],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1, 2],
    "reg_lambda": [0.5, 1, 1.5, 2, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.5, 1],
}
PARAMS_JSON = HERE / "xgb_best_params.json"


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    return df.reset_index(drop=True)


def labels(cols):
    return [LABELS.get(c, c) for c in cols]


def is_commute(col):
    return col in COMMUTE


def ipi(need, absphi, ws):
    """IPI = rank%(need) * rank%(weighted |SHAP|), zeroed where need <= 0."""
    wv = pd.Series(ws).reindex(FEATS).fillna(0)
    L = absphi.mul(wv, axis=1).sum(axis=1)
    score = need.rank(pct=True) * L.rank(pct=True)
    return score.where(need > 0, 0.0)


def topk_overlap(base, alt, codes, k):
    """Fraction of base top-k priority areas retained by an alternative weighting."""
    b = set(codes[np.argsort(-np.asarray(base))[:k]])
    a = set(codes[np.argsort(-np.asarray(alt))[:k]])
    return len(b & a) / k


def nearest_cross(grid, gva, cur, target):
    xs = np.linspace(grid[0], grid[-1], 500)
    ys = np.interp(xs, grid, gva)
    idx = np.where(ys >= target)[0]
    if len(idx) == 0:
        return None
    return float(xs[idx[np.argmin(np.abs(xs[idx] - cur))]])


def save_params(params: dict):
    PARAMS_JSON.write_text(json.dumps(params, indent=1))


def load_params() -> dict:
    return json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}
