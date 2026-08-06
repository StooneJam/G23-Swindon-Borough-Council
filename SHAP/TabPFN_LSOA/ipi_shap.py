"""
Stage 1 (TabPFN, LSOA): out-of-fold `need` + local SHAP.

Same recipe and outputs as SHAP/XGBoost/ipi_shap.py so the two are comparable, but
the model is TabPFN and, because TabPFN is not a tree, SHAP uses the model-agnostic
PermutationExplainer (not TreeSHAP).

  need_i = max(0, oof_pred_i - y_i)         (5-fold TabPFN out-of-fold)
  phi_ij = permutation SHAP contribution    (log-GVA units, signed)

Outputs (SHAP/TabPFN_LSOA/): ipi_need.csv, shap_local.csv, shap_global.csv
Run:  python SHAP/TabPFN_LSOA/ipi_shap.py
Env:  MAX_BG (background samples, default 80), MAX_EVALS (per row, default 200)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import shap
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import ipi_common as C

warnings.filterwarnings("ignore")

MAX_BG = int(os.environ.get("MAX_BG", "80"))
MAX_EVALS = int(os.environ.get("MAX_EVALS", "200"))


def new_model():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3)


def oof_predict(X, y):
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=C.RANDOM_STATE).split(X):
        m = new_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def main():
    print(f"[config] MAX_BG={MAX_BG} MAX_EVALS={MAX_EVALS}", flush=True)
    df = C.load_df()
    X = df[C.FEATS].to_numpy(float)
    y = df[C.TARGET].to_numpy(float)
    print(f"rows: {len(df)} | features: {len(C.FEATS)}", flush=True)

    oof = oof_predict(X, y)
    need = np.clip(oof - y, 0, None)
    print(f"OOF R2={r2_score(y, oof):.4f} | areas with need>0: {(need > 0).sum()}", flush=True)

    reg = new_model()
    reg.fit(X, y)
    print(f"in-sample R2={r2_score(y, reg.predict(X)):.4f}", flush=True)

    bg = shap.maskers.Independent(X, max_samples=min(MAX_BG, len(X)))
    sv = shap.PermutationExplainer(reg.predict, bg)(X, max_evals=MAX_EVALS)
    phi = sv.values  # (n_areas, n_feats), signed, log-GVA units

    shap_local = pd.DataFrame(phi, columns=C.FEATS)
    shap_local.insert(0, "LSOA21CD", df["LSOA21CD"].to_numpy())
    shap_local.to_csv(C.HERE / "shap_local.csv", index=False)

    mean_abs = np.abs(phi).mean(axis=0)
    pd.DataFrame({
        "feature": C.FEATS, "label": C.labels(C.FEATS),
        "group": ["commuting" if C.is_commute(f) else "engineered" for f in C.FEATS],
        "weight": [C.W[f] for f in C.FEATS], "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).to_csv(
        C.HERE / "shap_global.csv", index=False)

    pd.DataFrame({
        "LSOA21CD": df["LSOA21CD"].to_numpy(), "MSOA21CD": df["MSOA21CD"].to_numpy(),
        "need": need,
    }).to_csv(C.HERE / "ipi_need.csv", index=False)

    print("saved shap_local.csv, shap_global.csv, ipi_need.csv", flush=True)


if __name__ == "__main__":
    main()
