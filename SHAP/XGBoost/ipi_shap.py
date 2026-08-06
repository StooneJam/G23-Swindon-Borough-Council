"""
Stage 1 of the XGBoost IPI pipeline: out-of-fold `need` + local SHAP.

Mirrors SHAP/TabPFN/ipi_shap.py but swaps the model to XGBoost and the dataset to
the LSOA-commute file. Two products feed the IPI build:

  * need_i  = max(0, oof_pred_i - y_i)   -- underperformance vs the model's
              out-of-fold expectation (log-GVA). OOF avoids in-sample optimism.
  * phi_ij  = local SHAP contribution of feature j for area i, from the EXACT
              TreeExplainer (a genuine XGBoost advantage over TabPFN, which needs
              a slower model-agnostic explainer).

XGBoost hyper-parameters are tuned once on the full Swindon sample and cached to
xgb_best_params.json so every stage uses the same fitted configuration.

Outputs (SHAP/XGBoost/): ipi_need.csv, shap_local.csv, shap_global.csv,
                         xgb_best_params.json
Run:  python SHAP/XGBoost/ipi_shap.py   (env N_ITER, default 80)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score

import shap

import ipi_common as C

warnings.filterwarnings("ignore")

N_ITER = int(os.environ.get("N_ITER", "80"))


def tune(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=C.RANDOM_STATE, n_jobs=-1),
        param_distributions=C.PARAM_DIST, n_iter=N_ITER, scoring="r2",
        cv=cv, random_state=C.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_params_, search.best_score_


def oof_predict(X, y, params):
    oof = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    for tr, te in cv.split(X):
        m = xgb.XGBRegressor(objective="reg:squarederror", random_state=C.RANDOM_STATE,
                             n_jobs=-1, **params)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def main():
    print(f"[config] N_ITER={N_ITER}", flush=True)
    df = C.load_df()
    X = df[C.FEATS].to_numpy(float)
    y = df[C.TARGET].to_numpy(float)
    print(f"rows: {len(df)} | features: {len(C.FEATS)}", flush=True)

    params, cv_r2 = tune(X, y)
    C.save_params(params)
    print(f"tuned XGBoost | CV R2(search)={cv_r2:.4f}", flush=True)

    # ---- need (out-of-fold underperformance) --------------------------------- #
    oof = oof_predict(X, y, params)
    need = np.clip(oof - y, 0, None)
    print(f"OOF R2={r2_score(y, oof):.4f} | areas with need>0: {(need > 0).sum()}", flush=True)

    # ---- local + global SHAP on the full-fit model (exact TreeExplainer) ------ #
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=C.RANDOM_STATE,
                             n_jobs=-1, **params)
    model.fit(X, y)
    print(f"in-sample R2={r2_score(y, model.predict(X)):.4f}", flush=True)

    explainer = shap.TreeExplainer(model)
    phi = explainer(X).values  # (n_areas, n_feats), signed, in log-GVA units

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

    print("saved shap_local.csv, shap_global.csv, ipi_need.csv, xgb_best_params.json", flush=True)


if __name__ == "__main__":
    main()
