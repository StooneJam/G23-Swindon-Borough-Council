import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import shap
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "commuting-regression" / "data_swindon_with_commute.csv"

TARGET = "log_total_GVA_2023"
ENG = [
    "log_voa_rv_2023", "rv_per_working_age", "sme_density", "qualification_index",
    "firm_size_diversity", "rv_per_employee", "sme_qual_interaction",
    "employment_quality", "modern_sector_leverage", "asset_growth_diversity",
]
COMMUTE = [
    "msoa_out_commute_share", "msoa_same_msoa_work_share", "msoa_wfh_share",
    "msoa_workplace_commuters", "msoa_total_employed", "msoa_in_commute_share",
    "msoa_local_worker_share", "msoa_workers_at_workplace", "msoa_inbound_worker_count",
]
FEATS = ENG + COMMUTE


def new_model():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3)


def oof_pred(X, y):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
        m = new_model()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    df = pd.read_csv(DATA)
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    X = df[FEATS].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    need = np.clip(oof_pred(X, y) - y, 0, None)

    reg = new_model()
    reg.fit(X, y)
    bg = shap.maskers.Independent(X, max_samples=min(80, len(X)))
    sv = shap.PermutationExplainer(reg.predict, bg)(X, max_evals=200)

    phi = pd.DataFrame(sv.values, columns=FEATS)
    phi.insert(0, "LSOA21CD", df["LSOA21CD"].to_numpy())
    phi.to_csv(HERE / "shap_local_engineered_commute.csv", index=False)

    pd.DataFrame({
        "LSOA21CD": df["LSOA21CD"].to_numpy(),
        "MSOA21CD": df["MSOA21CD"].to_numpy(),
        "need": need,
    }).to_csv(HERE / "ipi_need.csv", index=False)

    print("saved shap_local_engineered_commute.csv, ipi_need.csv")


if __name__ == "__main__":
    main()
