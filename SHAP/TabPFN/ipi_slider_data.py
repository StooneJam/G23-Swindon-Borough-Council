import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
COM = [
    "msoa_out_commute_share", "msoa_same_msoa_work_share", "msoa_wfh_share",
    "msoa_workplace_commuters", "msoa_total_employed", "msoa_in_commute_share",
    "msoa_local_worker_share", "msoa_workers_at_workplace", "msoa_inbound_worker_count",
]
FEATS = ENG + COM

TARGETS = {
    "E01015473": "msoa_out_commute_share",
    "E01033845": "msoa_wfh_share",
    "E01033861": "firm_size_diversity",
    "E01015569": "msoa_same_msoa_work_share",
    "E01015524": "employment_quality",
}
GRID = 25

LABELS = {
    "msoa_out_commute_share": "Out-commute share",
    "msoa_wfh_share": "Work-from-home share",
    "firm_size_diversity": "Firm-size diversity",
    "msoa_same_msoa_work_share": "Same-MSOA work share",
    "employment_quality": "Employment quality",
}


def main():
    df = pd.read_csv(DATA)
    if "is_swindon" in df.columns:
        df = df[df["is_swindon"] == "Swindon"].copy()
    df = df.reset_index(drop=True)
    X = df[FEATS].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
    reg.fit(X, y)

    out = {}
    for code, feat in TARGETS.items():
        i = int(df.index[df["LSOA21CD"] == code][0])
        j = FEATS.index(feat)
        lo, hi = float(df[feat].min()), float(df[feat].max())
        grid = np.linspace(lo, hi, GRID)
        rows = np.repeat(X[i:i + 1], GRID, axis=0)
        rows[:, j] = grid
        gva = np.exp(reg.predict(rows))
        out[code] = {
            "feature": feat,
            "label": LABELS[feat],
            "grid": [float(v) for v in grid],
            "gva": [float(v) for v in gva],
            "current_value": float(X[i, j]),
            "current_pred_gva": float(np.exp(reg.predict(X[i:i + 1])[0])),
            "actual_gva": float(np.exp(y[i])),
            "vmin": lo,
            "vmax": hi,
        }

    (HERE / "slider_data.json").write_text(json.dumps(out, indent=1))
    print("wrote slider_data.json for", list(out))


if __name__ == "__main__":
    main()
