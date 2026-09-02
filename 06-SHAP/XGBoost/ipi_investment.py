from __future__ import annotations

import os
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

import ipi_common as C

warnings.filterwarnings("ignore")

N_TOP = int(os.environ.get("N_TOP", "5"))
GRID = int(os.environ.get("GRID", "25"))
UPLIFT = 1.30


def main():
    ipi = pd.read_csv(C.HERE / "ipi_xgboost.csv")
    top = ipi[ipi["IPI"] > 0].sort_values("rank").head(N_TOP).reset_index(drop=True)

    df = C.load_df()
    X = df[C.FEATS].to_numpy(float)
    y = df[C.TARGET].to_numpy(float)

    params = C.load_params()
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=C.RANDOM_STATE,
                             n_jobs=-1, **params)
    model.fit(X, y)

    curves, rows = {}, []
    for _, r in top.iterrows():
        code, feat = r["LSOA21CD"], r["bottleneck"]
        i = int(df.index[df["LSOA21CD"] == code][0])
        j = C.FEATS.index(feat)
        lo, hi = float(df[feat].min()), float(df[feat].max())
        grid = np.linspace(lo, hi, GRID)
        rowsX = np.repeat(X[i:i + 1], GRID, axis=0)
        rowsX[:, j] = grid
        gva = np.exp(model.predict(rowsX))
        cur = float(X[i, j])
        actual = float(np.exp(y[i]))
        base = float(np.interp(cur, grid, gva))
        target = base * UPLIFT
        xc = C.nearest_cross(grid, gva, cur, target)

        curves[code] = {"feature": feat, "label": C.LABELS.get(feat, feat),
                        "grid": grid.tolist(), "gva": gva.tolist(),
                        "current_value": cur, "actual_gva": actual,
                        "vmin": lo, "vmax": hi}
        rows.append({
            "rank": int(r["rank"]), "LSOA21CD": code, "MSOA21CD": r["MSOA21CD"],
            "IPI": round(float(r["IPI"]), 4), "bottleneck": feat,
            "bottleneck_label": C.LABELS.get(feat, feat),
            "gva_observed_m": round(actual, 1), "gva_now_model_m": round(base, 1),
            "gva_unrealised_m": round(base - actual, 1), "gva_target_+30pct_m": round(target, 1),
            "lever_now": round(cur, 4),
            "lever_at_+30pct": None if xc is None else round(xc, 4),
            "lever_change": None if xc is None else round(xc - cur, 4),
            "reachable_within_observed_range": xc is not None,
        })

    (C.HERE / "lever_curves.json").write_text(json.dumps(curves, indent=1))
    inv = pd.DataFrame(rows)
    inv.to_csv(C.HERE / "ipi_investment.csv", index=False)
    print(inv[["rank", "LSOA21CD", "bottleneck_label", "gva_now_model_m",
               "gva_target_+30pct_m", "lever_now", "lever_at_+30pct",
               "reachable_within_observed_range"]].to_string(index=False), flush=True)
    print("\nsaved ipi_investment.csv, lever_curves.json", flush=True)


if __name__ == "__main__":
    main()
