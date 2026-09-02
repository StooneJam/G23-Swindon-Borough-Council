from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ipi_common as C


def main():
    need_df = pd.read_csv(C.HERE / "ipi_need.csv")
    phi = pd.read_csv(C.HERE / "shap_local.csv")
    phi = need_df.merge(phi, on="LSOA21CD")

    need = phi["need"]
    absphi = phi[C.FEATS].abs()
    w = pd.Series(C.W).reindex(C.FEATS).fillna(0)

    base = C.ipi(need, absphi, C.W)
    levered = [c for c in C.FEATS if w[c] > 0]
    bottleneck = phi[levered].idxmin(axis=1)

    out = pd.DataFrame({
        "LSOA21CD": phi["LSOA21CD"], "MSOA21CD": phi["MSOA21CD"],
        "need": need, "IPI": base, "bottleneck": bottleneck,
        "bottleneck_label": bottleneck.map(lambda c: C.LABELS.get(c, c)),
    })

    schemes = {
        "indirect_low": {**C.W, **{c: 0.3 for c in C.ENG if C.W[c] == 0.5}},
        "commute_down": {**C.W, **{c: 0.5 for c in C.COMMUTE if C.W[c] == 1}},
        "uniform_lever": {**C.W, **{c: 1.0 for c in C.FEATS if C.W[c] > 0}},
    }
    codes = phi["LSOA21CD"].to_numpy()
    sens = []
    for name, ws in schemes.items():
        alt = C.ipi(need, absphi, ws)
        out[f"IPI_{name}"] = alt
        sp = spearmanr(base, alt).correlation
        ov5 = C.topk_overlap(base.to_numpy(), alt.to_numpy(), codes, 5)
        ov10 = C.topk_overlap(base.to_numpy(), alt.to_numpy(), codes, 10)
        sens.append({"scheme": name, "spearman_vs_base": round(sp, 3),
                     "top5_overlap": round(ov5, 2), "top10_overlap": round(ov10, 2)})
        print(f"{name}: spearman={sp:.3f}  top5={ov5:.2f}  top10={ov10:.2f}", flush=True)

    pd.DataFrame(sens).to_csv(C.HERE / "ipi_sensitivity.csv", index=False)

    out = out.sort_values("IPI", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out.to_csv(C.HERE / "ipi_tabpfn_lsoa.csv", index=False)
    print("\nTop-10:\n",
          out[["rank", "LSOA21CD", "IPI", "bottleneck_label"]].head(10).to_string(index=False),
          flush=True)
    print("\nsaved ipi_tabpfn_lsoa.csv, ipi_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
