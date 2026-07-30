from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent

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

W = {
    "log_voa_rv_2023": 0, "rv_per_working_age": 0, "rv_per_employee": 0,
    "msoa_total_employed": 0, "msoa_workplace_commuters": 0,
    "msoa_workers_at_workplace": 0, "msoa_inbound_worker_count": 0,
    "sme_density": 0.5, "qualification_index": 0.5, "firm_size_diversity": 0.5,
    "sme_qual_interaction": 0.5, "employment_quality": 0.5,
    "modern_sector_leverage": 0.5, "asset_growth_diversity": 0.5,
    "msoa_out_commute_share": 1, "msoa_in_commute_share": 1,
    "msoa_same_msoa_work_share": 1, "msoa_local_worker_share": 1, "msoa_wfh_share": 0.5,
}


def ipi(need, absphi, ws):
    wv = pd.Series(ws).reindex(FEATS).fillna(0)
    L = absphi.mul(wv, axis=1).sum(axis=1)
    score = need.rank(pct=True) * L.rank(pct=True)
    return score.where(need > 0, 0.0)


def main():
    need_df = pd.read_csv(HERE / "ipi_need.csv")
    phi = pd.read_csv(HERE / "shap_local_engineered_commute.csv")
    phi = need_df.merge(phi, on="LSOA21CD")

    need = phi["need"]
    absphi = phi[FEATS].abs()
    w = pd.Series(W).reindex(FEATS).fillna(0)

    base = ipi(need, absphi, W)
    levered = [c for c in FEATS if w[c] > 0]
    bottleneck = phi[levered].idxmin(axis=1)

    out = pd.DataFrame({
        "LSOA21CD": phi["LSOA21CD"], "MSOA21CD": phi["MSOA21CD"],
        "need": need, "IPI": base, "bottleneck": bottleneck,
    })

    schemes = {
        "indirect_low": {**W, **{c: 0.3 for c in ENG if W[c] == 0.5}},
        "commute_down": {**W, **{c: 0.5 for c in COMMUTE if W[c] == 1}},
    }
    for name, ws in schemes.items():
        alt = ipi(need, absphi, ws)
        out[f"IPI_{name}"] = alt
        print(f"{name}: spearman={spearmanr(base, alt).correlation:.3f}")

    out = out.sort_values("IPI", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out.to_csv(HERE / "ipi_tabpfn.csv", index=False)
    print(out[["rank", "LSOA21CD", "IPI", "bottleneck"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
