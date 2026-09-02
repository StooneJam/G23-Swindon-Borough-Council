from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_CSV = ROOT / "commuting-data" / "data_swindon_with_lsoa_commute.csv"

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

W = {
    "log_voa_rv_2023": 0, "rv_per_working_age": 0, "rv_per_employee": 0,
    "sme_density": 0.5, "qualification_index": 0.5, "firm_size_diversity": 0.5,
    "sme_qual_interaction": 0.5, "employment_quality": 0.5,
    "modern_sector_leverage": 0.5, "asset_growth_diversity": 0.5,
    "lsoa_total_employed": 0, "lsoa_home_or_no_fixed_count": 0,
    "lsoa_workplace_commuters": 0, "lsoa_same_lsoa_worker_count": 0,
    "lsoa_workers_at_workplace": 0, "lsoa_inbound_worker_count": 0,
    "lsoa_outbound_from_swindon_count": 0, "lsoa_inbound_to_swindon_count": 0,
    "lsoa_home_or_no_fixed_share": 0.5,
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
    wv = pd.Series(ws).reindex(FEATS).fillna(0)
    L = absphi.mul(wv, axis=1).sum(axis=1)
    score = need.rank(pct=True) * L.rank(pct=True)
    return score.where(need > 0, 0.0)


def topk_overlap(base, alt, codes, k):
    b = set(codes[np.argsort(-np.asarray(base))[:k]])
    a = set(codes[np.argsort(-np.asarray(alt))[:k]])
    return len(b & a) / k


def rankify(score: pd.Series) -> pd.Series:
    """1 = highest IPI. Ties share the min rank."""
    return score.rank(ascending=False, method="min")


def rank_shift_stats(base: pd.Series, alt: pd.Series) -> dict:
    diff = (rankify(base) - rankify(alt)).abs()
    return {"median_abs_rank_change": float(diff.median()), "_rank_diff": diff}


def ipi_zscore(need, absphi, ws):
    """Raw z-score product, kept only as an index-construction diagnostic, NOT a
    standardisation-robustness test: z is centred on 0, so two below-average values
    (need and leverage both negative) multiply to a large *positive* score, i.e. a
    low-need/low-leverage area can outrank a high-need/high-leverage one. A Spearman
    drop under this variant reflects that sign-flip artefact, not outlier sensitivity
    -- use `ipi_minmax_winsor()` for the actual standardisation-swap comparison."""
    wv = pd.Series(ws).reindex(FEATS).fillna(0)
    L = absphi.mul(wv, axis=1).sum(axis=1)

    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    score = z(need) * z(L)
    return score.where(need > 0, 0.0)


def ipi_minmax_winsor(need, absphi, ws, q_lo: float = 0.05, q_hi: float = 0.95):
    """Standardisation-swap variant of `ipi()`: winsorised min-max instead of
    percentile rank. Both `need` and leverage are clipped to [q_lo, q_hi] quantiles
    then rescaled to [0, 1], so -- like percentile rank -- higher need/leverage still
    maps to a higher score (no sign-flip), and the product stays comparable to the
    baseline. This isolates whether outliers (rather than the choice of rank vs.
    continuous scale) are driving the ranking."""
    wv = pd.Series(ws).reindex(FEATS).fillna(0)
    L = absphi.mul(wv, axis=1).sum(axis=1)

    def winsor_minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(q_lo), s.quantile(q_hi)
        clipped = s.clip(lo, hi)
        span = hi - lo
        return (clipped - lo) / span if span > 0 else clipped * 0.0

    score = winsor_minmax(need) * winsor_minmax(L)
    return score.where(need > 0, 0.0)


def perturb_tier(ws: dict, tier_value: float, pct: float) -> dict:
    base_sum = sum(v for v in ws.values() if v > 0)
    new = dict(ws)
    for k, v in ws.items():
        if v == tier_value:
            new[k] = v * (1 + pct)
    new_sum = sum(v for v in new.values() if v > 0)
    if new_sum > 0:
        scale = base_sum / new_sum
        new = {k: (v * scale if v > 0 else v) for k, v in new.items()}
    return new


def lever_for_target(grid, gva, cur, target):
    grid = np.asarray(grid, float)
    gva = np.asarray(gva, float)
    direction = float(np.sign(np.polyfit(grid, gva, 1)[0]))
    if direction == 0:
        return None, 0.0
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(grid * direction, gva)
    xs = np.linspace(grid[0], grid[-1], 500)
    ys = iso.predict(xs * direction)
    side = (xs - cur) * direction >= 0
    hit = np.where(side & (ys >= target))[0]
    if len(hit) == 0:
        return None, direction
    return float(xs[hit[np.argmin(np.abs(xs[hit] - cur))]]), direction
