from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEATURE_FILE = ROOT / "commuting-regression" / "data_swindon_with_lsoa_commute.csv"
EXTERNAL_FILE = ROOT / "commuting-regression" / "swindon_external_lsoa_commute_summary.csv"
BOUNDARY_FILE = ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
ALL_AREA_DIR = HERE / "all areas simulation"

BOOST = 0.30
SHRINKAGE_JOBS = 250.0
N_FOLDS = 5


def authority_name(label: str) -> str:
    return re.sub(r"\s+\d+[A-Za-z]?$", "", str(label)).strip()


def _distance_km(o_e, o_n, d_e, d_n):
    return np.sqrt((np.asarray(o_e, float) - np.asarray(d_e, float)) ** 2
                   + (np.asarray(o_n, float) - np.asarray(d_n, float)) ** 2) / 1000.0


def load_inputs():
    features = pd.read_csv(FEATURE_FILE).copy()
    features["gva"] = np.exp(features["log_total_GVA_2023"])
    features = features.sort_values("LSOA21CD").reset_index(drop=True)
    sw = features["LSOA21CD"].tolist()
    sw_set = set(sw)

    external_raw = pd.read_csv(EXTERNAL_FILE)
    all_codes = set(external_raw["origin_lsoa"]) | set(external_raw["destination_lsoa"]) | sw_set
    geo = gpd.read_file(BOUNDARY_FILE)
    geo = geo[geo["LSOA21CD"].isin(all_codes)].copy()
    centroids = geo.set_index("LSOA21CD")[["LSOA21NM", "BNG_E", "BNG_N", "LAT", "LONG"]]
    sw_geo = geo[geo["LSOA21CD"].isin(sw_set)].copy()

    external = external_raw.copy()
    external["flow_type"] = external["flow_type"].replace({
        "inbound_to_swindon": "inbound",
        "outbound_from_swindon": "outbound",
    })
    external = external[
        ((external["flow_type"] == "inbound") & external["destination_lsoa"].isin(sw_set))
        | ((external["flow_type"] == "outbound") & external["origin_lsoa"].isin(sw_set))
    ].copy()
    for side, code_col in [("origin", "origin_lsoa"), ("dest", "destination_lsoa")]:
        external[f"{side}_e"] = external[code_col].map(centroids["BNG_E"])
        external[f"{side}_n"] = external[code_col].map(centroids["BNG_N"])
        external[f"{side}_lat"] = external[code_col].map(centroids["LAT"])
        external[f"{side}_lon"] = external[code_col].map(centroids["LONG"])

    missing_geo = external[["origin_e", "dest_e"]].isna().any(axis=1)
    missing_geo_rows = int(missing_geo.sum())
    missing_geo_commuters = int(external.loc[missing_geo, "commuter_count"].sum())
    external = external.loc[~missing_geo].copy()
    external["distance_km"] = _distance_km(
        external["origin_e"], external["origin_n"], external["dest_e"], external["dest_n"]
    )
    external = external.rename(columns={
        "destination_lsoa": "dest_lsoa",
        "destination_lsoa_name": "dest_lsoa_name",
        "commuter_count": "count",
    })

    raw_sw_codes = set(external_raw.loc[
        external_raw["flow_type"].eq("outbound_from_swindon"), "origin_lsoa"
    ]) | set(external_raw.loc[
        external_raw["flow_type"].eq("inbound_to_swindon"), "destination_lsoa"
    ])
    omitted_sw = sorted(raw_sw_codes - sw_set)
    quality = {
        "modelled_swindon_lsoas": len(sw),
        "omitted_swindon_lsoas_without_feature_rows": omitted_sw,
        "external_rows_missing_english_centroids": missing_geo_rows,
        "commuters_in_rows_missing_english_centroids": missing_geo_commuters,
        "external_inbound_modelled_total": int(
            external.loc[external["flow_type"].eq("inbound"), "count"].sum()
        ),
        "external_outbound_modelled_total": int(
            external.loc[external["flow_type"].eq("outbound"), "count"].sum()
        ),
    }
    return features, external_raw, external, centroids, sw_geo, quality


def build_inbound_choice_grid(features, external, centroids):
    sw = features["LSOA21CD"].tolist()
    inbound = external[external["flow_type"].eq("inbound")].copy()
    origins = sorted(inbound["origin_lsoa"].unique())
    grid = pd.MultiIndex.from_product(
        [origins, sw], names=["origin_lsoa", "dest_lsoa"]
    ).to_frame(index=False)

    observed = inbound.groupby(["origin_lsoa", "dest_lsoa"])["count"].sum()
    pair_index = pd.MultiIndex.from_frame(grid[["origin_lsoa", "dest_lsoa"]])
    grid["observed"] = pair_index.map(observed).fillna(0.0).to_numpy(float)

    origin_total = inbound.groupby("origin_lsoa")["count"].sum()
    workplace_jobs = features.set_index("LSOA21CD")["lsoa_workers_at_workplace"]
    grid["O_i"] = grid["origin_lsoa"].map(origin_total).to_numpy(float)
    grid["D_j"] = grid["dest_lsoa"].map(workplace_jobs).to_numpy(float)

    oe = grid["origin_lsoa"].map(centroids["BNG_E"])
    on = grid["origin_lsoa"].map(centroids["BNG_N"])
    de = grid["dest_lsoa"].map(centroids["BNG_E"])
    dn = grid["dest_lsoa"].map(centroids["BNG_N"])
    grid["distance_km"] = _distance_km(oe, on, de, dn).clip(min=0.5)
    return grid, origins, sw


def _design_matrix(grid):
    return sm.add_constant(np.column_stack([
        np.log(grid["O_i"].to_numpy(float)),
        np.log(grid["D_j"].to_numpy(float)),
        np.log(grid["distance_km"].to_numpy(float)),
    ]), has_constant="add")


def _production_constrain(mu, origin_codes, observed):
    origin_id, _ = pd.factorize(origin_codes, sort=False)
    observed_total = np.bincount(origin_id, weights=np.asarray(observed, float))
    model_total = np.bincount(origin_id, weights=np.asarray(mu, float))
    scale = np.divide(
        observed_total,
        model_total,
        out=np.zeros_like(observed_total, dtype=float),
        where=model_total > 0,
    )
    return np.asarray(mu, float) * scale[origin_id]


def fit_inbound_choice_model(grid, origins):
    X = _design_matrix(grid)
    y = grid["observed"].to_numpy(float)

    fold_map = {origin: i % N_FOLDS for i, origin in enumerate(origins)}
    folds = grid["origin_lsoa"].map(fold_map).to_numpy(int)
    oof = np.zeros(len(grid), dtype=float)
    fold_coefficients = []
    for fold in range(N_FOLDS):
        train = folds != fold
        test = ~train
        result = sm.GLM(y[train], X[train], family=sm.families.Poisson()).fit()
        fold_coefficients.append(result.params.tolist())
        mu = np.exp(X[test] @ result.params)
        oof[test] = _production_constrain(
            mu, grid.loc[test, "origin_lsoa"].to_numpy(), y[test]
        )

    final = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    mu = np.exp(X @ final.params)
    baseline = _production_constrain(mu, grid["origin_lsoa"].to_numpy(), y)
    positive = y > 0
    diagnostics = {
        "n_candidate_pairs": int(len(grid)),
        "n_positive_pairs": int(positive.sum()),
        "n_external_origins": int(len(origins)),
        "folds": N_FOLDS,
        "coefficients": {
            "intercept": float(final.params[0]),
            "origin_mass": float(final.params[1]),
            "destination_jobs": float(final.params[2]),
            "distance": float(final.params[3]),
        },
        "fold_coefficients": fold_coefficients,
        "oof_correlation_all_pairs": float(np.corrcoef(y, oof)[0, 1]),
        "oof_correlation_positive_pairs": float(np.corrcoef(y[positive], oof[positive])[0, 1]),
        "oof_mae_positive_pairs": float(np.mean(np.abs(y[positive] - oof[positive]))),
    }
    return np.asarray(final.params, dtype=float), mu, baseline, oof, diagnostics


def estimate_internal_od(features, centroids, distance_coefficient):
    f = features.set_index("LSOA21CD")
    sw = f.index.tolist()
    n = len(sw)
    ext_out = f["lsoa_outbound_from_swindon_count"].to_numpy(float)
    ext_in = f["lsoa_inbound_to_swindon_count"].to_numpy(float)
    diagonal = f["lsoa_same_lsoa_worker_count"].to_numpy(float)
    row_internal = f["lsoa_workplace_commuters"].to_numpy(float) - ext_out
    col_internal_raw = f["lsoa_workers_at_workplace"].to_numpy(float) - ext_in
    row_off = np.maximum(row_internal - diagonal, 0)
    col_off_raw = np.maximum(col_internal_raw - diagonal, 0)
    col_scale = row_off.sum() / col_off_raw.sum()
    col_off = col_off_raw * col_scale

    e = f.index.map(centroids["BNG_E"]).to_numpy(float)
    north = f.index.map(centroids["BNG_N"]).to_numpy(float)
    dist = _distance_km(e[:, None], north[:, None], e[None, :], north[None, :])
    dist = np.clip(dist, 0.5, None)
    seed = np.exp(float(distance_coefficient) * np.log(dist))
    np.fill_diagonal(seed, 0.0)
    matrix = seed
    for _ in range(1000):
        matrix *= np.divide(
            row_off, matrix.sum(axis=1),
            out=np.zeros(n), where=matrix.sum(axis=1) > 0
        )[:, None]
        matrix *= np.divide(
            col_off, matrix.sum(axis=0),
            out=np.zeros(n), where=matrix.sum(axis=0) > 0
        )[None, :]
        row_error = np.max(np.abs(matrix.sum(axis=1) - row_off))
        col_error = np.max(np.abs(matrix.sum(axis=0) - col_off))
        if max(row_error, col_error) < 1e-7:
            break
    matrix[np.diag_indices(n)] = diagonal

    rows, cols = np.indices((n, n))
    internal = pd.DataFrame({
        "origin_lsoa": np.asarray(sw)[rows.ravel()],
        "dest_lsoa": np.asarray(sw)[cols.ravel()],
        "count": matrix.ravel(),
        "distance_km": dist.ravel(),
        "flow_type": "internal_estimated",
        "source_status": "estimated_from_margins",
    })
    internal["origin_lsoa_name"] = internal["origin_lsoa"].map(centroids["LSOA21NM"])
    internal["dest_lsoa_name"] = internal["dest_lsoa"].map(centroids["LSOA21NM"])
    for side, code in [("origin", "origin_lsoa"), ("dest", "dest_lsoa")]:
        internal[f"{side}_lat"] = internal[code].map(centroids["LAT"])
        internal[f"{side}_lon"] = internal[code].map(centroids["LONG"])

    diagnostics = {
        "resident_side_internal_total": float(row_internal.sum()),
        "workplace_side_internal_total_raw": float(col_internal_raw.sum()),
        "workplace_off_diagonal_scale": float(col_scale),
        "fixed_same_lsoa_total": float(diagonal.sum()),
        "ipf_row_max_abs_error": float(
            np.max(np.abs(matrix.sum(axis=1) - row_internal))
        ),
        "ipf_adjusted_col_max_abs_error": float(
            np.max(np.abs(matrix.sum(axis=0) - (col_off + diagonal)))
        ),
        "distance_coefficient_used": float(distance_coefficient),
    }
    return matrix, internal, diagnostics


def gwr_local_elasticity(features, sw_geo):
    frame = sw_geo.merge(
        features[["LSOA21CD", "gva", "lsoa_workers_at_workplace"]],
        on="LSOA21CD", how="inner"
    ).to_crs(27700)
    frame["ln_gva"] = np.log(frame["gva"])
    frame["ln_jobs"] = np.log(frame["lsoa_workers_at_workplace"])
    coords = np.column_stack([frame.geometry.centroid.x, frame.geometry.centroid.y])
    y = frame["ln_gva"].to_numpy(float)
    X = np.column_stack([np.ones(len(frame)), frame["ln_jobs"].to_numpy(float)])
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    n, p = X.shape

    def fit(k):
        beta = np.zeros((n, p))
        trace_s = 0.0
        for i in range(n):
            bandwidth = np.sort(distance[i])[k - 1]
            u = distance[i] / bandwidth
            weights = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
            weights[i] = max(weights[i], 1e-8)
            xtwx = X.T @ (X * weights[:, None])
            inverse_part = np.linalg.solve(
                xtwx + 1e-9 * np.eye(p), (X * weights[:, None]).T
            )
            beta[i] = inverse_part @ y
            trace_s += X[i] @ inverse_part[:, i]
        residual = y - np.einsum("ij,ij->i", X, beta)
        return beta, trace_s, residual

    search = []
    for k in range(25, min(121, n), 5):
        beta, trace_s, residual = fit(k)
        sigma = np.sqrt((residual @ residual) / n)
        denominator = n - 2 - trace_s
        aicc = np.inf if denominator <= 0 or sigma <= 0 else (
            2 * n * np.log(sigma)
            + n * np.log(2 * np.pi)
            + n * (n + trace_s) / denominator
        )
        search.append((k, aicc, trace_s))
    best_k = min(search, key=lambda item: item[1])[0]
    beta, trace_s, residual = fit(best_k)
    frame["elasticity"] = beta[:, 1]
    tss = np.sum((y - y.mean()) ** 2)
    global_beta = np.linalg.lstsq(X, y, rcond=None)[0]
    diagnostics = {
        "adaptive_neighbours": int(best_k),
        "pseudo_r2": float(1 - (residual @ residual) / tss),
        "effective_df": float(trace_s),
        "global_employment_elasticity": float(global_beta[1]),
        "local_elasticity_min": float(frame["elasticity"].min()),
        "local_elasticity_median": float(frame["elasticity"].median()),
        "local_elasticity_max": float(frame["elasticity"].max()),
        "aicc_search": [
            {"k": int(k), "aicc": float(aicc), "trace_s": float(tr)}
            for k, aicc, tr in search
        ],
    }
    return frame, diagnostics


def _scenario_prediction(mu, grid, coefficients, boost_vector, sw):
    dest_index = pd.Categorical(grid["dest_lsoa"], categories=sw).codes
    factor = np.power(1.0 + np.asarray(boost_vector, float)[dest_index], coefficients[2])
    scenario_mu = mu * factor
    scenario = _production_constrain(
        scenario_mu,
        grid["origin_lsoa"].to_numpy(),
        grid["observed"].to_numpy(float),
    )
    return scenario, dest_index


def build_scenario_outputs(features, grid, coefficients, mu, baseline, sw, centroids):
    indexed = features.set_index("LSOA21CD").reindex(sw)
    gva = indexed["gva"].to_numpy(float)
    jobs = indexed["lsoa_workers_at_workplace"].to_numpy(float)
    global_productivity = gva.sum() / jobs.sum()
    adjusted_productivity = (
        gva + SHRINKAGE_JOBS * global_productivity
    ) / (jobs + SHRINKAGE_JOBS)
    raw_productivity = gva / jobs
    dest_index = pd.Categorical(grid["dest_lsoa"], categories=sw).codes

    def evaluate(vector):
        prediction, _ = _scenario_prediction(mu, grid, coefficients, vector, sw)
        delta = prediction - baseline
        delta_jobs = np.bincount(
            dest_index, weights=delta, minlength=len(sw)
        )
        delta_gva = float(delta_jobs @ adjusted_productivity)
        return delta_gva, delta_jobs, prediction

    rows = []
    single_results = {}
    for i, code in enumerate(sw):
        vector = np.zeros(len(sw))
        vector[i] = BOOST
        delta_gva, delta_jobs, _ = evaluate(vector)
        single_results[code] = delta_jobs
        rows.append({
            "LSOA21CD": code,
            "MSOA21CD": indexed.loc[code, "MSOA21CD"],
            "label": centroids.loc[code, "LSOA21NM"],
            "workplace_jobs": jobs[i],
            "gva": gva[i],
            "raw_gva_per_workplace_commuter": raw_productivity[i],
            "adjusted_gva_per_workplace_commuter": adjusted_productivity[i],
            "jobs_pulled_to_target": delta_jobs[i],
            "dGVA_allocation": delta_gva,
            "allocation_pct": delta_gva / gva.sum() * 100,
            "total_inbound_delta": delta_jobs.sum(),
        })
    ranking = pd.DataFrame(rows).sort_values(
        "dGVA_allocation", ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    top_return = ranking.head(2)["LSOA21CD"].tolist()
    top_productivity = [
        sw[i] for i in np.argsort(adjusted_productivity)[::-1][:2]
    ]
    top_jobs = [sw[i] for i in np.argsort(jobs)[::-1][:2]]
    strategy_codes = {
        "LSOA-optimised (top 2 returns)": top_return,
        "2 top adjusted-productivity LSOAs": top_productivity,
        "uniform across 137 LSOAs": None,
        "2 biggest workplace hubs": top_jobs,
    }
    strategies = []
    allocation_rows = []
    for name, codes in strategy_codes.items():
        if codes is None:
            vector = np.full(len(sw), 2 * BOOST / len(sw))
            code_text = "all 137"
        else:
            vector = np.zeros(len(sw))
            for code in codes:
                vector[sw.index(code)] = BOOST
                allocation_rows.append({
                    "strategy": name, "LSOA21CD": code,
                    "label": centroids.loc[code, "LSOA21NM"], "boost": BOOST,
                })
            code_text = "; ".join(codes)
        delta_gva, delta_jobs, _ = evaluate(vector)
        strategies.append({
            "strategy": name,
            "selected_lsoas": code_text,
            "dGVA_allocation": delta_gva,
            "dGVA_allocation_pct": delta_gva / gva.sum() * 100,
            "budget_used": float(vector.sum()),
            "total_inbound_delta": float(delta_jobs.sum()),
        })
    strategies = pd.DataFrame(strategies)
    allocation = pd.DataFrame(allocation_rows)

    top_code = ranking.iloc[0]["LSOA21CD"]
    sensitivity = []
    for boost in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        vector = np.zeros(len(sw))
        vector[sw.index(top_code)] = boost
        delta_gva, delta_jobs, _ = evaluate(vector)
        sensitivity.append({
            "LSOA21CD": top_code,
            "label": centroids.loc[top_code, "LSOA21NM"],
            "boost_pct": boost * 100,
            "dGVA_allocation": delta_gva,
            "dGVA_allocation_pct": delta_gva / gva.sum() * 100,
            "total_inbound_delta": delta_jobs.sum(),
        })
    sensitivity = pd.DataFrame(sensitivity)

    contrast_codes = [top_jobs[0], top_code]
    contrast = ranking.set_index("LSOA21CD").loc[contrast_codes].reset_index()

    best_vector = np.zeros(len(sw))
    for code in top_return:
        best_vector[sw.index(code)] = BOOST
    best_dgva, best_delta_jobs, _ = evaluate(best_vector)
    accounting = pd.DataFrame({
        "LSOA21CD": sw,
        "label": [centroids.loc[code, "LSOA21NM"] for code in sw],
        "baseline_inbound_predicted": np.bincount(
            dest_index, weights=baseline, minlength=len(sw)
        ),
        "scenario_delta_inbound_jobs": best_delta_jobs,
        "adjusted_gva_per_workplace_commuter": adjusted_productivity,
        "dGVA_allocation": best_delta_jobs * adjusted_productivity,
    })

    diagnostics = {
        "boost_fraction": BOOST,
        "productivity_shrinkage_jobs": SHRINKAGE_JOBS,
        "global_gva_per_workplace_commuter": float(global_productivity),
        "top_return_lsoa": top_code,
        "top_return_label": centroids.loc[top_code, "LSOA21NM"],
        "top_return_pct": float(ranking.iloc[0]["allocation_pct"]),
        "top_two_strategy_pct": float(
            strategies.loc[
                strategies["strategy"].eq("LSOA-optimised (top 2 returns)"),
                "dGVA_allocation_pct",
            ].iloc[0]
        ),
        "biggest_hubs_strategy_pct": float(
            strategies.loc[
                strategies["strategy"].eq("2 biggest workplace hubs"),
                "dGVA_allocation_pct",
            ].iloc[0]
        ),
        "top_two_strategy_dGVA": float(best_dgva),
    }
    return ranking, strategies, allocation, sensitivity, contrast, accounting, diagnostics


def build_all_outputs():
    ALL_AREA_DIR.mkdir(parents=True, exist_ok=True)
    features, external_raw, external, centroids, sw_geo, quality = load_inputs()
    grid, origins, sw = build_inbound_choice_grid(features, external, centroids)
    coefficients, mu, baseline, oof, gravity_diag = fit_inbound_choice_model(grid, origins)
    internal_matrix, internal, internal_diag = estimate_internal_od(
        features, centroids, coefficients[3]
    )
    elasticity_geo, gwr_diag = gwr_local_elasticity(features, sw_geo)
    (ranking, strategies, allocation, sensitivity, contrast,
     accounting, scenario_diag) = build_scenario_outputs(
        features, grid, coefficients, mu, baseline, sw, centroids
    )

    external_out = external[[
        "origin_lsoa", "origin_lsoa_name", "dest_lsoa", "dest_lsoa_name",
        "count", "flow_type", "distance_km", "origin_lat", "origin_lon",
        "dest_lat", "dest_lon",
    ]].copy()
    external_out["source_status"] = "observed"
    combined = pd.concat([internal, external_out], ignore_index=True, sort=False)

    external_out.to_csv(HERE / "lsoa_external_flows_geo.csv", index=False)
    internal.to_csv(HERE / "lsoa_internal_od_estimated.csv", index=False)
    combined.to_csv(HERE / "lsoa_od_flows_geo.csv", index=False)

    positive = grid["observed"].gt(0)
    validation = grid.loc[positive, [
        "origin_lsoa", "dest_lsoa", "observed", "distance_km", "O_i", "D_j"
    ]].copy()
    validation["predicted_oof"] = oof[positive]
    validation["predicted_baseline"] = baseline[positive]
    validation.to_csv(HERE / "lsoa_gravity_validation.csv", index=False)

    elasticity = pd.DataFrame(elasticity_geo.drop(columns="geometry"))
    elasticity.to_csv(HERE / "lsoa_local_elasticity.csv", index=False)
    elasticity.to_csv(HERE / "gwr_local_elasticity.csv", index=False)
    ranking.to_csv(ALL_AREA_DIR / "all_zone_ranking.csv", index=False)
    strategies.to_csv(ALL_AREA_DIR / "optimisation_result.csv", index=False)
    allocation.to_csv(ALL_AREA_DIR / "optimisation_allocation.csv", index=False)
    sensitivity.to_csv(ALL_AREA_DIR / "sensitivity_curve.csv", index=False)
    contrast.to_csv(HERE / "scenario_lsoa_contrast.csv", index=False)
    accounting.to_csv(HERE / "closed_loop_gva_accounting.csv", index=False)
    accounting.to_csv(HERE / "closed_loop_gva_impact.csv", index=False)

    diagnostics = {
        "data_quality": quality,
        "gravity_model": gravity_diag,
        "internal_od_reconstruction": internal_diag,
        "gwr": gwr_diag,
        "scenario": scenario_diag,
    }
    (HERE / "lsoa_model_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(
        f"LSOA outputs built: {len(sw)} Swindon LSOAs; "
        f"OOF r={gravity_diag['oof_correlation_all_pairs']:.3f}; "
        f"top={scenario_diag['top_return_label']} "
        f"({scenario_diag['top_return_pct']:+.2f}%)"
    )
    return diagnostics


if __name__ == "__main__":
    build_all_outputs()
