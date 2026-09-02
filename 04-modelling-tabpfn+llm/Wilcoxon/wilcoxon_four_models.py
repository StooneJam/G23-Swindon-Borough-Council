from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data preprocessing+EDA"

FEATURES = [
    "Urban_rura_Urban",
    "LU_pct_micro_2025_msoa",
    "LU_pct_large_2025_msoa",
    "log_enterprises_per_1k_residents_2025",
    "turnover_diversity_1-HHI_2025_msoa",
    "LU_diversity_1-HHI_2025",
    "share_enterprises_kibs_2025_msoa",
    "emp_rate",
    "full_time_share",
    "log_Mid-2024 population",
    "IMD_decile",
]

TARGETS = [
    "log_total_GVA_2023",
    "log_gva_per_worker_2023",
    "log_GVA_2025_predicted",
]

MODEL_ORDER = ["Linear Regression", "LightGBM", "XGBoost", "TabPFN"]
SEED = 42

LIGHTGBM_PARAMS = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    bagging_freq=0,
    feature_fraction=0.8,
    seed=SEED,
    feature_fraction_seed=SEED,
    bagging_seed=SEED,
    verbosity=-1,
)

XGBOOST_PARAMS = {
    "log_total_GVA_2023": dict(
        colsample_bytree=0.8, gamma=0.5, learning_rate=0.05, max_depth=5,
        min_child_weight=5, n_estimators=200, reg_alpha=0.05, reg_lambda=2,
        subsample=0.5,
    ),
    "log_gva_per_worker_2023": dict(
        colsample_bytree=0.7, gamma=0.5, learning_rate=0.005, max_depth=7,
        min_child_weight=1, n_estimators=1100, reg_alpha=0, reg_lambda=0.5,
        subsample=0.7,
    ),
    "log_GVA_2025_predicted": dict(
        colsample_bytree=0.7, gamma=0.2, learning_rate=0.02, max_depth=6,
        min_child_weight=7, n_estimators=600, reg_alpha=0.01, reg_lambda=1.5,
        subsample=0.5,
    ),
}


class NativeLightGBMRegressor:
    def fit(self, x_train, y_train):
        dataset = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
        self.booster = lgb.train(
            LIGHTGBM_PARAMS,
            dataset,
            num_boost_round=500,
        )
        return self

    def predict(self, x_test):
        return self.booster.predict(x_test)


def model_set(target):
    return {
        "Linear Regression": LinearRegression(),
        "LightGBM": NativeLightGBMRegressor(),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
            **XGBOOST_PARAMS[target],
        ),
        "TabPFN": TabPFNRegressor.create_default_for_version(ModelVersion.V3),
    }


def point_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def error_frame(y_true, predictions, groups=None):
    frame = pd.DataFrame({
        name: np.abs(np.asarray(y_true) - np.asarray(predictions[name]))
        for name in MODEL_ORDER
    })
    if groups is not None:
        frame["group"] = np.asarray(groups)
        frame = frame.groupby("group", sort=True)[MODEL_ORDER].mean()
    return frame


def rank_biserial(difference):
    nonzero = difference[difference != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for position, index in enumerate(order):
        candidate = (len(p_values) - position) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def paired_tests(errors, target, level):
    rows = []
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        improvement = (errors[model_b] - errors[model_a]).to_numpy()
        nonzero = improvement[improvement != 0]
        if len(nonzero) == 0:
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        rows.append({
            "target": target,
            "analysis_level": level,
            "n_pairs": int(len(errors)),
            "model_a": model_a,
            "model_b": model_b,
            "model_a_mean_absolute_error": float(errors[model_a].mean()),
            "model_b_mean_absolute_error": float(errors[model_b].mean()),
            "mean_paired_improvement_a_over_b": float(improvement.mean()),
            "median_paired_improvement_a_over_b": float(np.median(improvement)),
            "wilcoxon_statistic": statistic,
            "p_value_raw": p_value,
            "rank_biserial_a_over_b": float(rank_biserial(improvement)),
            "lower_mean_error": model_a if improvement.mean() > 0 else model_b,
        })
    adjusted = holm_adjust([row["p_value_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adjusted)
        row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
    return rows


def main():
    train = pd.read_csv(DATA / "train_updated.csv")
    test = pd.read_csv(DATA / "test_updated.csv")
    if len(train) != 900 or len(test) != 221:
        raise ValueError(f"Expected the fixed 900/221 split, found {len(train)}/{len(test)}")

    x_train = train[FEATURES].to_numpy()
    x_test = test[FEATURES].to_numpy()
    metric_rows = []
    wilcoxon_rows = []
    friedman_rows = []
    prediction_output = test[["LSOA21CD", "MSOA21CD"]].copy()

    for target in TARGETS:
        print(f"\n=== {target} ===", flush=True)
        y_train = train[target].to_numpy()
        y_test = test[target].to_numpy()
        predictions = {}

        for name, model in model_set(target).items():
            model.fit(x_train, y_train)
            predictions[name] = model.predict(x_test)
            values = point_metrics(y_test, predictions[name])
            metric_rows.append({"target": target, "model": name, **values})
            prediction_output[f"{target}__{name.replace(' ', '_')}"] = predictions[name]
            print(name, values, flush=True)

        for level, groups in (
            ("MSOA_primary", test["MSOA21CD"].to_numpy()),
            ("LSOA_sensitivity", None),
        ):
            errors = error_frame(y_test, predictions, groups)
            omnibus = friedmanchisquare(*(errors[name].to_numpy() for name in MODEL_ORDER))
            friedman_rows.append({
                "target": target,
                "analysis_level": level,
                "n_blocks": int(len(errors)),
                "friedman_statistic": float(omnibus.statistic),
                "p_value": float(omnibus.pvalue),
            })
            wilcoxon_rows.extend(paired_tests(errors, target, level))

    metrics = pd.DataFrame(metric_rows)
    wilcoxon_results = pd.DataFrame(wilcoxon_rows)
    friedman_results = pd.DataFrame(friedman_rows)

    metrics.to_csv(HERE / "four_model_point_metrics.csv", index=False)
    wilcoxon_results.to_csv(HERE / "four_model_wilcoxon.csv", index=False)
    friedman_results.to_csv(HERE / "four_model_friedman.csv", index=False)
    prediction_output.to_csv(HERE / "four_model_test_predictions.csv", index=False)

    print("\n=== Friedman omnibus tests ===")
    print(friedman_results.to_string(index=False))
    print("\n=== Pairwise Wilcoxon tests ===")
    print(wilcoxon_results.to_string(index=False))
    print("\nSaved four_model_point_metrics.csv, four_model_wilcoxon.csv, "
          "four_model_friedman.csv and four_model_test_predictions.csv")


if __name__ == "__main__":
    main()
