from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tabpfn import TabPFNRegressor
import warnings

warnings.filterwarnings("ignore")

def load_env(path: str = None) -> None:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

PRICES = {
    "gpt-4.1": (2.00, 8.00),
}

@dataclass
class Config:
    data_path: str = "D:/github/G23-Swindon-Borough-Council/data preprocessing+EDA/features.csv"
    train_path: str = "D:/github/G23-Swindon-Borough-Council/data preprocessing+EDA/train_updated.csv"
    test_path: str = "D:/github/G23-Swindon-Borough-Council/data preprocessing+EDA/test_updated.csv"
    id_col: str = "LSOA21CD"
    group_col: str = "MSOA21CD"
    id_cols: tuple = ("LSOA21CD", "MSOA21CD")
    targets: tuple = ("log_total_GVA_2023", "log_gva_per_worker_2023", "log_GVA_2025_predicted")
    exclude_cols: tuple = (
        "LSOA21CD", "LSOA21NM", "MSOA21CD", "MSOA21NM",
        "Urban_rura", "LSOA GVA Estimates (millions)",
    )
    max_features_to_keep: int = 11
    out_dir: str = "output"

    project_goal: str = (
        "Our research investigates how Small and Medium Enterprise (SME) composition, "
        "economic resilience, and employment characteristics shape local Gross Value Added (GVA) "
        "in Swindon, UK. The target variable is {target_name}. "
        "You may create new features by combining, transforming, or aggregating existing ones. "
        "Keep the total number of features (original + new) at most {max_features_to_keep}."
    )
    data_dictionary_path: str = "data_dictionary.md"
    data_dictionary_text: str = ""
    n_col_samples: int = 10
    n_eval_splits: int = 1
    valid_size: float = 0.33
    n_oof_splits: int = 5
    max_iterations: int = 10
    llm_model: str = "gpt-4.1"
    temperature: float = 0.7
    random_state: int = 42

    def __post_init__(self):
        if self.data_dictionary_path and os.path.exists(self.data_dictionary_path) and not self.data_dictionary_text:
            with open(self.data_dictionary_path, 'r', encoding='utf-8') as f:
                self.data_dictionary_text = f.read()
        if not self.data_dictionary_text:
            self.data_dictionary_text = "Socio-economic indicators for UK LSOAs."

def load_data(cfg: Config):
    df = pd.read_csv(cfg.data_path, low_memory=False)
    upd = pd.concat([pd.read_csv(cfg.train_path), pd.read_csv(cfg.test_path)], ignore_index=True)
    df = df.merge(upd[[cfg.id_col, *cfg.targets]], on=cfg.id_col, how="left")
    df = df.dropna(subset=[*cfg.targets, cfg.group_col]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in cfg.exclude_cols and c not in cfg.targets]
    print(f"Loaded {len(df)} rows, {len(feat_cols)} features, {len(cfg.targets)} targets.")
    return df, feat_cols

def describe_columns(X: pd.DataFrame, n: int) -> str:
    lines = []
    for c in X.columns:
        s = X[c]
        nan = 100 * s.isna().mean()
        samples = s.dropna().head(n).tolist()
        samples_str = [str(v)[:60] + "..." if len(str(v)) > 60 else str(v) for v in samples]
        lines.append(f"{c} ({s.dtype}): NaN-freq [{nan:.1f}%], Samples {samples_str}")
    return "\n".join(lines)

def build_hybrid_prompt(cfg: Config, col_desc: str, n_rows: int, current_n_features: int,
                        accepted: list, last_error: str, best_r2: float, target_name: str) -> str:
    history = "\n\n".join(accepted)
    error = (f'\nThe previous code failed with: "{last_error}". '
             "Fix it in the next code block.\n") if last_error else ""

    dict_text = cfg.data_dictionary_text[:3000]
    project_goal_filled = cfg.project_goal.format(target_name=target_name,
                                                  max_features_to_keep=cfg.max_features_to_keep)

    return f"""The dataframe `df` is loaded and in memory. The target variable ({target_name}) is NOT in `df`; it is stored separately.

## PROJECT RESEARCH GOAL
{project_goal_filled}

## DATA DICTIONARY
{dict_text}

## Current columns in `df` (total {current_n_features} features):
{col_desc}

Number of samples (rows): {n_rows}

## Your task
You can:
1. Create new features by transforming, combining, or aggregating existing columns (e.g., ratios, differences, log, bins, interactions).
2. Drop existing columns that are redundant or irrelevant.
3. Keep the total number of columns (original + new) at most {cfg.max_features_to_keep}.

Output Python code that modifies `df` (add new columns, drop old ones).
Include comments explaining the usefulness of each added feature.
Do NOT reference the target column.
Ensure all new column names are unique and do not contain special characters.

Example:
```python
df['log_voa'] = np.log1p(df['VOA_total_RV_million_2023'])
df['gva_per_capita_proxy'] = df['VOA_total_RV_million_2023'] / (df['Mid-2024 population'] + 1)
df.drop(columns=['VOA_total_RV_million_2023'], inplace=True)
```end

{history}
{error}
Current best holdout R2: {best_r2:.4f}
Codeblock:"""

def extract_code(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def run_code(code: str, X: pd.DataFrame) -> pd.DataFrame:
    ns = {"df": X.copy(), "pd": pd, "np": np}
    exec(code, ns)
    new_df = ns["df"]
    if not isinstance(new_df, pd.DataFrame):
        raise ValueError("Code did not produce a pandas DataFrame")
    return new_df

def as_matrix(X: pd.DataFrame) -> np.ndarray:
    M = X.copy()
    for c in M.columns:
        if not pd.api.types.is_numeric_dtype(M[c]):
            M[c] = pd.factorize(M[c])[0]
    return M.to_numpy()

def evaluate(cfg: Config, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    gss = GroupShuffleSplit(n_splits=cfg.n_eval_splits, test_size=cfg.valid_size, random_state=cfg.random_state)
    M = as_matrix(X)
    scores = []
    for tr, va in gss.split(M, y, groups):
        model = TabPFNRegressor()
        model.fit(M[tr], y[tr])
        scores.append(r2_score(y[va], model.predict(M[va])))
    return float(np.mean(scores))

def oof_predict(cfg: Config, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict:
    M = as_matrix(X)
    pred = np.empty(len(y))
    gkf = GroupKFold(n_splits=cfg.n_oof_splits)
    for tr, va in gkf.split(M, y, groups):
        model = TabPFNRegressor()
        model.fit(M[tr], y[tr])
        pred[va] = model.predict(M[va])
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE": float(mean_absolute_percentage_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }

def call_llm(cfg: Config, prompt: str, usage: dict) -> str:
    client = OpenAI()
    r = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.temperature,
    )
    usage["in"] += r.usage.prompt_tokens
    usage["out"] += r.usage.completion_tokens
    return r.choices[0].message.content

def fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna('missing')
    return X

def run_caafe_for_target(cfg: Config, df: pd.DataFrame, feat_cols: list, target_name: str, usage: dict):
    groups = df[cfg.group_col].to_numpy()
    y = df[target_name].to_numpy()
    X = fill_missing(df[feat_cols].copy())

    accepted, history, last_error = [], [], ""
    current_X = X.copy()
    current_desc = describe_columns(X, cfg.n_col_samples)
    best_subset = {"cv_r2": -np.inf, "features": None, "oof_metrics": None, "iteration": None}

    for it in range(1, cfg.max_iterations + 1):
        prompt = build_hybrid_prompt(cfg, current_desc, len(current_X), current_X.shape[1],
                                     accepted, last_error,
                                     best_subset["cv_r2"] if best_subset["cv_r2"] > -np.inf else 0,
                                     target_name)
        raw = call_llm(cfg, prompt, usage)
        code = extract_code(raw)
        try:
            X_new = run_code(code, current_X)
            if X_new.shape[1] > cfg.max_features_to_keep:
                raise ValueError(f"Feature count {X_new.shape[1]} > {cfg.max_features_to_keep}")
            score = evaluate(cfg, X_new, y, groups)
            last_error = ""
            keep = score > best_subset["cv_r2"]
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            history.append({"status": "error", "score": None, "code": code, "features": None})
            print(f"[{target_name}] iteration {it}/{cfg.max_iterations}: error {last_error}")
            continue

        status = "accepted" if keep else "rejected"
        history.append({"status": status, "score": round(score, 4), "code": code, "features": list(X_new.columns)})
        print(f"[{target_name}] iteration {it}/{cfg.max_iterations}: R2 {score:.4f} ({status})")
        if keep:
            best_subset = {"cv_r2": score, "features": list(X_new.columns), "iteration": it, "oof_metrics": None}
            current_X, current_desc = X_new, describe_columns(X_new, cfg.n_col_samples)
            accepted.append(code)

    if best_subset["features"] is not None:
        best_X_df = fill_missing(current_X[best_subset["features"]].copy())
    else:
        best_subset["features"] = list(X.columns)
        best_X_df = X
    oof_metrics = oof_predict(cfg, best_X_df, y, groups)
    best_subset["oof_metrics"] = oof_metrics

    ids = df[[c for c in cfg.id_cols if c in df.columns]].reset_index(drop=True)
    final_df = pd.concat([ids, best_X_df.reset_index(drop=True),
                          pd.Series(y, name=target_name).reset_index(drop=True)], axis=1)
    final_df.to_csv(os.path.join(cfg.out_dir, f"caafe_{target_name}_features.csv"), index=False, encoding='utf-8')
    return best_subset, history

def write_log(cfg: Config, target_name: str, history: list, best_subset: dict) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": cfg.llm_model, "target": target_name,
        "best_subset": best_subset, "history": history,
    }
    with open(os.path.join(cfg.out_dir, f"caafe_{target_name}_log.json"), "w", encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def run_hybrid_caafe(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    df, feat_cols = load_data(cfg)
    usage = {"in": 0, "out": 0}
    results = {}
    for target_name in cfg.targets:
        best_subset, history = run_caafe_for_target(cfg, df, feat_cols, target_name, usage)
        write_log(cfg, target_name, history, best_subset)
        results[target_name] = best_subset

    for target_name in cfg.targets:
        m = results[target_name]["oof_metrics"]
        print(f"Target: {target_name}")
        print(f"  MAE  {m['MAE']:.4f}")
        print(f"  RMSE {m['RMSE']:.4f}")
        print(f"  MAPE {m['MAPE']:.4f}")
        print(f"  R2   {m['R2']:.4f}")

    p_in, p_out = PRICES.get(cfg.llm_model, (0.0, 0.0))
    cost = usage["in"] / 1e6 * p_in + usage["out"] / 1e6 * p_out
    print(f"Tokens in/out: {usage['in']}/{usage['out']} | Cost: ${cost:.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--group_col", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--max_features_to_keep", type=int, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--n_eval_splits", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config()
    if args.data_path:
        config.data_path = args.data_path
    if args.group_col:
        config.group_col = args.group_col
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.max_features_to_keep:
        config.max_features_to_keep = args.max_features_to_keep
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.n_eval_splits:
        config.n_eval_splits = args.n_eval_splits

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable or add it to LLM/.env.")

    run_hybrid_caafe(config)
