# XGBoost IPI Pipeline — Method Notes

Full XGBoost run of the SHAP → IPI → intervention-investment → sensitivity workflow
on the **LSOA-commute + LLM-feature** dataset
(`commuting-regression/data_swindon_with_lsoa_commute.csv`), target
`log_total_GVA_2023`, Swindon-only (137 LSOAs, 25 predictors = 10 engineered + 15
LSOA commuting). Built to mirror `SHAP/TabPFN/` so the two models can be compared
like-for-like. **No plotting** — data products only.

Files: `ipi_common.py` (config/weights/formulas) · `ipi_shap.py` (stage 1) ·
`ipi_build.py` (stage 2) · `ipi_investment.py` (stage 3).

Each method below is documented as: **Why → Theory/formula → Evaluation → Expected result**.

---

## 1. Predictive model — XGBoost regressor

- **Why.** The IPI needs a fitted model whose per-area predictions and attributions
  are trustworthy. On small/medium tabular data with mixed, partly weak signals,
  gradient-boosted trees are a strong, well-understood default, and XGBoost gives an
  **exact, fast** SHAP explainer (TreeSHAP) — the practical reason to include it
  alongside TabPFN.
- **Theory / formula.** XGBoost fits an additive ensemble
  $\hat{y}(x)=\sum_{t=1}^{T} f_t(x)$, each $f_t$ a regression tree, by greedily
  minimising a regularised objective
  $\mathcal{L}=\sum_i \ell(y_i,\hat{y}_i)+\sum_t \Omega(f_t)$ with squared-error loss
  and $\Omega(f)=\gamma T+\tfrac{1}{2}\lambda\lVert w\rVert^2$. Split gains use the
  second-order (gradient/Hessian) approximation.
- **Evaluation.** 5-fold `RandomizedSearchCV` (R²) selects hyper-parameters once on
  the full sample; generalisation is judged by out-of-fold R², with in-sample R² as a
  (optimistic) sanity check. Params cached to `xgb_best_params.json` so every stage
  uses the same configuration.
- **Expected result.** Moderate fit on 137 rows; commute features add modest lift.
  *Obtained:* CV R² ≈ 0.58, OOF R² ≈ 0.66, in-sample R² ≈ 0.86.

## 2. Feature attribution — exact TreeSHAP (global + local)

- **Why.** A council must know *why* the model ranks an area as it does. SHAP gives a
  single, axiomatic attribution that is additive per prediction, so local
  contributions can be aggregated into a global ranking. XGBoost admits the **exact**
  TreeSHAP estimator — a concrete advantage over TabPFN's slower model-agnostic
  explainer.
- **Theory / formula.** SHAP assigns each feature the Shapley value
  $\phi_{ij}=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{j\})-f(S)\big]$.
  It is the unique additive attribution satisfying local accuracy, missingness and
  consistency; on the log scale contributions are additive, so
  $\sum_j \phi_{ij}=\hat{y}_i-\mathbb{E}[\hat{y}]$. TreeSHAP computes these exactly for
  tree ensembles in polynomial time.
- **Evaluation.** Local check: per-area $\sum_j\phi_{ij}+\text{base}$ reconstructs the
  prediction. Global view = mean$_i|\phi_{ij}|$ per feature (`shap_global.csv`); local
  matrix saved as `shap_local.csv`.
- **Expected result.** A feature ranking dominated by structural/scale terms, with the
  commuting shares surfacing as the leverable signal used downstream.

## 3. Local need — out-of-fold underperformance

- **Why.** Priority should target areas doing **worse than comparable areas predict**,
  not merely low-GVA areas. Using the model's own expectation as the benchmark
  isolates unrealised potential.
- **Theory / formula.** $\text{need}_i=\max\!\big(0,\ \hat{y}_i^{(-k)}-y_i\big)$, where
  $\hat{y}_i^{(-k)}$ is the prediction for area $i$ when it is held out (5-fold OOF) and
  $y_i$ is observed log-GVA. Out-of-fold prediction is used so need reflects genuine
  shortfall, not in-sample fit (which is optimistically small). Areas at/above
  expectation get need $=0$ and are not intervention targets.
- **Evaluation.** OOF R² gauges how meaningful the benchmark is; the count of
  need $>0$ areas bounds the eligible priority set.
- **Expected result.** A minority of areas carry positive need. *Obtained:* 81 of 137.

## 4. Intervention Priority Index (IPI)

- **Why.** Attribution alone does not tell a council where to act. IPI fuses *where the
  gap is* (need) with *how actionable the local drivers are* (weighted SHAP) into one
  transparent ranking plus a per-area bottleneck.
- **Theory / formula.**
  $L_i=\sum_j w_j|\phi_{ij}|$ (actionable leverage), and
  $\text{IPI}_i=\text{rank}_\%(\text{need}_i)\times\text{rank}_\%(L_i)$ if
  $\text{need}_i>0$, else $0$. Percentile ranks make the index scale/skew-invariant;
  the product means an area scores highly only when it is **both** underperforming and
  has an actionable profile. Bottleneck
  $b_i=\arg\min_{j:\,w_j>0}\phi_{ij}$ = the leverable feature most dragging the area's
  prediction down.
- **Evaluation.** Face validity of the ranking + bottleneck mix; robustness under
  re-weighting (§6).
- **Expected result.** A ranked `ipi_xgboost.csv` (137 areas) with a small high-priority
  head. *Obtained:* top bottlenecks are Local worker share and Firm-size diversity.

## 5. Actionability weights — policy judgement, not learned

- **Why.** The data holds no record of what a council *can move*; letting attribution
  magnitude alone drive priority would elevate levers no one can pull. Weights encode
  controllability explicitly.
- **Theory / mapping.** Three tiers, $w_j\in\{0,0.5,1\}$: **0** structural/scale &
  headcount counts (rateable value, employed/commuter/worker counts); **0.5**
  semi-actionable (SME density, qualification, firm/asset diversity, employment
  quality, home/no-fixed share); **1** direct commuting-share levers (in/out/same-area/
  local-worker/outbound/inbound shares). Zeroing headcount counts also strips
  size-proxy attributions from $L_i$.
- **Evaluation.** Because the mapping is normative, its influence **must be measured**
  (§6), not assumed.
- **Expected result.** Ranking driven by genuinely actionable levers rather than
  scale.

## 6. Sensitivity / robustness analysis

- **Why.** The weights are the one subjective input; a defensible index must show the
  ranking does not hinge on a single arbitrary choice.
- **Theory / formula.** Re-compute IPI under alternative weightings —
  `indirect_low` (engineered 0.5→0.3), `commute_down` (commute 1→0.5),
  `uniform_lever` (all levers =1) — spanning commute-led to structure-led stances.
  Compare each to base by **Spearman rank correlation** $\rho$ (whole-ordering
  stability) and **top-$k$ overlap** $|\text{top}_k^{base}\cap\text{top}_k^{alt}|/k$ for
  $k=5,10$ (shortlist stability).
- **Evaluation.** High $\rho$ and high top-$k$ overlap ⇒ conclusions are weight-robust;
  areas that flip are flagged for council confirmation.
- **Expected result.** Strong agreement. *Obtained (`ipi_sensitivity.csv`):*
  $\rho$ = 0.997 / 0.995 / 0.995; top-5 overlap = 0.80 / 1.00 / 1.00; top-10 = 0.90 all.

## 7. Intervention "investment" — bottleneck lever to +30% GVA

- **Why.** Turn each priority into an actionable target: how far must the area's
  bottleneck lever move for predicted GVA to reach the +30% 2036 ambition.
- **Theory / formula.** An ICE / individual partial-dependence sweep: hold area $i$'s
  other features fixed, vary the bottleneck $b_i$ across Swindon's observed range,
  predict $\text{GVA}(v)=\exp(\hat{y})$, and find the crossing where
  $\text{GVA}(v)=1.30\times\text{GVA}(\text{now})$. "Investment" = required lever change
  $\Delta=v^\*-v_{\text{now}}$. **No £-cost model exists in the data**, so this is a
  lever movement, associational, not a guaranteed return on spend.
- **Evaluation.** Whether the +30% target is reachable within the observed range (flag).
- **Expected result.** Per-area lever targets in `ipi_investment.csv` +
  `lever_curves.json`. *Obtained:* all five top areas reach +30% within range.

## 8. Planned XGBoost vs TabPFN comparison

- **Why.** Two model families with the same pipeline test whether the priorities are a
  property of the data or of the model.
- **What to compare.** OOF R² (fit); global SHAP ranking (Spearman/overlap of feature
  importances); IPI ranking agreement (Spearman + top-$k$ overlap of areas);
  bottleneck agreement; investment-target directions.
- **⚠️ Fairness caveat.** The current TabPFN IPI uses the **MSOA**-commute dataset
  while this XGBoost run uses **LSOA**-commute. A clean comparison requires **both on
  the same dataset** — recommended next step: re-run the TabPFN IPI on
  `data_swindon_with_lsoa_commute.csv`. Until then, differences confound model with
  commuting geography.
