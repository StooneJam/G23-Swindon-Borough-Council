# How SME Composition, Commuting and Employment Shape Small Area GVA in Swindon

MSc dissertation project (Group 23) developed with Swindon Borough Council, applying data science methods to understand neighbourhood-level economic variation and support evidence-based, place-targeted policy.

## 1. Team

**Name:** Shi Qin, Anis Binti Shahrulhisham, Bo-Yan Lu  
**Supervisor:** Dr. Ayush Joshi  
**Degree:** MSc, University of Bristol  
**Project title:** *How SME Composition, Commuting and Employment Shape Small Area GVA in Swindon*  
**Year:** 2026

## 2. Project Overview

### 2.1 Research Background

Swindon Borough Council's Economic Growth Plan (2026–2031) sets a working ambition to increase Gross Value Added (GVA) by around 30% by 2036, targeting priority sectors and enterprise types. The Council must decide where limited resources can support growth while under significant financial pressure — its statement of accounts records planned savings of £14.4 million for 2025/26 alongside £14.7 million of Exceptional Financial Support in the same year. A single borough-wide GVA figure cannot guide this: it hides how neighbourhoods differ in business structure, skills, deprivation, employment and access to work, and it obscures the fact that GVA is recorded at the location of economic activity rather than where workers live, making commuting patterns central to interpreting local output.

This project uses Lower layer Super Output Areas (LSOAs) — and Middle layer Super Output Areas (MSOAs) where finer data are unavailable — as the unit of analysis, combining open economic and demographic data (ONS, VOA, UK Business Counts, IMD, skills and connectivity indicators, 2021 Census commuting flows) with a governance-approved extract of local business rates data provided by the Council. The final analytical dataset covers 1,125 LSOAs, including all 137 within Swindon. The goal is decision support rather than a pure prediction contest: predictive performance matters as a foundation, but the value lies in whether findings can be interpreted, explained and connected to actionable council decisions.

### 2.2 Research Objectives

The dissertation is organised around three research questions:

- **RQ1 — Predict:** Which feature selection approaches and regression models give the most reliable predictions of small-area GVA, and do commuting variables and LLM-selected features add useful information?
- **RQ2 — Diagnose:** What neighbourhood types exist within Swindon, and which actionable local characteristics most limit an area's economic performance?
- **RQ3 — Act:** How can commuting scenarios and RAG-based evidence retrieval turn this diagnosis into sourced policy options that can be tested through local pilots?

Throughout, the project is explicit that regression, SHAP and clustering describe **associations**, not causal effects; the 30% GVA ambition is treated as a strategic benchmark rather than a forecast the methods claim to deliver.

### 2.3 Main Methods

| Stage | Method | Purpose |
|---|---|---|
| Data & EDA | Manual DAG-guided feature selection vs. LLM-assisted feature selection (CAAFE-inspired screening over the *existing* variable list, not feature generation) | Compare human- and LLM-assisted variable selection under a fixed, leakage-checked candidate set |
| Regression | Linear regression baseline, LightGBM, XGBoost, TabPFN (grouped train/test split, MSOA-level significance testing, and a Swindon geographic holdout) | Build a credible, transferable predictive engine for total GVA (and GVA per worker) |
| Interpretation | SHAP — permutation explainer on TabPFN, exact TreeSHAP on an XGBoost benchmark — combined into an Intervention Priority Index (IPI) | Identify associational drivers, rank LSOAs by need and actionability, cross-check rankings across model engines |
| Typology | K-means clustering (k=5) across five domains: commercial assets, skills, labour intensity, commuting connectivity and business structure | Derive a repeatable neighbourhood typology (GVA held out of clustering, used only for post-hoc validation) |
| Scenario simulation | IPF-estimated commuting OD matrix + Poisson gravity/spatial-interaction model + geographically weighted regression (GWR) for local elasticity | Test conditional "what-if" changes in workplace attractiveness/commuting against the 2036 target gap |
| Policy translation | Retrieval-Augmented Generation (RAG) over four tiered, human-verified corpora (statutory, guidance, international precedent, local evidence), served locally (Qwen2.5, all-MiniLM-L6-v2 embeddings, Chroma, via Ollama) | Ground local findings in sourced, traceable policy evidence and pilot options |
| Communication | Self-contained interactive dashboard | Present findings from all stages to a non-technical audience |

### 2.4 Key Results

- **Regression:** TabPFN with LLM-selected features gave the strongest fit for observed 2023 log GVA (R² = 0.7078, MAPE = 9.90% on the held-out test set), retained R² = 0.7429 when trained outside Swindon and tested on the borough, and rose to R² = 0.7782 when LSOA commuting features were added.
- **Typology:** Five neighbourhood clusters (k=5) were identified across Swindon's 137 LSOAs, with GVA excluded from clustering but showing significant differences across clusters when compared post hoc (η² = 0.605).
- **SHAP / IPI:** Commuting features accounted for ~63–64% of SHAP importance under both TabPFN and XGBoost; the two engines' IPI rankings correlated at Spearman ρ = 0.73.
- **Scenario simulation:** The same simulated 30% increase in workplace attractiveness produced different GVA outcomes depending on where it was applied (e.g. +1.40% of borough GVA at the strongest-responding LSOA), showing that commuting-based targeting is location-dependent.
- **RAG:** The retrieval pipeline reached recall@10 = 1.000 and NDCG@10 = 0.791 against gold evidence, producing sourced, page-traceable answers versus generic, ungrounded output from the same model without retrieval.

All findings are reported as associations for decision support, not causal or forecast claims; final decisions remain with Swindon Borough Council, tested through local pilots.

## 3. Repository Structure

The analysis pipeline is organised as numbered stages, each corresponding to a chapter/section of the dissertation:

| Folder | Contents |
|---|---|
| `00-data/` | Base feature table, LSOA boundary/naming reference and the feature DAG used to guide variable selection |
| `01-data preprocessing+EDA/` | Data cleaning and exploratory analysis, including individual member EDA (`Anis/`, `Bryan/`, `Shi/`), the 2025 GVA projection, and the train/test splits used downstream |
| `02-literature review/` | Reference literature (PDFs) and bibliography files supporting the literature review chapter |
| `03-modelling-Treemodel/` | XGBoost baseline regression notebooks and supporting methodology references |
| `04-modelling-tabpfn+llm/` | TabPFN regression, LLM-assisted feature engineering (CAAFE), and Wilcoxon/Friedman significance testing across the four-model comparison |
| `05-cluster_lsoa/` | K-means neighbourhood typology (LSOA clustering) and associated figures |
| `06-SHAP/` | SHAP explainability and the Intervention Priority Index, split by engine (`IPI/` final outputs, `TabPFN/`, `TabPFN_LSOA/`, `XGBoost/` benchmark) plus commuting-feature SHAP comparisons (`figures_commute/`) and cross-model stability checks |
| `07-Scenario Simulation/` | Gravity model of commuting flows, closed-loop GVA accounting, and borough-wide and all-area scenario simulations |
| `08-commuting-data/` | Commuting/travel-to-work datasets (OD flows, MSOA commuting features) and TabPFN commuting predictions |
| `09-RAG/` | Retrieval-Augmented Generation pipeline (ingestion, search agent, review panel) linking findings to policy documents |
| `10-dashboard/` | Self-contained HTML/JS dashboard integrating the four analytical stages for presentation |

Internal working documents (`0-application methods/`, `1-report outline/`, `2-dissertation/`) are kept local only and excluded from version control via `.gitignore`, as are a few large raw data files referenced by the commuting and scenario-simulation stages.
