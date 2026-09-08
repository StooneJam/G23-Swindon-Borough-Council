# 05-cluster_lsoa

Swindon LSOA neighbourhood typology for the G23 dissertation (owned workstream: cluster analysis).

## Contents
- `cluster_feature_exploration.ipynb` — final k-means pipeline (feature families, k selection, profiles, map, held-out GVA)
- `cluster_lsoa.csv` — LSOA cluster labels (k = 5)
- `cluster_profiles.csv` / `cluster_gva_summary.csv` / `cluster_summary_table.csv` — summary tables
- `figures/` — typology, map, feature-family search, and held-out GVA validation figures
- `SHARE_WITH_GROUP.md` — notes for integrating outputs with other modules

## Method notes
- Sample: 137 Swindon LSOAs
- Inputs: 18 features across assets, skills, labour, commute, and business domains
- Algorithm: median impute → winsorise → StandardScaler → k-means
- **GVA is held out of clustering** and used only for post-hoc validation (ANOVA / Kruskal–Wallis / η²)

## Selected solution
- k = 5 (stability + usable group size + strongest held-out GVA separation)

## Reproducibility note
Notebook paths auto-resolve for local folders and the group-repo `05-cluster_lsoa/` layout; exports write `cluster_lsoa.csv`, profiles and GVA summaries beside the notebook output directory.
