# Cluster analysis — Option C, k=6 (group decision)

**Status:** Updated — **137 Swindon LSOAs**, LLM + commute features, **k=6**.

## Why k=6

| k | Silhouette | Min n | Notes |
|---|------------|-------|-------|
| 4 | 0.238 | 17 | Easier narrative |
| **6** | **0.274** | **5** | Near-peak silhouette; no n=1 clusters |
| 8 | 0.275 | 1 | Messy tiny clusters |

Report line: *Silhouette peaked near k=6–8; k=6 was selected as a compromise between separation and interpretable group sizes (minimum n=5).*

## Results (by mean log GVA)

| Rank | Cluster | N | Mean log GVA | Colour |
|------|---------|---|--------------|--------|
| 1 | Employment & enterprise hub | 17 | 4.94 | Red |
| 2 | High-RV compact workplace | 5 | 4.38 | Dark orange |
| 3 | Mid SME / mixed residential | 30 | 3.87 | Yellow |
| 4 | Moderate-skill local-work residential | 13 | 3.00 | Orange |
| 5 | High-skilled outbound residential | 33 | 2.88 | Light blue |
| 6 | Deprived low-enterprise residential | 39 | 2.66 | Blue |

Cluster labels are **interpretation** of group means (not model outputs).

## Files

```
cluster_analysis/
  cluster_analysis.ipynb
  data_swindon_with_commute.csv
  swindon_lsoa_2021_all137.geojson
  swindon_clusters_for_tableau.geojson   ← Tableau single source
  swindon_tableau_ready.csv
  cluster_summary_table.csv
  cluster_profiles.csv
  cluster_gva_summary.csv
  swindon_cluster_map.png
  swindon_cluster_labels.csv
```

