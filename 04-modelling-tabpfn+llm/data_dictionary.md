# Data Dictionary

> Swindon local labour productivity / GVA prediction project — source documentation for raw features.
> Geography: mostly LSOA. Business-structure fields are MSOA-level (watch for grouping leakage when broadcasting/disaggregating down to LSOA).

---

## Identification / Metadata

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `LSOA21CD`, `LSOA21NM` | Official 2021 LSOA codes and names | ONS Open Geography | LSOA | 2021 |
| `MSOA21CD`, `MSOA21NM` | Official 2021 MSOA codes and names | ONS Open Geography | MSOA | 2021 |

---

## Economic Output & Productivity

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `VOA_total_RV_million_2023` | Total commercial rateable value proxy for local business asset value | GOV.UK VOA | LSOA | 2023 |
| `VOA_median_RV_thousand_2023` | Median rateable value per hereditament (in £000s) | GOV.UK VOA | LSOA | 2023 |
| `VOA_hereditaments_2023` | Count of commercial hereditaments (rateable units) | GOV.UK VOA | LSOA | 2023 |
| `VOA_total_RV_million_2026` | Total commercial rateable value proxy for local business asset value | GOV.UK VOA | LSOA | 2026 |
| `VOA_median_RV_thousand_2026` | Median rateable value per hereditament (in £000s) | GOV.UK VOA | LSOA | 2026 |
| `VOA_hereditaments_2026` | Count of commercial hereditaments (rateable units) | GOV.UK VOA | LSOA | 2026 |
| `VOA_total_RV_pct_change` | Percentage change for total commercial rateable value proxy for local business asset value 2023–2026 | GOV.UK VOA (Calculated) | LSOA | 2023, 2026 |
| `LSOA GVA Estimates (millions)` | Gross Value Added local economic output (in £ millions) | NOMIS UK small area GVA | LSOA | 2023 |

---

## Skills, Education & Labour Market

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `Apprenticeship`, `Level 1`, `Level 2`, `Level 3`, `Level 4`, `No qualifications` | % Population with different skill levels | ONS Census | LSOA | 2021 |
| `Pct_working_Age` | % Population between ages 16–66 | ONS | LSOA | 2024 |
| `Working_Age_Pop` | Population between ages 16–66 | ONS | LSOA | 2024 |
| `full_to_part_ratio` | Ratio of full-time to part-time employment | BRES (Calculated) | LSOA | 2024 |
| `total_employees` | Total in full-time and part-time employment | BRES (Calculated) | LSOA | 2024 |
| `full_time_employees` | Workers in full-time employment | BRES | LSOA | 2024 |
| `part_time_employees` | Workers in part-time employment | BRES | LSOA | 2024 |
| `employment_rate_per_pop` | Total full and part-time employees divided by Mid-2024 population | ONS/BRES (Calculated) | LSOA | 2021 |

---

## Business Structure & Enterprise Composition

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `LU_Total_2025_msoa` | Count of local units | Nomis UK Business Counts | MSOA | 2025 |
| `total_enterprises_2025_msoa` | Count of unique enterprises | Nomis UK Business Counts | MSOA | 2025 |
| `LU_pct_micro_2025_msoa`, `LU_pct_small_2025_msoa`, `LU_pct_medium_2025_msoa`, `%LU_pct_large_2025_msoa` | Business size distribution (% of local units by employee count) | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `pct_enterprise_0_to_49_thousand_2025_msoa` – `pct_enterprise_5000plus_thousand_2025_msoa` | Distribution of businesses by turnover band (£000s) | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `turnover_diversity_1-HHI_2025_msoa` | Enterprise turnover diversity index (1 minus Herfindahl–Hirschman Index) | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `LU_diversity_1-HHI_2025_msoa` | Local unit industry diversity index (1 minus Herfindahl–Hirschman Index) | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `enterprises_per_1k_residents_2025` | Enterprise density (per 1,000 residents) | Nomis UK Business Counts & ONS (Calculated) | MSOA | 2025 |
| `creative`, `foundational`, `green_aligned`, `industrial`, `kibs`, `anchor` | Sectoral breakdown of employment by economic category | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `share_enterprises_public_2025_msoa` | Public sector enterprise share | Nomis UK Business Counts (Calculated) | MSOA | 2025 |
| `share_enterprises_private_2025_msoa` | Private sector enterprise share | Nomis UK Business Counts (Calculated) | MSOA | 2025 |

---

## Deprivation, Demographics & Population

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `Income_decile`, `Employment_decile`, `Education_decile`, `AdultSkills_decile` | Index of Multiple Deprivation (IMD) domain deciles | MHCLG IMD | LSOA | 2025 |
| `IMD_decile` | Overall IMD decile | MHCLG IMD | LSOA | 2025 |
| `Mid-2024 population` | Mid-year population estimate | ONS | LSOA | 2024 |

---

## Digital Connectivity

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `Average download speed (Mbit/s)` | Mean broadband download speed | Digital Exclusion Risk Index | LSOA | 2022 |
| `Percentage of connections receiving less than 10Mbit/s broadband` | Premises with poor connectivity | Digital Exclusion Risk Index | LSOA | 2022 |
| `Percentage of homes unable to receive at least 30Mbit/s broadband` | Homes below superfast threshold | Digital Exclusion Risk Index | LSOA | 2022 |

---

## Local Connectivity

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `bus_stop_density` | Number of bus stops per km² within each LSOA | OpenStreetMap network via OSMnx (Calculated) | LSOA | 2025 |
| `dist_to_primary_road_m` | Mean distance (metres) from sampled walk network nodes to nearest A road / primary road | OpenStreetMap network via OSMnx (Calculated) | LSOA | 2025 |

---

## Land Use

| Feature | Description | Source | Geography | Year |
|---|---|---|---|---|
| `RUC21NM` | Rural-urban classification | ONS Open Geography | LSOA | 2021 |
