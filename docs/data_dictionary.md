# Data Dictionary

All released data are **municipality-level aggregates for the 18 prefecture-level
divisions of Henan Province, China** (17 prefecture-level cities plus the
provincially administered demonstration zone of Jiyuan). The study year for the
efficiency analysis is **2024**. No point coordinates, no named heritage-site
records and no individual inheritor information are included.

All files are UTF-8 encoded CSV with comma delimiters.

---

## 1. `data/henan_ich_tourism_master.csv`

One row per municipality (18 rows). This is the single input file for scripts
`r1`–`r5`, `r7` and `r8`.

### Identity and geography

| Column | Type | Unit | Definition |
|---|---|---|---|
| `city` | string | — | Municipality name in English. |
| `area_km2` | float | km² | Administrative land area. |

### Cultural-heritage inputs

| Column | Type | Unit | Definition |
|---|---|---|---|
| `national_ich_items` | int | items | Count of national-list intangible cultural heritage (ICH) items located in the municipality. |
| `provincial_ich_items` | int | items | Count of Henan provincial-list ICH items. |
| `provincial_representative_inheritors` | int | persons | Count of provincial representative ICH inheritors. |
| `traditional_villages` | int | villages | Count of Chinese traditional villages (a village-level heritage proxy). |
| `national_key_cultural_relic_units` | int | sites | Count of national key cultural-relic protection units (material heritage). |
| `ich_composite_index` | float | index | Equal-weight ICH-project-reserve index for Model B: `0.5*(z(national_ich_items)+z(provincial_ich_items))`, z-scores with population SD (ddof=0); inheritors and protection units are excluded. Reproduced by `src/r0_composite_index.py`. |
| `ich_density_per_1000km2` | float | items / 1000 km² | ICH item count divided by area and scaled to 1000 km²; the dependent variable of the GeoDetector analysis. |

### Socioeconomic context

| Column | Type | Unit | Definition |
|---|---|---|---|
| `gdp_100m_yuan` | float | 100 million yuan | Gross domestic product. |
| `permanent_population_10k` | float | 10,000 persons | Year-end permanent resident population. |
| `population_status` | categorical | — | `direct_bulletin`, `back_cast_2025_bulletin`, or `trend_estimated` (see provenance below). |
| `urbanization_rate_pct` | float | % | Share of permanent population residing in urban areas. |
| `tertiary_share_pct` | float | % | Share of value added from the tertiary (service) sector. |
| `tertiary_status` | categorical | — | `direct_bulletin` or `estimated`. |
| `mean_elevation_m` | float | metres | Mean elevation of the municipality, derived from a digital elevation model. |

### Outputs

| Column | Type | Unit | Definition |
|---|---|---|---|
| `tourism_revenue_100m_yuan_2024` | float | 100 million yuan | Total tourism revenue in 2024, the single output of the primary DEA. |
| `tourism_revenue_status` | categorical | — | `reported` for all 18 municipalities (official 2024 bulletin total; see `data/tourism_revenue_provenance.csv`). |
| `a_level_scenic_areas_2024` | int | areas | Frozen official end-2024 count of **all-grade** A-level scenic areas; the second output of the dual-output DEA. |
| `a_level_source_type` | categorical | — | `municipal_2024_statistical_communique` (13 municipalities), `municipal_culture_and_tourism_authority` (four municipalities) or `municipal_government_release` (Zhengzhou). |

### Primary DEA results (derived by `r1_dea.py`)

| Column | Type | Definition |
|---|---|---|
| `te_crs` | float | Input-oriented SBM technical efficiency under constant returns to scale (CRS), full machine precision; range (0, 1], 1 = efficient. |
| `pte_vrs` | float | Pure technical efficiency under variable returns to scale (VRS). |
| `scale_efficiency` | float | Scale efficiency, equal to `te_crs / pte_vrs`. |
| `returns_to_scale` | categorical | `CRS`, `IRS` (increasing) or `DRS` (decreasing), identified with the non-increasing-returns-to-scale comparison. |
| `quadrant_median_split` | categorical | Quadrant I–IV from a median split of pure technical efficiency and scale efficiency (`I` = benchmark). |

---

## 2. `data/a_level_scenic_areas_2024_sources.csv`

City-by-city source register for the frozen A-level scenic-area counts (18 rows).

| Column | Definition |
|---|---|
| `city` | Municipality name in English. |
| `a_level_scenic_areas_2024` | Frozen all-grade count used in the analysis. |
| `geographic_scope` | Administrative scope of the figure (whole prefecture-level city; Shangqiu includes the county-level city of Yongcheng). |
| `source_type` | `municipal_2024_statistical_communique`, `municipal_culture_and_tourism_authority` or `municipal_government_release`. |
| `source_detail` | Free-text provenance note naming the bulletin/authority and the graded subtotal where available. |

Thirteen figures come directly from each city's 2024 statistical communique.
Four municipalities (Kaifeng, Anyang, Xinxiang and Sanmenxia) publish only a
grade subtotal or a related-year figure in their statistical communique; their
all-grade total is taken from the corresponding municipal culture-and-tourism
authority. Zhengzhou's end-2024 count is taken from the municipal government's
2025 official release, which reports the all-grade total that its statistical
communique does not carry. All 18 figures are official; none is estimated by the
authors.

---

## 3. `data/tourism_revenue_provenance.csv`

City-by-city source register for the 2024 tourism-revenue output (18 rows).
Every value is a direct official 2024 municipal figure; no revenue is
extrapolated or back-cast.

| Column | Definition |
|---|---|
| `city` | Municipality name. |
| `tourism_revenue_100m_yuan_2024` | 2024 total tourism revenue (100 million yuan). |
| `status` | `reported` for all 18 municipalities. |
| `method` | Short note on the figure used. |
| `source_kind` | Nature of the source document (e.g. 2024 statistical bulletin). |
| `source_title` | Full title and issuing authority of the source document. |
| `source_url` | Verifiable URL of the source document. |

---

## 4. `data/spatial_point_pattern_inputs.csv`

Municipality-level aggregate inputs sufficient to reproduce density-based and
quadrat point-pattern summaries (18 rows). The underlying individual ICH point
geometry comes from the third-party IhChina product and is **not** redistributed;
see `spatial_reconstruction/README.md`.

| Column | Definition |
|---|---|
| `city` | Municipality name. |
| `area_km2` | Administrative land area. |
| `national_ich_items` | National-list ICH item count. |
| `provincial_ich_items` | Provincial-list ICH item count. |
| `ich_density_per_1000km2` | National-list ICH item count per 1,000 km² (`national_ich_items / area_km2 × 1000`); the dependent variable of the GeoDetector and the ICH-density predictor in Model B. |
| `total_ich_items_national_plus_provincial` | Sum of national- and provincial-list item counts. |
| `crude_density_items_per_1000km2` | `total / area × 1000`. |

---

## 5. `outputs/r8_simar_wilson.csv` and `outputs/r8_interior_subsample.csv`

`r8_simar_wilson.csv` (one row per predictor, nine rows) holds the all-$n=18$
OLS slopes (`beta_ols`, identical to Table 6 of the paper) and the 2.5–97.5
percentile ranges (`boot_ci_low`, `boot_ci_high`) of the 1,000 pseudo-frontier
bootstrap draws (`n_bootstrap`, seed `20261017`); `ci_contains_zero` flags the
range.

`r8_interior_subsample.csv` is an audit diagnostic on the 13 inefficient
municipalities (θ < 1), for which alone a truncated-normal second stage is
defined. Columns: `model`, `term` (`const`, `z1`, `z2`), `beta_ols_all18`,
`beta_ols_interior13` and `beta_truncmle_interior13`. The truncated-MLE and
interior-OLS slopes coincide to within 0.006, but Model B differs from the
all-sample OLS: composite +0.1692 → −0.0095, density −0.1507 → −0.0714,
showing the nominal full-sample Model B association is carried by the five
frontier units.

---

## Provenance and reconstruction codes

Several explanatory variables could not be read directly from a 2024 bulletin
for every municipality. The status columns document the treatment; the exact
reconstruction rules and source citations are given in the paper's Supplementary
Table S3. Briefly:

- **Tourism revenue** (`reported` for all 18 municipalities): every value is the
  2024 total taken directly from that city's official 2024 statistical
  communique; the value, communique title and a verifiable source URL for each
  city are in `data/tourism_revenue_provenance.csv`. No tourism revenue is
  extrapolated. The `r7` Monte Carlo is therefore a generic measurement-error
  sensitivity (independent +/-10%/+/-20% and common multiplicative shocks), not
  a reconstruction-stress test; it shows the efficiency ranking is robust.
- **Population** (`back_cast_2025_bulletin`, three municipalities;
  `trend_estimated`, seven municipalities): interpolated from documented
  adjacent-year bulletins.
- **Tertiary share** (`estimated`, three municipalities): derived from
  documented value-added components.
- **A-level scenic areas**: official in all 18 cases; `source_type` records
  whether the all-grade total appears in the statistical communique (13
  municipalities), was provided by the municipal culture-and-tourism authority
  (Kaifeng, Anyang, Xinxiang, Sanmenxia) or is taken from the municipal
  government's 2025 official release (Zhengzhou).

All source statistics are public government figures (municipal 2024 statistical
communiques, culture-and-tourism authority releases). Elevation was computed
from a public digital-elevation raster. No coordinate-level heritage data are
released.
