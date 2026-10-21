# Henan ICH–Tourism DEA: Replication Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](requirements.txt)

Reproducible analysis code for the manuscript on intangible cultural heritage
(ICH) and tourism efficiency across the **18 municipalities of Henan Province,
China (2024 cross-section)**. The repository reproduces the input-oriented
slacks-based measure (SBM) DEA, the three-input ICH-only robustness model, the
dual-output sensitivity DEA, the second-stage OLS models with Benjamini–Hochberg
adjustment and Freedman–Lane permutation p-values, the Simar–Wilson-style
pseudo-frontier bootstrap of the second stage, the GeoDetector factor and
interaction analysis, the tourism-revenue
uncertainty Monte Carlo, and (with restricted spatial inputs) the point-pattern
and Moran's-I diagnostics.

> **Data license differs from code license.** Code here is **MIT**. The bundled
> CSVs in `data/` are released under **CC-BY-4.0**; see
> [`DATA_LICENSE.txt`](DATA_LICENSE.txt) and
> [`DATA_USE_STATEMENT.txt`](DATA_USE_STATEMENT.txt). Point coordinates,
> point-level heritage/inheritor records and restricted third-party layers are
> **not** included.

## Repository layout

```
henan_ich_tourism_dea_replication/
├── run_all.py                    # run every step in order (r0,r1,r4,r3,r2,r2b,r5,r7 [,r8])
├── requirements.txt
├── README.md
├── LICENSE                       # MIT (code)
├── DATA_LICENSE.txt              # CC-BY-4.0 (data)
├── DATA_USE_STATEMENT.txt        # what is / is not released, and why
├── CITATION.cff
├── src/
│   ├── r0_composite_index.py     # M2 equal-weight z-mean ICH index (data prep)
│   ├── r1_dea.py                 # primary input-oriented SBM-DEA (CRS/VRS/SE)
│   ├── r4_ich_only_dea.py        # ICH-only three-input robustness DEA
│   ├── r3_dual_output_dea.py     # dual-output (revenue + A-level scenic areas)
│   ├── r2_regressions.py         # OLS Models A/B/C, VIF, BH FDR
│   ├── r2b_permutation_tests.py  # Freedman-Lane permutation p-values (companion to r2)
│   ├── r5_geodetector.py         # GeoDetector + exact 1-D Fisher-Jenks DP
│   ├── r6_spatial_statistics.py  # NNI + global/bivariate Moran's I (optional)
│   ├── r7_revenue_monte_carlo.py
│   └── r8_simar_wilson.py        # Simar-Wilson-style pseudo-frontier bootstrap (slow)
├── data/
│   ├── henan_ich_tourism_master.csv          # 18 municipalities, 25 columns
│   ├── a_level_scenic_areas_2024_sources.csv # frozen official A-level counts + sources
│   ├── tourism_revenue_provenance.csv        # city-by-city 2024 revenue source register
│   └── spatial_point_pattern_inputs.csv      # municipality-level aggregates
├── outputs/                      # reference CSVs produced by the scripts
├── docs/
│   └── data_dictionary.md        # column-by-column variable dictionary
├── spatial_reconstruction/
│   └── README.md                 # lawful reconstruction of the restricted spatial inputs
└── tests/
    └── test_reproduction.py      # asserts key reported numbers
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py                  # add SKIP_BOOTSTRAP=1 for a fast run
pytest -q                          # check key reported numbers
```

Each script is independent, uses only relative paths, reads from `../data`
(relative to `src/`) and writes to `../outputs`. Random seeds are fixed
(GeoDetector permutation seed `20261017`, B = 9,999; Monte Carlo 2,000 draws,
seed `20261017`; Freedman–Lane permutation seed `20261018`, N = 99,999;
Simar–Wilson bootstrap seed `20261017`, B = 1,000), so the reported numbers
reproduce deterministically except for tiny Monte Carlo sampling variation.

## What each step returns

| Step | Key result (manuscript) |
|------|-------------------------|
| r0 | ICH composite = 0.5·[z(national items) + z(provincial items)], ddof=0 |
| r1 | Primary frontier (5 cities) = Zhengzhou, Kaifeng, Luoyang, Anyang, Nanyang; lowest = Hebi 0.159 |
| r4 | ICH-only (3-input) frontier contracts to 2 (Zhengzhou, Luoyang); Spearman ρ = 0.921 (Pearson r = 0.881); Anyang 1.000→0.673, Kaifeng →0.608, Nanyang →0.593 |
| r3 | Dual-output frontier expands to 10; mean dual TE = 0.861; Spearman ρ = 0.54 (reported scores) / 0.44 (full precision) |
| r2 | Model B: composite β = 0.169 (p = .027), density β = −0.151 (p = .045); Model C: A-level β = 0.092 (p = .289), joint p = .472 |
| r2b | Freedman–Lane permutation p-values (99,999 draws): Model B composite p = .027, density p = .042, joint F p = .047 |
| r5 | Quantile Her×Pop q = 0.993 (p = .008; 14 strata, 11 singletons) and Her×GDP q = 0.951 (p = .017; 12 strata, 7 singletons); natural-breaks q: Her×Pop 0.563, Her×GDP 0.645, Her×Urban 0.589 |
| r6 | NNI 0.539 (z = −9.86) for all 125 records; unique-location NNI 1.051 (p = .398); Queen I = −0.075 (p = .488), KNN-4 I = 0.054 (p = .160), bivariate I = 0.069 (p = .298) |
| r7 | Revenue MC median ρ = 0.984 (independent ±10%), 0.949 (±20%), 1.000 (common shock) |
| r8 | Simar–Wilson-style pseudo-frontier bootstrap (B = 1,000, seed 20261017): all nine second-stage intervals contain zero (e.g. Model B composite β = 0.169, 95% range [−0.037, +0.236]). `r8_interior_subsample.csv` audits the truncated-normal sample (13 inefficient units): Model B composite +0.169 → −0.010, density −0.151 → −0.071, i.e. the nominal full-sample Model B association is carried by the five frontier units |

## ICH composite index (Model B) — definition

The composite used by exploratory Model B is **subjective-weight free**:

```
ICH_composite_i = 0.5 · [ z(national_ich_items_i) + z(provincial_ich_items_i) ]
```

with population standard deviations (`ddof=0`), matching the standardisation
applied to every regressor in `r2_regressions.py`. Only the two ICH
*project-list* counts enter; representative inheritors (people) and national key
cultural-relic protection units (physical heritage) are excluded so the index
measures one concept (listed ICH-project stock). The two administrative levels
get equal weight after standardisation — no national:provincial multiplier is
used. All four resource counts still enter the first-stage DEA. `r0`
recomputes the column idempotently from the two raw counts; it must run before
`r2`. The ICH *density* term is a separate, raw variable: national-list items
per 1,000 km².

## Optional: spatial diagnostics (r6)

`r6_spatial_statistics.py` is **not** on the `run_all.py` chain because it
needs restricted third-party spatial layers and an optional geo stack:

```bash
pip install geopandas libpysal esda
# place, under a directory pointed to by SPATIAL_INPUT_DIR (default:
# ../spatial_inputs relative to the repository root):
#   - the IhChina/ICH-project point dataset (.dbf or .shp with WGS84 lon/lat),
#   - the 18-municipality boundary polygon (.shp, EPSG:4326),
#   - the traditional-village point/attribute table,
python src/r6_spatial_statistics.py
```

If the layers are absent the script exits 0 after writing a status file, so a
restricted-data-free installation never breaks. Queen/KNN weights are built in
the boundary layer's native geographic CRS before projection (constructing KNN
weights only after projection changes high-latitude neighbour sets). See
[`spatial_reconstruction/README.md`](spatial_reconstruction/README.md) for
lawful reconstruction of these inputs.

## Reproducibility notes

- The dual-output DEA is solved in **envelopment form** (min θ with
  `Xλ ≤ θx_j`, `Yλ = y_j`); its single-output comparator is the primary SBM
  score, so adding the second output is the *only* change. Adding the A-level
  output expands the frontier from five to ten cities. The reported
  Spearman ρ = 0.54 uses the three-decimal scores with average ranks for the
  numerically tied frontier units; on the full-precision solver outputs
  (frontier units differ only below 1e-15, so the ordering among them is
  machine-precision dependent) the value is ≈ 0.44.
- `r8_simar_wilson.py` keeps the Simar–Wilson pseudo-frontier resampling but
  fits the second-stage OLS on all 18 municipalities, because a truncated-normal
  MLE is undefined on the five CRS-efficient units (they sit exactly at θ = 1,
  where the truncated likelihood is unbounded). The interior-only truncated MLE
  on the 13 inefficient units is run as a diagnostic and, together with the
  interior-only OLS, is written to `outputs/r8_interior_subsample.csv`; it
  reproduces the interior OLS slopes (max absolute difference 0.006) but shows
  that the Model B composite association vanishes once the frontier units are
  excluded (+0.169 on all 18 versus −0.010 on the 13), a sensitivity disclosed
  in the paper's Limitations section.
- `r2b_permutation_tests.py` reuses the Model A/B/C design matrices of
  `r2_regressions.py` and adds Freedman–Lane residual-permutation p-values
  (99,999 permutations, seed `20261018`) for each coefficient and for the joint
  F-test, complementing the analytic OLS p-values.
- `r5_geodetector.py` implements the exact one-dimensional Fisher-Jenks dynamic
  program; the backtracking table stores the optimal cut position (`bi`), which
  is required for correct natural-breaks strata. Permutation p-values come from
  9,999 permutations with seed `20261017`; the manuscript reports exactly the
  p-values this seed produces.

## Data provenance

Municipality-level aggregates only. Sources: Henan Statistical Yearbook and the
municipalities' 2024 statistical communiqués; official national/provincial ICH
lists and national key cultural-relic-protection-unit lists; A-level scenic
area counts from each city's 2024 communiqué (13 cities), the municipal
culture-and-tourism authority (Kaifeng, Anyang, Xinxiang, Sanmenxia) or the
municipal government's 2025 official release (Zhengzhou). All 18
tourism-revenue values are direct official 2024 municipal totals with a
city-by-city source register in `data/tourism_revenue_provenance.csv`. The
IhChina point dataset, administrative boundaries and the SRTM/CASEarth DEM are
**not** redistributed; see [`spatial_reconstruction/README.md`](spatial_reconstruction/README.md)
for lawful reconstruction and [`docs/data_dictionary.md`](docs/data_dictionary.md)
for the full variable dictionary.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the manuscript and the
archived Zenodo dataset (DOI to be inserted at publication).
