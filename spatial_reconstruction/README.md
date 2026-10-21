# Spatial Data Reconstruction Guide

This directory documents how the spatial inputs behind the published
municipality-level aggregates were obtained and processed. **No point-level
geospatial file is redistributed in this archive.** The raw ICH point dataset is
a licensed third-party product, and releasing individual heritage-site
coordinates is outside the permission obtained for this study. Everything
needed to reproduce the reported analyses is released as municipality-level
aggregates in `../data/`.

## 1. Inputs used in the study (not redistributed)

| Input | Product / source | Role | Access and terms |
|---|---|---|---|
| ICH point records, 2006–2021 | **IhChina** historical ICH dataset | National- and provincial-list item point locations used for counts, nearest-neighbour indices (NNI) and cross-type coincident-location analysis | Licensed academic product; redistributable only under the vendor's terms. Obtain directly from the IhChina provider. |
| Municipal boundaries | 1:1,000,000 administrative divisions of Henan Province | Study window and area; municipal aggregation and projection | National/standard administrative boundary product; obtain from the official mapping authority. |
| Elevation | SRTM / CASEarth digital-elevation raster | `mean_elevation_m` per municipality | Public DEM; mosaic, clip to the municipal polygons and compute the zonal mean. |

## 2. Processing workflow

1. Reproject the ICH points and municipal polygons to a common projected CRS
   (the study used **UTM zone 49N, EPSG:32649**) so distances and areas are in
   metres.
2. Spatially join the ICH points to the 18 municipal polygons and count, for
   each municipality, the national-list and provincial-list items. These are the
   released columns `national_ich_items` and `provincial_ich_items`.
3. For density, divide the relevant count (or the composite index) by municipal
   area and scale to 1000 km²; see `ich_density_per_1000km2` in
   `../data/henan_ich_tourism_master.csv`.
4. For elevation, clip the DEM to each municipal polygon and compute the zonal
   mean; see `mean_elevation_m`.
5. For point-pattern diagnostics (NNI, cross-type coincident locations), run the
   analysis on the projected points within the union of the 18 municipal
   polygons; no edge correction was applied. Only municipal aggregate counts
   are released (`../data/spatial_point_pattern_inputs.csv`).

## 3. Reproducing the spatial aggregates

A researcher who obtains the licensed IhChina point file and the administrative
boundary product can regenerate the released counts with a standard spatial
join, for example with `geopandas` (illustrative; substitute the actual file
paths):

```python
import geopandas as gpd

points = gpd.read_file("IhChina_2006-2021.shp").to_crs(32649)
cities = gpd.read_file("henan_municipalities.shp").to_crs(32649)
joined = gpd.sjoin(points, cities, how="inner", predicate="within")
counts = joined.groupby(["city_name", "ich_level"]).size().unstack(fill_value=0)
counts["area_km2"] = cities.to_crs(32649).area / 1e6
```

The resulting counts correspond column-for-column to
`spatial_point_pattern_inputs.csv`. The NNI itself is computed from point
coordinates with the standard observed-mean-nearest-neighbour-distance /
expected-distance formula; because that requires the licensed point geometry,
only the inputs to the aggregate spatial analyses are published here.

## 4. Privacy and coordinate handling

To avoid re-identifying individual heritage resources and to respect the
IhChina redistribution restriction, this archive deliberately omits:

- latitude/longitude of individual ICH items or inheritors;
- site names attached to point geometry;
- the raw raster or vector products.

Only 18-row, municipality-level statistics are released. This granularity is
sufficient to reproduce every numerical result in the paper, all of which are
estimated at the municipal level.

## 5. Reproducing the spatial statistics (`r6_spatial_statistics.py`)

The point- and area-scale spatial statistics (NNI, univariate and bivariate
Moran's I) require the restricted point table and boundary polygon, so `r6` is
not part of the default `run_all.py` chain. Once you have obtained the layers:

1. place them in one folder, e.g. `spatial_inputs/`, containing
   - `IhChina_2006-2021.dbf` (national ICH records with `X`, `Y`, `ProvinceCN`);
   - the Henan municipality boundary shapefile (18 polygons);
   - the 274 Chinese Traditional Villages register CSV;
2. install the optional dependencies (`geopandas`, `libpysal`, `esda`);
3. set `SPATIAL_INPUT_DIR` to that folder and run
   `python src/r6_spatial_statistics.py`.

The script writes `outputs/r6_nni.csv` and `outputs/r6_moran.csv`. NNI uses UTM
zone 49N projected points, no edge correction, study-window area equal to the
union of the 18 municipal polygons, and SE = 0.26136/sqrt(n^2/A). Moran weights
are built from the boundary in its native geographic CRS (Queen contiguity and
k=4 neighbours, row-standardized) with 9,999 permutations (seed 20261017);
permutation p-values differ only in the last decimal across RNG implementations.
Expected values: NNI = 0.539 (z = -9.86) for all 125 records and 1.051
(p = 0.398) for 74 unique coordinates; global Moran's I = -0.075 (Queen) and
0.054 (k = 4); bivariate Moran's I = 0.069 (Queen).
