#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r6_spatial_statistics.py
Reproduce the point- and area-scale spatial statistics of the manuscript:

  1. Nearest-Neighbour Index (NNI) for national-list ICH point records:
       - all 125 Henan records (co-located listings kept -> nearest dist 0);
       - 74 unique geocoded locations.
  2. Global Moran's I of municipal ICH density (national items per 1,000 km2):
       Queen contiguity and k-nearest-neighbours (k=4), both row-standardized.
  3. Bivariate Moran's I of ICH density vs traditional-village density (Queen).

Data needed (RESTRICTED THIRD-PARTY LAYERS, NOT redistributed in this archive):
  - national ICH point table with coordinates  -> any *.dbf carrying X/Y and a
    ProvinceCN field (the IhChina_2006-2021 product);
  - Henan municipality boundary polygon shapefile (18 features);
  - the 274 Chinese Traditional Villages register CSV (columns 所属地区/村落名称).
Set the folder holding these via SPATIAL_INPUT_DIR (default ../spatial_inputs),
following spatial_reconstruction/README.md. If any are missing this script
writes r6_spatial_status.csv describing what is absent and exits 0, so the core
run_all pipeline still succeeds without the restricted layers.

Method (identical to the manuscript):
  NNI: projected to UTM zone 49N (EPSG:32649); no edge correction; study
       window A = total area of the 18 municipal polygons; d_exp = .5/sqrt(n/A);
       SE = 0.26136/sqrt(n^2/A); z=(d_obs-d_exp)/SE; two-sided normal p.
  Moran: row-standardized weights; 9,999 conditional permutations (seed 20261017).
"""
import os
import struct
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
SPAT = os.environ.get("SPATIAL_INPUT_DIR",
                      os.path.join(ROOT, "spatial_inputs"))
N_PERM = 9999
SEED = 20261017

CN = ["郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡", "焦作",
      "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口",
      "驻马店", "济源"]
EN = ["Zhengzhou", "Kaifeng", "Luoyang", "Pingdingshan", "Anyang", "Hebi",
      "Xinxiang", "Jiaozuo", "Puyang", "Xuchang", "Luohe", "Sanmenxia",
      "Nanyang", "Shangqiu", "Xinyang", "Zhoukou", "Zhumadian", "Jiyuan"]
CN2EN = dict(zip(CN, EN))


def read_dbf(path):
    with open(path, "rb") as f:
        b = f.read()
    nrec = struct.unpack("<I", b[4:8])[0]
    hlen = struct.unpack("<H", b[8:10])[0]
    rlen = struct.unpack("<H", b[10:12])[0]
    fields, pos = [], 32
    while b[pos] != 0x0D:
        name = b[pos:pos + 11].split(b"\x00")[0].decode("ascii", "ignore")
        fields.append((name, b[pos + 16]))
        pos += 32
    rows, off = [], hlen
    for _ in range(nrec):
        rec = b[off:off + rlen]
        off += rlen
        if rec[:1] == b"*":
            continue
        d, c = {}, 1
        for name, flen in fields:
            raw = rec[c:c + flen]
            c += flen
            try:
                d[name] = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                d[name] = raw.decode("gbk", "ignore").strip()
        rows.append(d)
    return rows


def find_file(predicate):
    for root, _, files in os.walk(SPAT):
        for fn in files:
            if predicate(fn, root):
                return os.path.join(root, fn)
    return None


def write_status(msg):
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"status": [msg]}).to_csv(
        os.path.join(OUT, "r6_spatial_status.csv"), index=False)
    print(msg)


def nni(coords_xy, area_m2):
    """coords_xy in projected metres; area_m2 study-window area."""
    from scipy.spatial import cKDTree
    from scipy import stats
    n = len(coords_xy)
    d, _ = cKDTree(coords_xy).query(coords_xy, k=2)
    d_obs = float(d[:, 1].mean())
    d_exp = 0.5 / np.sqrt(n / area_m2)
    se = 0.26136 / np.sqrt(n * n / area_m2)
    z = (d_obs - d_exp) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(n=n, mean_NN_km=round(d_obs / 1000, 3),
                expected_km=round(d_exp / 1000, 3), NNI=round(d_obs / d_exp, 3),
                z=round(z, 2), two_sided_p=float(f"{p:.4g}"))


def main():
    os.makedirs(OUT, exist_ok=True)
    dbf = find_file(lambda fn, r: fn.lower() == "ihchina_2006-2021.dbf")
    shp = find_file(lambda fn, r: fn.lower().endswith(".shp") and
                    ("行政" in fn or "市" in fn or "city" in fn.lower()))
    village = find_file(lambda fn, r: fn.lower().endswith(".csv") and
                        ("村落" in fn or "village" in fn.lower()))
    missing = [n for n, p in [("national ICH dbf", dbf),
                              ("municipality shp", shp),
                              ("village register csv", village)] if p is None]
    if missing:
        write_status("r6 skipped; restricted layers not found under %s: %s. "
                     "See spatial_reconstruction/README.md." % (SPAT, missing))
        return

    import geopandas as gpd
    from libpysal.weights import Queen, KNN
    from esda.moran import Moran, Moran_BV

    master = pd.read_csv(os.path.join(DATA, "henan_ich_tourism_master.csv"))
    area_sum = float(master["area_km2"].sum())

    # ---- NNI on projected points ----
    pts = []
    for r in read_dbf(dbf):
        if "河南" in r.get("ProvinceCN", ""):
            try:
                pts.append((float(r["X"]), float(r["Y"])))
            except (ValueError, KeyError):
                pass
    pts = np.array(pts)
    gpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1]),
                            crs="EPSG:4326").to_crs(32649)
    xy = np.array([(g.x, g.y) for g in gpts.geometry])
    city_proj = gpd.read_file(shp).to_crs(32649)
    A = float(city_proj.geometry.area.sum())
    nni_rows = [["All 125 item records", nni(xy, A)],
                ["Unique geocoded locations",
                 nni(np.unique(xy, axis=0), A)]]
    nni_df = pd.DataFrame([dict(specification=s, **d) for s, d in nni_rows])
    nni_df.to_csv(os.path.join(OUT, "r6_nni.csv"), index=False)
    print(nni_df.to_string(index=False))

    # ---- Moran: build weights on the boundary in its NATIVE (geographic) CRS,
    # aligned to master order. Using geographic coordinates matches the archived
    # audit script; projecting first changes k-nearest neighbours (longitude
    # compression at this latitude) and hence the k=4 statistic. Queen contiguity
    # is unaffected by projection. ----
    city = gpd.read_file(shp)
    namecol = next((c for c in ["name", "NAME", "市名", "城市", "city"]
                    if c in city.columns), city.columns[0])
    city["_base"] = city[namecol].astype(str).str.replace("市", "", regex=False)
    city["_en"] = city["_base"].map(CN2EN)
    if city["_en"].isna().any():
        # fall back: match by English column if present
        write_status("r6 NNI written; Moran skipped (boundary name column "
                     "%s does not map to the 18 cities)." % namecol)
        return
    city = city.set_index("_en").reindex(list(master["city"]))

    ich_dens = master.set_index("city")["ich_density_per_1000km2"].astype(float)

    # traditional-village density from the 274 register
    vil = pd.read_csv(village)

    def _is_region_column(s):
        vals = s.dropna().astype(str)
        return vals.str.contains("|".join(CN), regex=True).mean() > 0.5

    region_col = next(c for c in vil.columns if _is_region_column(vil[c]))
    counts = vil[region_col].astype(str).str.extract(
        "(" + "|".join(CN) + ")")[0].value_counts()
    vcount = pd.Series({CN2EN[k]: int(v) for k, v in counts.items() if k in CN2EN})
    vcount = vcount.reindex(list(master["city"])).fillna(0)
    area = master.set_index("city")["area_km2"].astype(float)
    vil_dens = vcount / area.reindex(vcount.index) * 1000.0

    wq = Queen.from_dataframe(city, use_index=False)
    wq.transform = "r"
    wk = KNN.from_dataframe(city, k=4)
    wk.transform = "r"
    np.random.seed(SEED)
    mq = Moran(ich_dens.values, wq, permutations=N_PERM)
    np.random.seed(SEED)
    mk = Moran(ich_dens.values, wk, permutations=N_PERM)
    np.random.seed(SEED)
    mb = Moran_BV(ich_dens.values, vil_dens.reindex(ich_dens.index).values,
                  wq, permutations=N_PERM)
    moran_df = pd.DataFrame([
        ["Global univariate ICH density", "Queen", round(mq.I, 3),
         round(float(mq.p_sim), 3)],
        ["Global univariate ICH density", "KNN k=4", round(mk.I, 3),
         round(float(mk.p_sim), 3)],
        ["Bivariate ICH vs village density", "Queen", round(mb.I, 3),
         round(float(mb.p_sim), 3)],
    ], columns=["statistic", "weights", "Moran_I", "permutation_p"])
    moran_df.to_csv(os.path.join(OUT, "r6_moran.csv"), index=False)
    print(moran_df.to_string(index=False))
    print("Study-window area (polygon) km2:", round(A / 1e6, 1),
          "| master area sum km2:", round(area_sum, 1))


if __name__ == "__main__":
    main()
