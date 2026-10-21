#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2: second-stage OLS Models A, B and C, with VIFs, correlations, Benjamini-Hochberg adjustment and 95% CIs.

Part of the reproducibility archive for:
  "From Heritage Resources to Tourism Outcomes: Spatial Efficiency Analysis
   of Intangible Cultural Heritage-Tourism Conversion in Henan, China".

The script reads ONLY the released, municipality-level CSV files in ../data and
writes results to ../outputs. No raw point records or coordinates are required.
Run from the code/ directory, or from anywhere (paths are resolved relative to
this file).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

MASTER = os.path.join(DATA, "henan_ich_tourism_master.csv")


def load_master():
    df = pd.read_csv(MASTER)
    assert len(df) == 18, "expected 18 municipalities"
    return df


def zscore(v):
    """Population standardization (ddof=0), as used throughout the paper."""
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=0)


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out


def fit(name, y, Xdf):
    Xc = sm.add_constant(Xdf)
    M = sm.OLS(y, Xc).fit()
    print("=" * 76)
    print(f"{name}: R2={M.rsquared:.3f} adjR2={M.rsquared_adj:.3f} "
          f"F={M.fvalue:.3f} joint_p={M.f_pvalue:.4f} n={int(M.nobs)}")
    ci = M.conf_int(0.05)
    for k in Xdf.columns:
        print(f"  {k:12s} beta={M.params[k]:+.3f} SE={M.bse[k]:.3f} "
              f"t={M.tvalues[k]:+.2f} p={M.pvalues[k]:.4f} "
              f"95%CI[{ci.loc[k,0]:+.3f},{ci.loc[k,1]:+.3f}]")
    if Xc.shape[1] > 2:
        print("  VIF:", [round(variance_inflation_factor(Xc, j), 2)
                        for j in range(1, Xc.shape[1])])
    return M


def main():
    df = load_master()
    y = df["te_crs"].to_numpy(float)

    gdp, pop, ter = zscore(df.gdp_100m_yuan), zscore(df.permanent_population_10k), zscore(df.tertiary_share_pct)
    comp, dens = zscore(df.ich_composite_index), zscore(df.ich_density_per_1000km2)
    alev = zscore(df.a_level_scenic_areas_2024)
    area = df.area_km2.to_numpy(float)
    alev_dens = zscore(df.a_level_scenic_areas_2024.to_numpy(float) / area)
    villages = zscore(df.traditional_villages)

    mA = fit("Model A (socioeconomic)", y,
             pd.DataFrame({"gdp": gdp, "population": pop, "tertiary": ter}))
    mB = fit("Model B (ICH construct)", y,
             pd.DataFrame({"composite": comp, "density": dens}))
    mC = fit("Model C (external tourism supply)", y,
             pd.DataFrame({"a_level_2024": alev, "traditional_villages": villages}))
    mCd = fit("Model C area-normalized", y,
              pd.DataFrame({"a_level_density": alev_dens, "traditional_villages": villages}))

    # BH across the seven second-stage predictor tests reported in the paper
    preds = {"A_gdp": mA.pvalues["gdp"], "A_pop": mA.pvalues["population"],
             "A_tertiary": mA.pvalues["tertiary"],
             "B_composite": mB.pvalues["composite"], "B_density": mB.pvalues["density"],
             "C_alevel": mC.pvalues["a_level_2024"],
             "C_alevel_density": mCd.pvalues["a_level_density"]}
    q = bh(list(preds.values()))
    print("\nBenjamini-Hochberg across seven second-stage predictor tests:")
    for (k, p_), qq in zip(preds.items(), q):
        print(f"  {k:16s} p={p_:.4f}  q={qq:.3f}")

    r = np.corrcoef(df.a_level_scenic_areas_2024, df.traditional_villages)[0, 1]
    print(f"\nspots-villages Pearson r = {r:.3f}")

    rows = []
    for mname, M, cols in [("A", mA, ["gdp", "population", "tertiary"]),
                           ("B", mB, ["composite", "density"]),
                           ("C", mC, ["a_level_2024", "traditional_villages"]),
                           ("C_density", mCd, ["a_level_density", "traditional_villages"])]:
        for c in cols:
            rows.append({"model": mname, "predictor": c,
                         "beta": M.params[c], "p": M.pvalues[c],
                         "r2": M.rsquared, "adj_r2": M.rsquared_adj,
                         "f": M.fvalue, "joint_p": M.f_pvalue})
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "r2_regressions.csv"), index=False)
    print("\nwrote r2_regressions.csv")


if __name__ == "__main__":
    main()
