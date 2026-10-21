#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1: primary input-oriented SBM-DEA (Tone 2001), CRS/VRS, NIRS, scale efficiency and median PTE-SE quadrants.

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


from scipy.optimize import linprog

INPUTS = ["national_ich_items", "provincial_ich_items",
          "provincial_representative_inheritors",
          "national_key_cultural_relic_units"]
OUTPUT = "tourism_revenue_100m_yuan_2024"


def io_sbm(X, Y, j, vrs=False):
    """Input-oriented, non-oriented-slack SBM under CRS (vrs=False) or VRS."""
    n, m = X.shape
    nv = n + m
    c = np.zeros(nv)
    for k in range(m):
        c[n + k] = -1.0 / (m * X[j, k])
    A_ub, b_ub = [], []
    row = np.zeros(nv)
    row[:n] = -Y.ravel()
    A_ub.append(row)
    b_ub.append(-Y[j, 0])
    A_eq, b_eq = [], []
    for k in range(m):
        row = np.zeros(nv)
        row[:n] = X[:, k]
        row[n + k] = 1.0
        A_eq.append(row)
        b_eq.append(X[j, k])
    if vrs:
        row = np.zeros(nv)
        row[:n] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * nv, method="highs")
    return (1 + res.fun if res.success else np.nan)


def io_sbm_nirs(X, Y, j):
    """Non-increasing returns to scale: sum(lambda) <= 1."""
    n, m = X.shape
    nv = n + m
    c = np.zeros(nv)
    for k in range(m):
        c[n + k] = -1.0 / (m * X[j, k])
    A_ub, b_ub = [], []
    row = np.zeros(nv)
    row[:n] = -Y.ravel()
    A_ub.append(row)
    b_ub.append(-Y[j, 0])
    rowl = np.zeros(nv)
    rowl[:n] = 1.0
    A_ub.append(rowl)
    b_ub.append(1.0)
    A_eq, b_eq = [], []
    for k in range(m):
        row = np.zeros(nv)
        row[:n] = X[:, k]
        row[n + k] = 1.0
        A_eq.append(row)
        b_eq.append(X[j, k])
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * nv, method="highs")
    return 1 + res.fun if res.success else np.nan


def main():
    df = load_master().sort_values("city").reset_index(drop=True)
    cities = df["city"].tolist()
    X = df[INPUTS].to_numpy(float)
    Y = df[[OUTPUT]].to_numpy(float)
    n = len(cities)

    TE = np.array([io_sbm(X, Y, j, False) for j in range(n)])
    PTE = np.array([io_sbm(X, Y, j, True) for j in range(n)])
    NIRS = np.array([io_sbm_nirs(X, Y, j) for j in range(n)])
    SE = TE / np.clip(PTE, 1e-9, None)

    def rts(j):
        if abs(TE[j] - 1) < 1e-7:
            return "CRS"
        if abs(NIRS[j] - PTE[j]) < 1e-6:
            return "DRS"
        if abs(NIRS[j] - TE[j]) < 1e-6:
            return "IRS"
        return "?"

    med_pte, med_se = np.median(PTE), np.median(SE)

    def quad(p, s_):
        # manuscript rule: a value equal to the median is assigned to the lower group
        if p > med_pte and s_ > med_se:
            return "I"
        if p > med_pte and s_ <= med_se:
            return "II"
        if p <= med_pte and s_ > med_se:
            return "III"
        return "IV"

    out = pd.DataFrame({
        "city": cities,
        "te_crs": TE, "te_crs_3dp": np.round(TE, 3),
        "pte_vrs": PTE, "pte_vrs_3dp": np.round(PTE, 3),
        "scale_efficiency": SE, "scale_efficiency_3dp": np.round(SE, 3),
        "nirs_score": np.round(NIRS, 4),
        "returns_to_scale": [rts(j) for j in range(n)],
        "quadrant": [quad(PTE[i], SE[i]) for i in range(n)],
    }).sort_values("te_crs", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(OUTDIR, "r1_dea.csv"), index=False)
    print(f"n={n} median PTE={med_pte:.6f} median SE={med_se:.6f}")
    print(out.to_string(index=False))
    print("\nfrontier CRS cities:", out.loc[out.te_crs >= 1 - 1e-7, "city"].tolist())


if __name__ == "__main__":
    main()
