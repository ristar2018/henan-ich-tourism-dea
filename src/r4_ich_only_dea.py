#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4: construct check that repeats the input-oriented CRS SBM-DEA using only the
three INTANGIBLE inputs (national ICH items, provincial ICH items, provincial
representative inheritors), excluding national key cultural-relic units.
Reports rank agreement with the main four-input model and per-city scores.
Reads ../data/henan_ich_tourism_master.csv; writes ../outputs/r4_ich_only_dea.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "henan_ich_tourism_master.csv"))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

FOUR = ["national_ich_items", "provincial_ich_items",
        "provincial_representative_inheritors",
        "national_key_cultural_relic_units"]
THREE = FOUR[:3]
Y = df[["tourism_revenue_100m_yuan_2024"]].to_numpy(float)


def iosbm(X, Yd, j, vrs=False):
    n, m = X.shape
    nv = n + m
    c = np.zeros(nv)
    for k in range(m):
        c[n + k] = -1.0 / (m * X[j, k])
    A_ub, b_ub = [], []
    row = np.zeros(nv)
    row[:n] = -Yd.ravel()
    A_ub.append(row)
    b_ub.append(-Yd[j, 0])
    A_eq, b_eq = [], []
    for k in range(m):
        row = np.zeros(nv)
        row[:n] = X[:, k]
        row[n + k] = 1
        A_eq.append(row)
        b_eq.append(X[j, k])
    if vrs:
        row = np.zeros(nv)
        row[:n] = 1
        A_eq.append(row)
        b_eq.append(1)
    rr = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                 bounds=[(0, None)] * nv, method="highs")
    return 1 + rr.fun if rr.success else np.nan


X4 = df[FOUR].to_numpy(float)
X3 = df[THREE].to_numpy(float)
n = len(df)
te4 = np.array([iosbm(X4, Y, j) for j in range(n)])
te3 = np.array([iosbm(X3, Y, j) for j in range(n)])
rho = stats.spearmanr(te4, te3).statistic
pear = np.corrcoef(te4, te3)[0, 1]

out = pd.DataFrame({"city": df.city, "te_4_inputs": np.round(te4, 4),
                    "te_ich_only_3_inputs": np.round(te3, 4),
                    "difference": np.round(te3 - te4, 4)}
                   ).sort_values("te_ich_only_3_inputs", ascending=False)
out.to_csv(os.path.join(OUTDIR, "r4_ich_only_dea.csv"), index=False)
print(out.to_string(index=False))
print(f"\nSpearman rho (4 vs 3 intangible inputs) = {rho:.3f}; Pearson r = {pear:.3f}")
print("Kaifeng falls from 1.000 to", round(float(te3[np.array(df.city == 'Kaifeng')][0]), 3))
