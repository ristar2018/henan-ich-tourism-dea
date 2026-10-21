#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R7: tourism-output measurement-error sensitivity of the CRS SBM-DEA (2024).

Context: all 18 city tourism revenues are now official 2024 statistical-communique
figures, so this is no longer a stress test of reconstructed values. It is a
generic output measurement-error sensitivity analysis: we perturb the single
output multiplicatively and recompute the CRS SBM-DEA for B = 2,000 draws
(seed 20261017), recording Spearman rank stability of the efficiency vector and
retention of the empirical five-city frontier (Zhengzhou, Kaifeng, Luoyang,
Anyang, Nanyang).

Two perturbation regimes:
  * independent +/-10% : every city gets an independent U(0.9,1.1) factor.
  * independent +/-20% : every city gets an independent U(0.8,1.2) factor.
A common (province-wide accounting) shock is reported as a third diagnostic:
one shared U(0.9,1.1) factor multiplies every city; under CRS a pure common
output scaling is neutral, so rho == 1 by construction and this mainly documents
that level shifts do not drive the ranking.

Reads ../data/henan_ich_tourism_master.csv; writes ../outputs/r7_revenue_monte_carlo.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "henan_ich_tourism_master.csv"))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

INPUTS = ["national_ich_items", "provincial_ich_items",
          "provincial_representative_inheritors",
          "national_key_cultural_relic_units"]
X = df[INPUTS].to_numpy(float)
Y0 = df["tourism_revenue_100m_yuan_2024"].to_numpy(float)
cities = df.city.to_numpy()
n = len(cities)
frontier0 = {"Zhengzhou", "Kaifeng", "Luoyang", "Anyang", "Nanyang"}


def crs_te(X, y):
    n, m = X.shape
    out = np.empty(n)
    for j in range(n):
        nv = n + m
        c = np.zeros(nv)
        for k in range(m):
            c[n + k] = -1.0 / (m * X[j, k])
        A_ub = np.zeros((1, nv))
        A_ub[0, :n] = -y
        b_ub = np.array([-y[j]])
        A_eq = np.zeros((m, nv))
        b_eq = np.zeros(m)
        for k in range(m):
            A_eq[k, :n] = X[:, k]
            A_eq[k, n + k] = 1
            b_eq[k] = X[j, k]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0, None)] * nv, method="highs")
        out[j] = 1 + r.fun if r.success else np.nan
    return out


te0 = crs_te(X, Y0)
rng = np.random.default_rng(20261017)
B = 2000
rows = []
designs = [("independent_pm10", 0.9, 1.1, "independent"),
           ("independent_pm20", 0.8, 1.2, "independent"),
           ("common_pm10", 0.9, 1.1, "common")]
for name, lo, hi, kind in designs:
    rhos, retain = [], []
    for _ in range(B):
        if kind == "independent":
            f = rng.uniform(lo, hi, n)
        else:  # one province-wide common factor: CRS-neutral for ranking
            f = np.full(n, rng.uniform(lo, hi))
        te = crs_te(X, Y0 * f)
        rhos.append(spearmanr(te0, te).statistic)
        front = set(cities[te > 1 - 1e-7])
        # average share of the five baseline frontier cities that stay efficient
        retain.append(len(frontier0 & front) / len(frontier0))
    rhos = np.asarray(rhos)
    rows.append({"design": name, "B": B,
                 "rho_median": round(float(np.median(rhos)), 3),
                 "rho_p05": round(float(np.quantile(rhos, 0.05)), 3),
                 "rho_min": round(float(rhos.min()), 3),
                 "frontier_five_mean_retained_pct": round(100 * float(np.mean(retain)), 1)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTDIR, "r7_revenue_monte_carlo.csv"), index=False)
print(res.to_string(index=False))
