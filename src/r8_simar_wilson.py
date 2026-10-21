#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R8: Simar-Wilson-style pseudo-frontier bootstrap for the second-stage
regression of CRS SBM technical efficiency on standardized external predictors.

Design notes
------------
Simar & Wilson (2007, Algorithm #2) characterise the sampling distribution of
second-stage coefficients by re-generating the DEA frontier in every draw and
re-estimating the second-stage model on the pseudo scores.  Their truncated
normal second stage is defined only on interior units: the CRS-efficient
municipalities sit exactly on the efficiency bound theta = 1, where the
truncated likelihood is unbounded (the truncated density at the bound diverges
as sigma -> 0), so those units cannot enter a truncated regression at all.
This script therefore keeps the frontier-resampling step but uses the ordinary
least-squares second-stage estimator on all n = 18 municipalities, so the
reported intervals are directly comparable with the all-sample OLS table of
the paper.  For transparency the interior-only truncated MLE (Simar-Wilson's
own regression sample, the 13 municipalities with theta < 1) is also computed
and printed: it reproduces the OLS slopes of the same subsample to within
0.007, i.e. the inverse-Mills truncation correction is numerically negligible
once the boundary units are excluded.  The all-sample and interior-subsample
slopes are written together to ../outputs/r8_interior_subsample.csv: they
show that the all-sample Model~B directions are carried by the five frontier
units (composite beta +0.169 on all 18 versus about -0.01 on the 13 interior
units), a sensitivity disclosed in the paper's Limitations section.

Input-oriented mirror of SW2007 step 3.1: the artificial error is drawn from
N(0, sigma_hat^2) right-truncated at (1 - z_i'beta_hat), so the pseudo score
theta_i* = z_i'beta_hat + eps_i never exceeds the efficiency bound.
Pseudo technology: x_i* = x_i * theta_hat_i / theta_i*.

Reads ../data/henan_ich_tourism_master.csv; writes ../outputs/r8_simar_wilson.csv
(all-18 OLS point slopes and the 1,000 pseudo-frontier bootstrap ranges) and
../outputs/r8_interior_subsample.csv (interior-only truncated-MLE and OLS
slopes for the 13 inefficient municipalities, as an audit diagnostic).
Models:
  A        : GDP, population, tertiary share
  B        : ICH composite, ICH density
  C_levels : A-level scenic areas (frozen 2024), traditional-village count
  C_density: A-level scenic-area density, traditional-village count
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog, minimize
from scipy.stats import norm, truncnorm

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "henan_ich_tourism_master.csv"))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

INPUTS = ["national_ich_items", "provincial_ich_items",
          "provincial_representative_inheritors",
          "national_key_cultural_relic_units"]
X = df[INPUTS].to_numpy(float)
y_rev = df["tourism_revenue_100m_yuan_2024"].to_numpy(float)
te = df["te_crs"].to_numpy(float)
area = df["area_km2"].to_numpy(float)
n, m = X.shape


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


MODELS = {
    "A": np.c_[z(df.gdp_100m_yuan), z(df.permanent_population_10k), z(df.tertiary_share_pct)],
    "B": np.c_[z(df.ich_composite_index), z(df.ich_density_per_1000km2)],
    "C_levels": np.c_[z(df.a_level_scenic_areas_2024), z(df.traditional_villages)],
    "C_density": np.c_[z(df.a_level_scenic_areas_2024.to_numpy(float) / area),
                       z(df.traditional_villages)],
}


def dea_crs(x, y):
    """CRS output-oriented-scaled (input-minimising) radial LP, one per DMU."""
    nn = len(y)
    mm = x.shape[1]
    scores = np.ones(nn)
    for k in range(nn):
        c = np.r_[1.0, np.zeros(nn)]
        A_ub = np.zeros((mm + 1, nn + 1))
        b_ub = np.zeros(mm + 1)
        A_ub[0, 1:] = -y
        b_ub[0] = -y[k]
        for r in range(mm):
            A_ub[1 + r, 0] = -x[k, r]
            A_ub[1 + r, 1:] = x[:, r]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (nn + 1),
                      method="highs")
        if res.success:
            scores[k] = res.x[0]
    return scores


def interior_trunc_mle(Zmat, u):
    """Truncated-normal MLE, input-oriented, right-truncation at theta = 1.

    Valid only when no observation sits on the bound, i.e. u < 1 strictly;
    used here as a diagnostic on the 13 inefficient municipalities.
    """
    p = Zmat.shape[1]
    ols = sm.OLS(u, Zmat).fit()

    def nll(th):
        b = th[:p]
        s = np.exp(th[p])
        mu = Zmat @ b
        zz = (u - mu) / s
        a = (1.0 - mu) / s
        ll = norm.logpdf(zz) - np.log(s) - norm.logcdf(a)
        if not np.all(np.isfinite(ll)):
            return 1e12
        return -np.sum(ll)

    th0 = np.concatenate([ols.params, [np.log(max(np.sqrt(ols.scale), 1e-3))]])
    res = minimize(nll, th0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-11})
    return res.x[:p], float(np.exp(res.x[p]))


B = 1000
rng = np.random.default_rng(20261017)
rows = []
interior_rows = []
for name, Z in MODELS.items():
    Zc = sm.add_constant(Z)
    fit = sm.OLS(te, Zc).fit()
    beta = fit.params
    sig = float(np.sqrt(fit.scale))

    interior = te < 1 - 1e-9
    b_int, _ = interior_trunc_mle(Zc[interior], te[interior])
    b_ols_int = sm.OLS(te[interior], Zc[interior]).fit().params
    print(f"[{name}] interior truncated MLE {np.round(b_int, 4)} vs interior OLS "
          f"{np.round(b_ols_int, 4)} (max abs diff {np.max(np.abs(b_int - b_ols_int)):.4f})")
    term_names = ["const"] + [f"z{j}" for j in range(1, Zc.shape[1])]
    for j, tname in enumerate(term_names):
        interior_rows.append({"model": name, "term": tname,
                              "beta_ols_all18": round(float(beta[j]), 4),
                              "beta_ols_interior13": round(float(b_ols_int[j]), 4),
                              "beta_truncmle_interior13": round(float(b_int[j]), 4)})

    boot = []
    for _ in range(B):
        mu = Zc @ beta
        eps = truncnorm.rvs(-np.inf, (1.0 - mu) / sig, loc=0.0, scale=sig,
                            random_state=rng)
        te_star = np.clip(mu + eps, 1e-3, 1.0)
        x_star = X * (te / te_star)[:, None]
        te_hat = dea_crs(x_star, y_rev)
        boot.append(sm.OLS(te_hat, Zc).fit().params)
    boot = np.asarray(boot)

    pcols = Zc.shape[1]
    for j in range(1, pcols):
        lo, hi = np.quantile(boot[:, j], [0.025, 0.975])
        rows.append({"model": name, "term": f"z{j}",
                     "beta_ols": round(beta[j], 3),
                     "boot_ci_low": round(lo, 3), "boot_ci_high": round(hi, 3),
                     "ci_contains_zero": bool(lo <= 0 <= hi),
                     "n_bootstrap": len(boot)})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTDIR, "r8_simar_wilson.csv"), index=False)
print(res.to_string(index=False))

intres = pd.DataFrame(interior_rows)
intres.to_csv(os.path.join(OUTDIR, "r8_interior_subsample.csv"), index=False)
print("\nInterior-only diagnostics (13 inefficient municipalities) written to "
      "r8_interior_subsample.csv")
print(intres.to_string(index=False))
