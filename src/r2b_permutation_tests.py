#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2b: distribution-free permutation tests for the second-stage OLS models.

Companion to r2_regressions.py. The parametric OLS p-values in r2 rely on
asymptotic t/F approximations that are fragile at n=18. This script provides
exact-in-spirit non-parametric p-values by the Freedman-Lane residual
permutation procedure:

  For predictor j, fit the REDUCED model y ~ 1 + X_{-j}; obtain fitted values
  h_{-j} and residuals e_{-j}. For each of B permutations pi, build
  y* = h_{-j} + P_pi e_{-j}, refit the FULL model, and record the t-statistic
  for coefficient j. The two-sided permutation p-value is
      p = (1 + #{|t*| >= |t_obs|}) / (B + 1).
  For the overall (joint) model test the reduced model is the intercept-only
  regression, so y* is a permutation of y about its mean, and the permutation
  distribution of the overall F statistic is used.

Predictors are z-standardized with ddof=0 exactly as in r2, and the dependent
variable is the same te_crs column, so point estimates match r2_regressions.csv.

Reads only ../data, writes ../outputs/r2b_permutation_tests.csv.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)
MASTER = os.path.join(DATA, "henan_ich_tourism_master.csv")

N_PERM = 99_999
SEED = 20261018

# Deterministic per-model / per-term seed offsets (do NOT use Python's built-in
# hash(): it is randomized across processes via PYTHONHASHSEED and would make
# the permutation p-values non-reproducible).
_JOINT_OFFSET = {"A": 101, "B": 202, "C": 303, "C_density": 404}
_TERM_OFFSET = {
    ("A", "gdp"): 11, ("A", "population"): 12, ("A", "tertiary"): 13,
    ("B", "composite"): 21, ("B", "density"): 22,
    ("C", "a_level_2024"): 31, ("C", "traditional_villages"): 32,
    ("C_density", "a_level_density"): 41,
    ("C_density", "traditional_villages"): 42,
}


def zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=0)


def ols(y, X):
    """OLS with intercept supplied inside X. Returns beta, se, t, p-less, SSE, F."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sse = float(resid @ resid)
    dfd = n - k
    sigma2 = sse / dfd
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    # F for all slopes (intercept is column 0)
    p = k - 1
    sst = float(((y - y.mean()) ** 2).sum())
    F = ((sst - sse) / p) / (sse / dfd)
    from scipy import stats
    p_t = 2.0 * stats.t.sf(np.abs(t), dfd)
    p_F = stats.f.sf(F, p, dfd)
    return beta, se, t, p_t, sse, F, p_F, XtX_inv


def perm_predictor(y, X, j, rng, B):
    """Freedman-Lane permutation p for coefficient j (0=intercept)."""
    n = y.shape[0]
    cols = [c for c in range(X.shape[1]) if c != j]
    Xr = X[:, cols]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fitted = Xr @ br
    resid = y - fitted
    beta, se, t, p_t, sse, F, p_F, XtX_inv = ols(y, X)
    t_obs = t[j]
    k = X.shape[1]
    # cache full-model projection A = (X'X)^-1 X'
    A = XtX_inv @ X.T
    XtX_inv_jj = XtX_inv[j, j]
    dfd = n - k
    # batch permutations
    idx = np.argsort(rng.random((B, n)), axis=1)
    R = resid[idx]                  # (B,n) permuted residuals
    Ys = fitted[None, :] + R        # (B,n)
    Bs = Ys @ A.T                   # (B,k) coefficients
    Xy = Ys @ X                     # (B,k)
    sse_s = (Ys * Ys).sum(axis=1) - (Bs * Xy).sum(axis=1)
    sigma2_s = sse_s / dfd
    t_s = Bs[:, j] / np.sqrt(sigma2_s * XtX_inv_jj)
    p_perm = (1.0 + np.sum(np.abs(t_s) >= abs(t_obs) - 1e-12)) / (B + 1)
    return beta[j], t_obs, p_t[j], p_perm


def perm_jointF(y, X, rng, B):
    """Overall F permutation p; reduced model is intercept-only."""
    n = y.shape[0]
    k = X.shape[1]
    p = k - 1
    dfd = n - k
    beta, se, t, p_t, sse, F_obs, p_F, XtX_inv = ols(y, X)
    ymean = y.mean()
    resid = y - ymean
    A = XtX_inv @ X.T
    idx = np.argsort(rng.random((B, n)), axis=1)
    Ys = ymean + resid[idx]
    Bs = Ys @ A.T
    Xy = Ys @ X
    sst_s = ((Ys - ymean) ** 2).sum(axis=1)
    sse_s = (Ys * Ys).sum(axis=1) - (Bs * Xy).sum(axis=1)
    F_s = ((sst_s - sse_s) / p) / (sse_s / dfd)
    p_perm = (1.0 + np.sum(F_s >= F_obs - 1e-12)) / (B + 1)
    return F_obs, p_F, p_perm


def main():
    df = pd.read_csv(MASTER)
    assert len(df) == 18
    y = df["te_crs"].to_numpy(float)

    gdp, pop, ter = zscore(df.gdp_100m_yuan), zscore(df.permanent_population_10k), zscore(df.tertiary_share_pct)
    comp, dens = zscore(df.ich_composite_index), zscore(df.ich_density_per_1000km2)
    alev = zscore(df.a_level_scenic_areas_2024)
    area = df.area_km2.to_numpy(float)
    alev_dens = zscore(df.a_level_scenic_areas_2024.to_numpy(float) / area)
    villages = zscore(df.traditional_villages)

    models = [
        ("A", ["gdp", "population", "tertiary"], np.column_stack([gdp, pop, ter])),
        ("B", ["composite", "density"], np.column_stack([comp, dens])),
        ("C", ["a_level_2024", "traditional_villages"],
         np.column_stack([alev, villages])),
        ("C_density", ["a_level_density", "traditional_villages"],
         np.column_stack([alev_dens, villages])),
    ]

    rows = []
    for mname, names, Xslope in models:
        X = np.column_stack([np.ones(len(y)), Xslope])
        # independent RNG stream per model, but a single fixed seed for reproducibility
        rng = np.random.default_rng(SEED + _JOINT_OFFSET[mname])
        F_obs, pF_param, pF_perm = perm_jointF(y, X, rng, N_PERM)
        print("=" * 76)
        print(f"Model {mname}: F={F_obs:.3f} joint p_param={pF_param:.4f} "
              f"p_perm={pF_perm:.4f} (B={N_PERM})")
        rows.append({"model": mname, "scope": "joint_F", "term": "ALL",
                     "estimate": F_obs, "parametric_p": pF_param,
                     "permutation_p": pF_perm, "n_permutations": N_PERM})
        for j, nm in enumerate(names, start=1):
            rng2 = np.random.default_rng(SEED + _TERM_OFFSET[(mname, nm)])
            b_obs, t_obs, p_param, p_perm = perm_predictor(y, X, j, rng2, N_PERM)
            print(f"  {nm:20s} beta={b_obs:+.3f} t={t_obs:+.2f} "
                  f"p_param={p_param:.4f} p_perm={p_perm:.4f}")
            rows.append({"model": mname, "scope": "coefficient", "term": nm,
                         "estimate": b_obs, "parametric_p": p_param,
                         "permutation_p": p_perm, "n_permutations": N_PERM})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "r2b_permutation_tests.csv"), index=False)
    print("\nwrote r2b_permutation_tests.csv")


if __name__ == "__main__":
    main()
