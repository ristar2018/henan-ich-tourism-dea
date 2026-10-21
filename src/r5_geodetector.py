#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R5: GeoDetector factor-detector and interaction analysis.

Dependent variable: ICH density (national ICH items per 1,000 km^2).
Six candidate drivers: GDP, permanent population, urbanization rate, tertiary
industry share, mean elevation, and national key cultural-relic units. There is
no accessibility/road-density factor and no per-capita-GDP factor.

Discretization (k = 5 classes): quantile, equal interval, and natural breaks.
Natural breaks use an EXACT one-dimensional Fisher-Jenks dynamic program over
the within-class sum of squared deviations (the backtracking table records the
optimal split position). q = 1 - SSW/SST. Significance uses 9,999 permutations
of the outcome, p = (1 + #{q* >= q_obs})/(1 + B), seed 20261017.

Reads ../data/henan_ich_tourism_master.csv and writes
r5_geodetector_single.csv / r5_geodetector_interaction.csv to ../outputs.
These reproduce manuscript Tables 5 and 7.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "henan_ich_tourism_master.csv"))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

K = 5
B = 9999
SEED = 20261017

FACTORS = {
    "GDP": "gdp_100m_yuan",
    "Pop": "permanent_population_10k",
    "Urban": "urbanization_rate_pct",
    "Tertiary": "tertiary_share_pct",
    "Elev": "mean_elevation_m",
    "Her": "national_key_cultural_relic_units",
}


def discretize(x, method, k=K):
    x = np.asarray(x, dtype=float)
    if method == "quantile":
        return np.asarray(pd.qcut(x, k, labels=False, duplicates="drop"))
    if method == "equal":
        return np.clip(((x - x.min()) / (x.max() - x.min()) * k).astype(int), 0, k - 1)
    if method == "natural":
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        n = len(xs)
        cs = np.r_[0, np.cumsum(xs)]
        cs2 = np.r_[0, np.cumsum(xs ** 2)]

        def ssd(i, j):
            mm = j - i + 1
            s = cs[j + 1] - cs[i]
            sq = cs2[j + 1] - cs2[i]
            return sq - s * s / mm

        D = np.full((k + 1, n + 1), np.inf)
        Bk = np.zeros((k + 1, n + 1), int)
        D[0, 0] = 0
        for kk in range(1, k + 1):
            for j in range(kk, n + 1):
                best = np.inf
                bi = -1
                for i in range(kk - 1, j):
                    v = D[kk - 1, i] + ssd(i, j - 1)
                    if v < best:
                        best = v
                        bi = i
                D[kk, j] = best
                Bk[kk, j] = bi          # optimal split position
        cuts = []
        j = n
        for kk in range(k, 0, -1):
            i = Bk[kk, j]
            cuts.append(i)
            j = i
        cuts = sorted(set(cuts))[1:]
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        lab = np.zeros(n, int)
        bd = [0] + list(cuts) + [n]
        for c in range(len(bd) - 1):
            lab[bd[c]:bd[c + 1]] = c
        return lab[rank]
    raise ValueError(method)


def q_stat(y, groups):
    g = np.asarray(groups)
    n = len(y)
    total = n * y.var(ddof=0)
    ssw = sum(int(np.sum(g == c)) * y[g == c].var(ddof=0) for c in np.unique(g))
    return 1.0 - ssw / total


def permutation_p(y, groups, q_obs, rng, b=B):
    ge = 0
    yp = y.copy()
    for _ in range(b):
        rng.shuffle(yp)
        if q_stat(yp, groups) >= q_obs - 1e-12:
            ge += 1
    return (ge + 1) / (b + 1)


def stratum_info(groups):
    g = np.asarray(groups)
    sizes = [int(np.sum(g == c)) for c in np.unique(g)]
    return len(sizes), sum(s <= 1 for s in sizes)


def overlay(g1, g2):
    pairs = list(zip(np.asarray(g1), np.asarray(g2)))
    codes = {pair: i for i, pair in enumerate(sorted(set(pairs)))}
    return np.array([codes[p] for p in pairs])


def main():
    y = df["ich_density_per_1000km2"].to_numpy(float)
    n = len(y)
    vals = {key: df[col].to_numpy(float) for key, col in FACTORS.items()}
    methods = ["quantile", "equal", "natural"]
    rng = np.random.default_rng(SEED)

    grouped = {key: {m: discretize(vals[key], m) for m in methods} for key in FACTORS}
    single_rows = []
    for key in FACTORS:
        row = {"factor": key}
        for m in methods:
            g = grouped[key][m]
            q = q_stat(y, g)
            p = permutation_p(y, g, q, rng)
            row[f"q_{m}"] = round(q, 3)
            row[f"p_{m}"] = round(p, 3)
        single_rows.append(row)
    single = pd.DataFrame(single_rows)
    single.to_csv(os.path.join(OUTDIR, "r5_geodetector_single.csv"), index=False)

    inter_rows = []
    for other in ["Pop", "GDP", "Tertiary", "Elev", "Urban"]:
        for m in methods:
            gi = overlay(grouped["Her"][m], grouped[other][m])
            q = q_stat(y, gi)
            p = permutation_p(y, gi, q, rng)
            strata, singleton = stratum_info(gi)
            inter_rows.append({"pair": f"Her_x_{other}", "method": m,
                               "q": round(q, 3), "p_perm": round(p, 3),
                               "n_strata": strata, "n_singleton_strata": singleton})
    inter = pd.DataFrame(inter_rows)
    inter.to_csv(os.path.join(OUTDIR, "r5_geodetector_interaction.csv"), index=False)

    print("Single-factor q (outcome = ICH density):")
    print(single.to_string(index=False))
    print("\nProtection-unit interactions:")
    print(inter.to_string(index=False))


if __name__ == "__main__":
    main()
