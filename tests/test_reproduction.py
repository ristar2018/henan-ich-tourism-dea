#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the key reported numbers (G2-B corrected official 2024 revenue series).

Run from the repository root after `python run_all.py`:
    pytest -q
or directly:
    python tests/test_reproduction.py

Tolerances are deliberately loose for the Monte Carlo / bootstrap steps;
the deterministic DEA / OLS / GeoDetector results are checked tightly.
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "outputs"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def _path(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        sys.exit("Missing %s — run `python run_all.py` first." % p)
    return p


def approx(a, b, tol=2e-3):
    return abs(a - b) <= tol


def test_r1_primary_frontier_and_kf():
    d = pd.read_csv(_path("r1_dea.csv"))
    frontier = set(d.loc[d.te_crs > 0.9999, "city"])
    assert frontier == {"Zhengzhou", "Kaifeng", "Luoyang", "Anyang", "Nanyang"}, frontier
    kf = d.loc[d.city == "Kaifeng"].iloc[0]
    assert kf.returns_to_scale == "CRS"
    # lowest performers on the corrected official series
    hebi = d.loc[d.city == "Hebi"].iloc[0]
    assert approx(hebi.te_crs, 0.159, 5e-3)


def test_r4_ich_only():
    d = pd.read_csv(_path("r4_ich_only_dea.csv"))
    kf = d.loc[d.city == "Kaifeng"].iloc[0]
    assert approx(kf.te_ich_only_3_inputs, 0.608, 5e-3)


def test_r3_dual_output():
    d = pd.read_csv(_path("r3_dual_output_dea.csv"))
    fr = set(d.loc[d.dual_output_te > 0.9999, "city"])
    assert fr == {"Zhengzhou", "Kaifeng", "Luoyang", "Nanyang", "Anyang",
                  "Xinyang", "Luohe", "Hebi", "Jiyuan", "Puyang"}, fr
    assert approx(d.dual_output_te.mean(), 0.861, 1e-3)
    assert approx((d.dual_output_te - d.single_output_te).mean(), 0.248, 1e-3)


def test_r2_model_c():
    """On the corrected official output series Model C is no longer significant."""
    d = pd.read_csv(_path("r2_regressions.csv"))
    c = d[d.model == "C"].set_index("predictor")
    assert approx(c.loc["a_level_2024", "beta"], 0.092, 5e-4)
    assert approx(c.loc["traditional_villages", "beta"], -0.008, 5e-4)
    assert c.loc["a_level_2024", "p"] > 0.20          # non-significant
    assert c.loc["traditional_villages", "p"] > 0.90  # essentially null
    assert approx(c.loc["a_level_2024", "r2"], 0.095, 8e-3)
    assert approx(c.loc["a_level_2024", "joint_p"], 0.472, 1e-3)
    dens = d[d.model == "C_density"].set_index("predictor")
    assert dens.loc["a_level_density", "p"] > 0.10  # non-significant


def test_r0_composite_m2():
    """M2 composite = 0.5*[z(national) + z(provincial)], ddof=0; NOT 3*nat+prov."""
    d = pd.read_csv(os.path.join(DATA, "henan_ich_tourism_master.csv"))

    def z(v):
        v = v.astype(float)
        return (v - v.mean()) / v.std(ddof=0)

    expect = 0.5 * (z(d.national_ich_items) + z(d.provincial_ich_items))
    assert (d.ich_composite_index - expect).abs().max() < 1e-5
    # regression guard against the old 3:1 weighted definition
    zz = d.loc[d.city == "Zhengzhou", "ich_composite_index"].iloc[0]
    assert approx(zz, 1.828, 2e-3), zz


def test_r2_model_b_m2():
    d = pd.read_csv(_path("r2_regressions.csv"))
    b = d[d.model == "B"].set_index("predictor")
    assert approx(b.loc["composite", "beta"], 0.169, 5e-3)
    assert approx(b.loc["composite", "p"], 0.027, 5e-3)
    assert approx(b.loc["density", "beta"], -0.151, 5e-3)
    assert approx(b.loc["density", "p"], 0.045, 5e-3)
    assert approx(b.loc["composite", "r2"], 0.328, 5e-3)
    # both are nominally significant (p<0.05) but NOT BH-robust;
    # the cross-7-test BH values (0.157/0.157) are reported in the manuscript.


def test_r5_geodetector_natural():
    """GeoDetector outcome is ICH density -> invariant to the revenue correction."""
    d = pd.read_csv(_path("r5_geodetector_interaction.csv"))
    row = d[(d.pair == "Her_x_Pop") & (d.method == "natural")].iloc[0]
    assert approx(row.q, 0.563, 3e-3)
    assert int(row.n_strata) == 10
    assert int(row.n_singleton_strata) == 7
    rowg = d[(d.pair == "Her_x_GDP") & (d.method == "natural")].iloc[0]
    assert approx(rowg.q, 0.645, 3e-3)


def test_r7_revenue_monte_carlo():
    d = pd.read_csv(_path("r7_revenue_monte_carlo.csv")).set_index("design")
    assert approx(d.loc["independent_pm10", "rho_median"], 0.984, 0.02)
    assert d.loc["independent_pm10", "rho_p05"] > 0.90
    # a multiplicative common shock is CRS-neutral
    assert approx(d.loc["common_pm10", "rho_median"], 1.0, 1e-9)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print("\n%d/%d tests passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
