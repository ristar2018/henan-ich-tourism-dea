#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the complete Henan ICH-tourism analysis pipeline.

Scripts live in src/ and read CSVs from ../data (relative to each script, i.e.
the repository's data/ folder) and write CSVs to the repository's outputs/
folder. Run from the repository root:

    python run_all.py

Set SKIP_BOOTSTRAP=1 to skip the slow Simar-Wilson bootstrap (r8) for a quick
check; r8 is still run in full verification.

Windows PowerShell:  $env:SKIP_BOOTSTRAP="1"; python run_all.py
Linux/macOS:         SKIP_BOOTSTRAP=1 python run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

SCRIPTS = [
    "r0_composite_index.py",       # ICH composite index for Model B (data prep)
    "r1_dea.py",                   # primary input-oriented SBM-DEA
    "r4_ich_only_dea.py",          # ICH-only three-input robustness DEA
    "r3_dual_output_dea.py",       # dual-output robustness DEA
    "r2_regressions.py",           # second-stage OLS Models A/B/C + BH
    "r2b_permutation_tests.py",   # Freedman-Lane permutation p-values (companion to r2)
    "r5_geodetector.py",           # GeoDetector factor/interaction detection
    "r7_revenue_monte_carlo.py",   # revenue uncertainty Monte Carlo
]
# NOTE: r6_spatial_statistics.py (NNI + global/bivariate Moran's I) is NOT on
# the main chain: it requires restricted spatial inputs (ICH item coordinates,
# municipal boundaries, traditional-village table) placed under
# ../spatial_inputs and an optional geo stack (geopandas, libpysal, esda).
# Run it manually:  python src/r6_spatial_statistics.py
if os.environ.get("SKIP_BOOTSTRAP") != "1":
    SCRIPTS.append("r8_simar_wilson.py")  # Simar-Wilson truncated bootstrap

for name in SCRIPTS:
    print("\n" + "=" * 78)
    print("RUNNING", name)
    print("=" * 78)
    r = subprocess.run([sys.executable, os.path.join(SRC, name)])
    if r.returncode != 0:
        print("FAILED:", name, "(exit code %d)" % r.returncode, file=sys.stderr)
        sys.exit(r.returncode)

print("\nAll scripts completed. Outputs are in",
      os.path.normpath(os.path.join(HERE, "outputs")))
