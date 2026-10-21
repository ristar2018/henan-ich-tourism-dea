# -*- coding: utf-8 -*-
"""r0_composite_index.py
Derive the ICH-project-reserve composite index used in exploratory Model B.

Definition (subjective-weight free):
    composite_i = 0.5 * [ z(national_ich_items)_i + z(provincial_ich_items)_i ],
where z() is the sample z-score with population standard deviation (ddof=0),
matching the standardisation used for every regressor in r2_regressions.py.

Only the two ICH *project-list* counts enter (national and provincial lists);
provincial representative inheritors (people) and national key cultural-relic
protection units (physical heritage) are deliberately excluded so that the
index measures one concept: the city's stock of listed ICH projects. The two
administrative levels receive equal weight after standardisation, i.e. there
is no subjective national:provincial multiplier.

The column is written back into the municipal master table (idempotent).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=0)


def main():
    path = os.path.join(DATA, "henan_ich_tourism_master.csv")
    df = pd.read_csv(path)
    composite = 0.5 * (zscore(df["national_ich_items"]) +
                       zscore(df["provincial_ich_items"]))
    df["ich_composite_index"] = np.round(composite, 6)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[r0] ich_composite_index (M2: equal-weight z-mean of national and "
          f"provincial ICH items) updated for {len(df)} cities -> {path}")


if __name__ == "__main__":
    main()
