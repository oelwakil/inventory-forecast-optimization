"""Synthetic demo data with the same schema as the Kaggle
"Store Item Demand Forecasting Challenge" training file (date, store, item, sales).

This is SIMULATED data. It exists so the pipeline and tests can run without a
download. It is not used for the results reported in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(
    n_stores: int = 10,
    n_items: int = 50,
    start: str = "2013-01-01",
    end: str = "2017-12-31",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    dow_profile = np.array([0.80, 0.85, 0.92, 1.00, 1.15, 1.30, 1.18])
    rows = []
    for s in range(1, n_stores + 1):
        store_scale = rng.uniform(0.7, 1.4)
        for i in range(1, n_items + 1):
            base = rng.uniform(8, 60) * store_scale
            amp = rng.uniform(0.1, 0.35)
            phase = rng.uniform(0, 2 * np.pi)
            growth = rng.uniform(0.0, 0.25)  # total growth over the whole period
            season = 1 + amp * np.sin(2 * np.pi * doy / 365.25 + phase)
            trend = 1 + growth * t / len(t)
            mean = base * dow_profile[dow] * season * trend
            sales = rng.poisson(mean)
            rows.append(
                pd.DataFrame({"date": dates, "store": s, "item": i, "sales": sales})
            )
    return pd.concat(rows, ignore_index=True)
