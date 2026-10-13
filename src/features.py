"""Leakage-safe feature engineering for daily store-item demand.

Design rule: every feature for a target date ``t`` is computed only from sales
observed on or before ``t - MIN_LAG``. A single model can then forecast any
horizon up to ``MIN_LAG`` days ahead without recursive prediction, and the
features can never see the target period. ``tests/test_no_leakage.py`` checks
this property directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LAG = 28  # days between the last observed sales and the date being forecast
LAGS = (28, 35, 42, 49, 56, 364)
ROLL_WINDOWS = (7, 28, 91)

CALENDAR_FEATURES = ["dow", "dom", "month", "weekofyear", "dayofyear"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLL_FEATURES = (
    [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["roll_std_28", "roll_mean_365"]
)
CATEGORICAL = ["store", "item"]
FEATURES = CATEGORICAL + CALENDAR_FEATURES + LAG_FEATURES + ROLL_FEATURES


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long (date, store, item, sales) table to a date x series matrix.

    Requires a complete date x store x item grid.
    """
    wide = df.pivot_table(
        index="date", columns=["store", "item"], values="sales", aggfunc="sum"
    ).sort_index()
    if wide.isna().any().any():
        raise ValueError("Missing (date, store, item) combinations found.")
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    if len(full_index) != len(wide.index):
        raise ValueError("Dates are not contiguous daily.")
    return wide


def build_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Return a long frame with target ``y`` and leakage-safe features."""
    lagged = wide.shift(MIN_LAG)  # info available MIN_LAG days before each date
    feats: dict[str, np.ndarray] = {}

    for k in LAGS:
        feats[f"lag_{k}"] = wide.shift(k).to_numpy().ravel()
    for w in ROLL_WINDOWS:
        feats[f"roll_mean_{w}"] = lagged.rolling(w).mean().to_numpy().ravel()
    feats["roll_std_28"] = lagged.rolling(28).std().to_numpy().ravel()
    feats["roll_mean_365"] = (
        lagged.rolling(365, min_periods=90).mean().to_numpy().ravel()
    )

    n_dates, n_series = wide.shape
    dates = wide.index
    stores = np.array([c[0] for c in wide.columns])
    items = np.array([c[1] for c in wide.columns])

    out = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_series),
            "store": np.tile(stores, n_dates),
            "item": np.tile(items, n_dates),
            "y": wide.to_numpy().ravel(),
        }
    )
    d = pd.DatetimeIndex(out["date"])
    out["dow"] = d.dayofweek
    out["dom"] = d.day
    out["month"] = d.month
    out["weekofyear"] = d.isocalendar().week.to_numpy().astype(int)
    out["dayofyear"] = d.dayofyear
    for name, values in feats.items():
        out[name] = values
    return out


def baseline_forecasts(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Simple benchmarks that respect the same MIN_LAG information limit."""
    seasonal_naive = wide.shift(MIN_LAG)
    dow_avg = sum(wide.shift(MIN_LAG + 7 * i) for i in range(4)) / 4.0
    return {"Seasonal naive (t-28)": seasonal_naive, "Weekday average (4 wks)": dow_avg}
