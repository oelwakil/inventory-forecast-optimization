"""Shared data preparation: splits, feature frame, and index helpers.

Both notebooks use the same splits and leakage-safe features from ``features.py``.
This module only packages that setup so the extension notebook can reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_features, to_wide

TRAIN_END = "2016-06-30"
VAL_END = "2016-12-31"
TEST_START, TEST_END = "2017-01-01", "2017-03-31"


@dataclass
class Splits:
    wide: pd.DataFrame
    feat: pd.DataFrame
    n_series: int
    tr_idx: np.ndarray        # train rows with a complete 91-day history
    va_idx: np.ndarray        # validation rows
    trva_idx: np.ndarray      # train + validation rows with complete history
    te_idx: np.ndarray        # test rows
    te_buf_idx: np.ndarray    # test rows plus `protection` days after the test end
    y_va: np.ndarray          # (T_val, N)
    y_te: np.ndarray          # (T_test, N)
    protection: int


def make_splits(raw: pd.DataFrame, lead_time: int = 2, review_period: int = 1,
                buffer_days: int | None = None) -> Splits:
    """Build features and row indices for the train / validation / test design.

    ``buffer_days`` extra forecast days after the test end are kept so the last
    orders can be sized (defaults to the protection period).
    """
    wide = to_wide(raw)
    feat = build_features(wide)
    dates = pd.DatetimeIndex(feat["date"])
    n_series = wide.shape[1]

    def rows(start, end):
        return ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).nonzero()[0]

    usable = feat["roll_mean_91"].notna().to_numpy().nonzero()[0]
    protection = lead_time + review_period - 1
    buf = protection if buffer_days is None else max(buffer_days, protection)
    val_start = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    te_end_buf = pd.Timestamp(TEST_END) + pd.Timedelta(days=buf)

    tr_idx = np.intersect1d(rows(wide.index.min(), TRAIN_END), usable)
    va_idx = rows(val_start, VAL_END)
    trva_idx = np.intersect1d(rows(wide.index.min(), VAL_END), usable)
    te_idx = rows(TEST_START, TEST_END)
    te_buf_idx = rows(TEST_START, te_end_buf)
    y = feat["y"].to_numpy()
    return Splits(wide, feat, n_series, tr_idx, va_idx, trva_idx, te_idx, te_buf_idx,
                  y[va_idx].reshape(-1, n_series).astype(float),
                  y[te_idx].reshape(-1, n_series).astype(float), protection)
