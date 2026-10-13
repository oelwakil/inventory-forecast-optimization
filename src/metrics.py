"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|e| / sum(y)."""
    return float(np.abs(y - yhat).sum() / np.maximum(y.sum(), 1e-9))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Relative bias: (sum(yhat) - sum(y)) / sum(y). Positive = over-forecast."""
    return float((yhat.sum() - y.sum()) / np.maximum(y.sum(), 1e-9))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def summarize(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "WAPE": wape(y, yhat),
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "Bias": bias(y, yhat),
    }
