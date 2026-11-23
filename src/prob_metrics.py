"""Scoring rules for probabilistic count forecasts, and protection-period quantiles.

Two predictive distributions are compared on the same footing:

* negative binomial NB(mean mu, variance mu + alpha mu^2) from the PyTorch model, and
* normal(mean = point forecast, sd = sigma) for a point forecast with the per-series
  error spread used by the classic safety-stock rule, discretised to integers by
  P(Y <= k) = Phi((k + 0.5 - mean) / sd).

Metrics: CRPS (lower is better; equals MAE for a point forecast), coverage of central
prediction intervals, and the randomised probability integral transform (PIT), which
is uniform when the forecast distribution is calibrated.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def nb_params(mu: np.ndarray, alpha: np.ndarray):
    """scipy's (n, p) for NB with mean mu and variance mu + alpha mu^2."""
    alpha = np.maximum(alpha, 1e-8)
    n = 1.0 / alpha
    p = n / (n + np.maximum(mu, 1e-9))
    return n, p


def cdf_grid(kind: str, a: np.ndarray, b: np.ndarray, kmax: int) -> np.ndarray:
    """P(Y <= k) for k = 0..kmax, shape (rows, kmax + 1)."""
    k = np.arange(kmax + 1)[None, :]
    if kind == "nb":
        n, p = nb_params(a, b)
        return stats.nbinom.cdf(k, n[:, None], p[:, None])
    if kind == "normal":
        return stats.norm.cdf((k + 0.5 - a[:, None]) / np.maximum(b[:, None], 1e-9))
    raise ValueError(kind)


def crps_discrete(y: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """CRPS of integer outcomes for a CDF given on k = 0..kmax (mass above kmax ~ 0)."""
    k = np.arange(cdf.shape[1])[None, :]
    step = (y[:, None] <= k).astype(float)
    return ((cdf - step) ** 2).sum(axis=1)


def randomized_pit(y: np.ndarray, cdf: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yi = y.astype(int)
    upper = cdf[np.arange(len(y)), yi]
    lower = np.where(yi > 0, cdf[np.arange(len(y)), np.maximum(yi - 1, 0)], 0.0)
    return lower + rng.uniform(size=len(y)) * (upper - lower)


def score(y: np.ndarray, kind: str, a: np.ndarray, b: np.ndarray, seed: int = 0,
          levels=(0.5, 0.8, 0.9, 0.95), chunk: int = 20000) -> dict:
    """CRPS, interval coverage and PIT values, computed in chunks to bound memory."""
    y = np.asarray(y, dtype=float).ravel()
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    kmax = int(max(y.max(), np.max(a) + 8 * np.sqrt(np.max(a) + np.max(a) ** 2 * 0.05) + 10)) + 1
    rng = np.random.default_rng(seed)
    crps, pit = [], []
    for i in range(0, len(y), chunk):
        sl = slice(i, i + chunk)
        c = cdf_grid(kind, a[sl], b[sl], kmax)
        crps.append(crps_discrete(y[sl], c))
        pit.append(randomized_pit(y[sl], c, rng))
    crps, pit = np.concatenate(crps), np.concatenate(pit)
    cov = {f"{int(l * 100)}% interval": float(np.mean(np.abs(pit - 0.5) <= l / 2)) for l in levels}
    return {"CRPS": float(crps.mean()), **cov, "pit": pit}


def protection_quantile(mu: np.ndarray, alpha: np.ndarray, q: float) -> np.ndarray:
    """q-quantile of demand summed over the protection period.

    ``mu`` and ``alpha`` have shape (P, ...) for the P days of the period. Days are
    treated as independent; the sum is moment-matched to a negative binomial (or a
    Poisson when the extra variance is negligible).
    """
    m = mu.sum(axis=0)
    v = (mu + alpha * mu ** 2).sum(axis=0)
    excess = v - m
    out = stats.poisson.ppf(q, np.maximum(m, 1e-9))
    nb = excess > 1e-3 * np.maximum(m, 1e-9)
    if np.any(nb):
        n = m[nb] ** 2 / excess[nb]
        p = m[nb] / v[nb]
        out[nb] = stats.nbinom.ppf(q, n, p)
    return out


def quantile_targets(mu: np.ndarray, alpha: np.ndarray, q: float, T: int,
                     protection: int) -> np.ndarray:
    """Order-up-to levels (T + 1, N): the q-quantile of protection-period demand."""
    return np.stack([protection_quantile(mu[t + 1 : t + 1 + protection],
                                         alpha[t + 1 : t + 1 + protection], q)
                     for t in range(-1, T)])


def poisson_alpha(mu: np.ndarray) -> np.ndarray:
    """Dispersion that makes NB collapse to Poisson (used to wrap point forecasts)."""
    return np.full_like(mu, 1e-9)
