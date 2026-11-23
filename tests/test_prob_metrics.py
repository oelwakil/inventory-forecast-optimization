import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prob_metrics import cdf_grid, crps_discrete, protection_quantile, score  # noqa: E402


def test_crps_of_point_mass_is_absolute_error():
    y = np.array([0.0, 3.0, 10.0])
    kmax = 20
    k = np.arange(kmax + 1)
    point = np.array([4, 4, 4])
    cdf = (k[None, :] >= point[:, None]).astype(float)
    assert np.allclose(crps_discrete(y, cdf), np.abs(y - point))


def test_calibrated_forecast_has_nominal_coverage():
    rng = np.random.default_rng(0)
    mu = rng.uniform(5, 60, 40000)
    alpha = np.full_like(mu, 0.05)
    n = 1 / alpha
    y = rng.negative_binomial(n, n / (n + mu)).astype(float)
    out = score(y, "nb", mu, alpha)
    for level in (50, 80, 90, 95):
        assert abs(out[f"{level}% interval"] - level / 100) < 0.01
    assert stats.kstest(out["pit"], "uniform").pvalue > 0.001


def test_overdispersed_truth_breaks_poisson_coverage():
    rng = np.random.default_rng(1)
    mu = np.full(20000, 30.0)
    y = rng.negative_binomial(5, 5 / 35, size=20000).astype(float)  # var >> mean
    out = score(y, "nb", mu, np.full_like(mu, 1e-9))                 # Poisson forecast
    assert out["90% interval"] < 0.8


def test_protection_quantile_matches_simulation():
    rng = np.random.default_rng(2)
    mu = np.array([[20.0, 50.0], [25.0, 40.0]])      # 2 days x 2 series
    alpha = np.array([[0.05, 0.01], [0.05, 0.01]])
    n = 1 / alpha
    draws = rng.negative_binomial(n, n / (n + mu), size=(200000, 2, 2)).sum(axis=1)
    for q in (0.5, 0.9, 0.98):
        ours = protection_quantile(mu, alpha, q)
        mc = np.quantile(draws, q, axis=0)
        assert np.all(np.abs(ours - mc) <= 2)


def test_normal_grid_is_a_cdf():
    c = cdf_grid("normal", np.array([10.0]), np.array([3.0]), 40)
    assert np.all(np.diff(c[0]) >= 0) and c[0, -1] > 0.999
