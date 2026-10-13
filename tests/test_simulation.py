import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulate import simulate_base_stock  # noqa: E402


def _toy(T=120, N=20, seed=1):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 40, size=N)
    demand = rng.poisson(mean, size=(T, N)).astype(float)
    return demand, mean


def test_perfect_foresight_gives_full_service():
    demand, _ = _toy()
    T, N = demand.shape
    buffer = np.vstack([demand, demand[:5]])  # protection <= 5 in this test
    out = simulate_base_stock(demand, buffer, np.zeros(N), z=0.0, lead_time=2)
    assert out["fill_rate"] > 0.999


def test_more_safety_stock_means_higher_service_and_more_inventory():
    demand, mean = _toy()
    T, N = demand.shape
    forecast = np.tile(mean, (T + 3, 1))
    sigma = np.sqrt(mean)
    results = [
        simulate_base_stock(demand, forecast, sigma, z=z, lead_time=2)
        for z in (0.0, 1.0, 2.0)
    ]
    fills = [r["fill_rate"] for r in results]
    stock = [r["avg_on_hand"] for r in results]
    assert fills[0] <= fills[1] <= fills[2]
    assert stock[0] < stock[1] < stock[2]


def test_cost_accounting():
    demand, mean = _toy(T=60, N=5)
    T, N = demand.shape
    forecast = np.tile(mean, (T + 3, 1))
    out = simulate_base_stock(
        demand, forecast, np.sqrt(mean), z=1.0, holding_cost=2.0, lost_sale_cost=10.0
    )
    expected = 2.0 * out["avg_on_hand"] * T * N + 10.0 * out["lost_units"]
    assert abs(out["total_cost"] - expected) < 1e-6


def test_forecast_buffer_is_required():
    demand, mean = _toy(T=30, N=3)
    forecast = np.tile(mean, (30, 1))  # no buffer
    try:
        simulate_base_stock(demand, forecast, np.sqrt(mean), z=1.0, lead_time=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
