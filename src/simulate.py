"""Periodic-review, order-up-to inventory simulation with lost sales.

Timing convention (one step = one day):
  1. Orders scheduled to arrive today are received (start of day).
  2. Demand is served from on-hand stock; unmet demand is lost.
  3. At end of day t an order is placed so that the inventory position
     (on-hand + on-order) reaches the order-up-to level
         S_t = sum(forecast over the protection period) + z * sigma * sqrt(protection)
     where protection = lead_time + review_period - 1 days (days t+1 .. t+protection).
  4. The order arrives at the start of day t + lead_time.
"""
from __future__ import annotations

import numpy as np


def simulate_base_stock(
    demand: np.ndarray,
    forecast: np.ndarray,
    sigma: np.ndarray,
    z: float,
    lead_time: int = 2,
    review_period: int = 1,
    holding_cost: float = 1.0,
    lost_sale_cost: float = 20.0,
) -> dict:
    """Simulate all series at once.

    Parameters
    ----------
    demand : (T, N) realised demand over the evaluation window.
    forecast : (T + protection, N) point forecasts for the evaluation window plus
        a buffer of ``protection`` days after it (needed to size the last orders).
    sigma : (N,) standard deviation of daily forecast error, estimated on data
        that comes *before* the evaluation window.
    z : safety factor (standard-normal quantile of the target service level).
    """
    if lead_time < 1 or review_period < 1:
        raise ValueError("lead_time and review_period must be >= 1")
    T, N = demand.shape
    protection = lead_time + review_period - 1
    if forecast.shape[0] < T + protection:
        raise ValueError("forecast needs a buffer of `protection` extra days")

    safety = z * sigma * np.sqrt(protection)

    def order_up_to(t: int) -> np.ndarray:
        # sum of forecasts for days t+1 .. t+protection (t = -1 for the start)
        return forecast[t + 1 : t + 1 + protection].sum(axis=0) + safety

    on_hand = order_up_to(-1).copy()  # warm start: stocked for the first days
    arrivals = np.zeros((T + lead_time + 2, N))
    lost_total = np.zeros(N)
    sold_total = np.zeros(N)
    on_hand_sum = np.zeros(N)
    stockout_days = np.zeros(N)

    for t in range(T):
        on_hand = on_hand + arrivals[t]
        sold = np.minimum(on_hand, demand[t])
        lost = demand[t] - sold
        on_hand = on_hand - sold
        sold_total += sold
        lost_total += lost
        stockout_days += lost > 0
        on_hand_sum += on_hand

        if t % review_period == 0:
            position = on_hand + arrivals[t + 1 :].sum(axis=0)
            order = np.maximum(0.0, order_up_to(t) - position)
            arrivals[t + lead_time] += order

    total_demand = sold_total + lost_total
    cost = holding_cost * on_hand_sum + lost_sale_cost * lost_total
    return {
        "fill_rate": float(sold_total.sum() / total_demand.sum()),
        "in_stock_rate": float(1.0 - stockout_days.sum() / (T * N)),
        "avg_on_hand": float(on_hand_sum.sum() / (T * N)),  # units per series-day
        "total_cost": float(cost.sum()),
        "lost_units": float(lost_total.sum()),
        "per_series_fill_rate": sold_total / np.maximum(total_demand, 1e-9),
    }


def simulate_order_up_to(
    demand: np.ndarray,
    targets: np.ndarray,
    lead_time: int = 2,
    review_period: int = 1,
    holding_cost: float = 1.0,
    lost_sale_cost: float = 20.0,
) -> dict:
    """Same dynamics as ``simulate_base_stock`` but with order-up-to levels given directly.

    ``targets`` has shape (T + 1, N): row 0 is the starting stock (decision at t = -1)
    and row t + 1 is the order-up-to level used at the end of day t. This lets any
    policy that produces a target level per day (for example a quantile of the
    forecast demand distribution) run through the identical simulator.
    """
    T, N = demand.shape
    if targets.shape[0] < T + 1:
        raise ValueError("targets needs T + 1 rows")
    on_hand = np.maximum(targets[0], 0.0).copy()
    arrivals = np.zeros((T + lead_time + 2, N))
    lost_total = np.zeros(N)
    sold_total = np.zeros(N)
    on_hand_sum = np.zeros(N)
    stockout_days = np.zeros(N)
    n_orders = np.zeros(N)
    for t in range(T):
        on_hand = on_hand + arrivals[t]
        sold = np.minimum(on_hand, demand[t])
        lost = demand[t] - sold
        on_hand = on_hand - sold
        sold_total += sold
        lost_total += lost
        stockout_days += lost > 0
        on_hand_sum += on_hand
        if t % review_period == 0:
            position = on_hand + arrivals[t + 1 :].sum(axis=0)
            order = np.maximum(0.0, targets[t + 1] - position)
            n_orders += order > 0
            arrivals[t + lead_time] += order
    total_demand = sold_total + lost_total
    cost = holding_cost * on_hand_sum + lost_sale_cost * lost_total
    return {
        "fill_rate": float(sold_total.sum() / total_demand.sum()),
        "in_stock_rate": float(1.0 - stockout_days.sum() / (T * N)),
        "avg_on_hand": float(on_hand_sum.sum() / (T * N)),
        "total_cost": float(cost.sum()),
        "lost_units": float(lost_total.sum()),
        "per_series_fill_rate": sold_total / np.maximum(total_demand, 1e-9),
        "per_series_cost": cost,
        "per_series_sold": sold_total,
        "per_series_lost": lost_total,
        "per_series_on_hand": on_hand_sum / T,
    }


def normal_targets(forecast: np.ndarray, sigma: np.ndarray, z: float, T: int,
                   protection: int) -> np.ndarray:
    """Order-up-to levels of the classic rule: forecast sum + z * sigma * sqrt(protection)."""
    safety = z * sigma * np.sqrt(protection)
    return np.stack([forecast[t + 1 : t + 1 + protection].sum(axis=0) + safety
                     for t in range(-1, T)])
