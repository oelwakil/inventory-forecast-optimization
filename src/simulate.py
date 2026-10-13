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
