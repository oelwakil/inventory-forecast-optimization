"""Vectorised lost-sales inventory environment for reinforcement learning.

The dynamics are identical to ``simulate.py`` (one step = one day): orders due today
arrive, demand is served from stock, unmet demand is lost, and at the end of the day an
order is placed that arrives ``lead_time`` days later. All series run in parallel.

Observation (per series, scaled by the series' typical daily demand):
    on-hand stock, the orders in transit, the forecast mean for each of the next
    ``lead_time + 1`` days, the mean and standard deviation of demand over the
    protection period, the weekday of tomorrow (one-hot), and whether an order was
    placed yesterday.

Action: a real number ``a``. The agent sets an order-up-to target
    target = m_prot + (z0 + a) * sd_prot          (action_unit = "sd", default)
    target = m_prot + z0 * sd_prot + a * scale    (action_unit = "scale")
and orders ``max(0, target - inventory position)``. ``scale`` is the series' typical
daily demand, so in the second form ``a`` is measured in days of demand, which suits
policies that order several days' worth at once. Because ``a`` may depend on the
whole observation (stock, pipeline, forecasts), any non-negative order quantity is
reachable, so this is a general ordering policy. ``a = 0`` is the classic base-stock
rule with safety factor ``z0``; the agent learns *residual* corrections to it.

Cost per day: holding_cost * on-hand at end of day + lost_sale_cost * lost units
+ order_cost per order placed. Reward = -cost / scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvConfig:
    lead_time: int = 2
    holding_cost: float = 1.0
    lost_sale_cost: float = 20.0
    order_cost: float = 0.0       # fixed cost per order placed (0 in the base scenario)


class InventoryEnv:
    """Forecast-driven inventory control over a window of real demand.

    Parameters
    ----------
    demand : (T, N) realised daily demand.
    mu, alpha : (T + lead_time + 2, N) negative-binomial forecast parameters for the
        same window plus a buffer (day 0 of the arrays = day 0 of ``demand``).
    z0 : base safety factor that ``a = 0`` corresponds to.
    """

    def __init__(self, demand, mu, alpha, cfg: EnvConfig, z0: float = 0.0,
                 action_unit: str = "sd"):
        self.d = np.asarray(demand, float)
        self.mu = np.asarray(mu, float)
        self.alpha = np.asarray(alpha, float)
        self.cfg = cfg
        self.z0 = z0
        if action_unit not in ("sd", "scale"):
            raise ValueError(action_unit)
        self.action_unit = action_unit
        self.T, self.N = self.d.shape
        L = cfg.lead_time
        if self.mu.shape[0] < self.T + L + 1:
            raise ValueError("forecasts need a buffer of lead_time + 1 days")
        self.protection = L  # daily review: protection = lead time
        self.scale = np.maximum(self.mu[: self.T].mean(axis=0), 1.0)
        var = self.mu + self.alpha * self.mu ** 2
        # protection-period mean / sd for a decision at end of day t (t = -1 .. T-1)
        self.m_prot = np.stack([self.mu[t + 1 : t + 1 + L].sum(0) for t in range(-1, self.T)])
        self.sd_prot = np.sqrt(np.stack([var[t + 1 : t + 1 + L].sum(0) for t in range(-1, self.T)]))
        self.dow = None  # optional (T + L + 2,) weekday index of each day, set by caller
        self.obs_dim = 1 + (L - 1) + (L + 1) + 2 + 7 + 1

    # ---- core -------------------------------------------------------------------------
    def target(self, t: np.ndarray, a: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Order-up-to level at end of day t (vector over the selected series)."""
        base = self.m_prot[t + 1, cols] + self.z0 * self.sd_prot[t + 1, cols]
        if self.action_unit == "sd":
            return base + a * self.sd_prot[t + 1, cols]
        return base + a * self.scale[cols]

    def reset(self, start: np.ndarray, cols: np.ndarray | None = None):
        """Start one episode per series at day ``start`` (array). Stock begins at the
        base-stock level, as in ``simulate.py``."""
        self.cols = np.arange(self.N) if cols is None else cols
        self.t = np.asarray(start, int).copy()
        n = len(self.cols)
        L = self.cfg.lead_time
        self.on_hand = np.maximum(self.target(self.t - 1, np.zeros(n), self.cols), 0.0)
        self.pipe = np.zeros((n, L))          # pipe[:, j] arrives j days after today
        self.last_ordered = np.zeros(n)
        return self._arrive_and_serve()

    def _arrive_and_serve(self):
        """Start of day t: receive, serve demand. Returns the observation for the
        end-of-day decision and stores this day's cost."""
        rows, c = self.t, self.cols
        self.on_hand = self.on_hand + self.pipe[:, 0]
        self.pipe = np.roll(self.pipe, -1, axis=1)
        self.pipe[:, -1] = 0.0
        dem = self.d[rows, c]
        sold = np.minimum(self.on_hand, dem)
        self.lost = dem - sold
        self.sold = sold
        self.on_hand = self.on_hand - sold
        self.day_cost = self.cfg.holding_cost * self.on_hand + self.cfg.lost_sale_cost * self.lost
        return self.observe()

    def observe(self) -> np.ndarray:
        rows, c = self.t, self.cols
        L = self.cfg.lead_time
        s = self.scale[c][:, None]
        fut = np.stack([self.mu[rows + k, c] for k in range(1, L + 2)], axis=1)
        parts = [self.on_hand[:, None] / s, self.pipe[:, : L - 1] / s, fut / s,
                 self.m_prot[rows + 1, c][:, None] / s, self.sd_prot[rows + 1, c][:, None] / s]
        dow = np.zeros((len(c), 7))
        if self.dow is not None:
            dow[np.arange(len(c)), self.dow[rows + 1]] = 1.0
        parts += [dow, self.last_ordered[:, None]]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def step(self, a: np.ndarray):
        """End of day t: order with action ``a``; then advance to day t + 1.

        Returns (next observation, reward for day t, info)."""
        L = self.cfg.lead_time
        position = self.on_hand + self.pipe.sum(axis=1)
        order = np.maximum(0.0, self.target(self.t, a, self.cols) - position)
        placed = order > 1e-9
        self.pipe[:, L - 1] += order          # arrives at the start of day t + L
        cost = self.day_cost + self.cfg.order_cost * placed
        reward = -cost / self.scale[self.cols]
        info = {"cost": cost, "lost": self.lost, "sold": self.sold, "on_hand": self.on_hand.copy(),
                "order": order, "placed": placed}
        self.last_ordered = placed.astype(float)
        self.t = self.t + 1
        return self._arrive_and_serve(), reward, info


def run_policy(env: InventoryEnv, policy, start: int = 0, steps: int | None = None) -> dict:
    """Run one deterministic pass over the window for all series; return totals.

    ``policy(obs) -> a`` (numpy). Matches ``simulate.py``'s accounting: on-hand is the
    end-of-day stock, fill rate = served / demand.
    """
    steps = env.T - start if steps is None else steps
    obs = env.reset(np.full(env.N, start))
    tot = {"cost": np.zeros(env.N), "lost": np.zeros(env.N), "sold": np.zeros(env.N),
           "on_hand": np.zeros(env.N), "orders": np.zeros(env.N)}
    for k in range(steps):
        a = policy(obs)
        last_day = k == steps - 1
        if last_day:  # record the final day's cost without needing forecasts past the window
            info = {"cost": env.day_cost + 0.0, "lost": env.lost, "sold": env.sold,
                    "on_hand": env.on_hand.copy(), "placed": np.zeros(env.N, bool)}
            # the order placed on the last day never arrives in the window; its fixed cost
            # is still charged for consistency with the simulator's daily ordering
            position = env.on_hand + env.pipe.sum(axis=1)
            order = np.maximum(0.0, env.target(env.t, a, env.cols) - position)
            info["placed"] = order > 1e-9
            info["cost"] = info["cost"] + env.cfg.order_cost * info["placed"]
        else:
            obs, _, info = env.step(a)
        tot["cost"] += info["cost"]
        tot["lost"] += info["lost"]
        tot["sold"] += info["sold"]
        tot["on_hand"] += info["on_hand"]
        tot["orders"] += info["placed"]
    demand = tot["sold"] + tot["lost"]
    return {
        "total_cost": float(tot["cost"].sum()),
        "fill_rate": float(tot["sold"].sum() / demand.sum()),
        "avg_on_hand": float(tot["on_hand"].sum() / (steps * env.N)),
        "orders_per_series_day": float(tot["orders"].sum() / (steps * env.N)),
        "per_series_cost": tot["cost"],
    }


def obs_layout(lead_time: int) -> dict[str, int | slice]:
    """Column positions in the observation vector (all stock quantities / scale)."""
    L = lead_time
    return {"on_hand": 0, "pipe": slice(1, L), "m_prot": 2 * L + 1, "sd_prot": 2 * L + 2}


def s_S_policy(env: InventoryEnv, z_s: float, cover_days: float):
    """Forecast-driven (s, S) rule expressed as an action for ``action_unit='scale'``.

    Reorder only when the inventory position falls below the reorder point
    s = m_prot + z_s * sd_prot; then order up to s + cover_days * (daily demand).
    Otherwise order nothing. This is the standard heuristic when each order has a fixed
    cost, and it is the benchmark the RL agent has to beat in that scenario.
    """
    if env.action_unit != "scale":
        raise ValueError("s_S_policy needs action_unit='scale'")
    lay = obs_layout(env.cfg.lead_time)

    def policy(obs):
        pos = obs[:, lay["on_hand"]] + obs[:, lay["pipe"]].sum(axis=1)
        m, sd = obs[:, lay["m_prot"]], obs[:, lay["sd_prot"]]
        s_level = m + z_s * sd
        reorder = pos < s_level
        # a is measured from the base target m + z0 * sd, in days of demand
        a_order = (s_level - m - env.z0 * sd) + cover_days
        # target one day of demand below the position: no order, with margin so that a
        # small learned correction on top of this rule does not trigger a tiny order
        a_skip = pos - m - env.z0 * sd - 1.0
        return np.where(reorder, a_order, a_skip).astype(np.float64)

    return policy
