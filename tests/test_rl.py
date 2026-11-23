import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl_agent import ActorCritic, PPOConfig, make_policy, train_ppo  # noqa: E402
from src.rl_env import EnvConfig, InventoryEnv, run_policy, s_S_policy  # noqa: E402
from src.simulate import simulate_order_up_to  # noqa: E402


def _data(T=90, N=25, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(5, 40, (T + 8, N))
    alpha = np.full_like(mu, 0.02)
    n = 1 / alpha[:T]
    demand = rng.negative_binomial(n, n / (n + mu[:T])).astype(float)
    return demand, mu, alpha


def test_env_reproduces_simulator_for_base_stock():
    demand, mu, alpha = _data()
    for L in (1, 2, 3, 5):
        env = InventoryEnv(demand, mu, alpha, EnvConfig(lead_time=L), z0=1.1)
        r = run_policy(env, lambda o: np.zeros(len(o)))
        s = simulate_order_up_to(demand, env.m_prot + 1.1 * env.sd_prot, lead_time=L)
        assert np.isclose(r["total_cost"], s["total_cost"])
        assert np.isclose(r["fill_rate"], s["fill_rate"])
        assert np.isclose(r["avg_on_hand"], s["avg_on_hand"])


def test_scale_action_unit_and_sS_with_no_cover_equal_base_stock():
    demand, mu, alpha = _data()
    env_sd = InventoryEnv(demand, mu, alpha, EnvConfig(), z0=0.8)
    env_sc = InventoryEnv(demand, mu, alpha, EnvConfig(), z0=0.8, action_unit="scale")
    base = run_policy(env_sd, lambda o: np.zeros(len(o)))["total_cost"]
    ss = run_policy(env_sc, s_S_policy(env_sc, z_s=0.8, cover_days=0.0))["total_cost"]
    assert np.isclose(base, ss)


def test_order_cost_is_charged_per_order():
    demand, mu, alpha = _data()
    env0 = InventoryEnv(demand, mu, alpha, EnvConfig(order_cost=0.0), z0=1.0)
    envk = InventoryEnv(demand, mu, alpha, EnvConfig(order_cost=50.0), z0=1.0)
    r0 = run_policy(env0, lambda o: np.zeros(len(o)))
    rk = run_policy(envk, lambda o: np.zeros(len(o)))
    n_orders = rk["orders_per_series_day"] * envk.T * envk.N
    assert np.isclose(rk["total_cost"] - r0["total_cost"], 50.0 * n_orders)


def test_batching_reduces_orders():
    demand, mu, alpha = _data()
    env = InventoryEnv(demand, mu, alpha, EnvConfig(order_cost=100.0), action_unit="scale")
    daily = run_policy(env, s_S_policy(env, 1.0, 0.0))
    batched = run_policy(env, s_S_policy(env, 1.0, 3.0))
    assert batched["orders_per_series_day"] < 0.6 * daily["orders_per_series_day"]


def test_untrained_residual_agent_equals_base_policy():
    demand, mu, alpha = _data()
    env = InventoryEnv(demand, mu, alpha, EnvConfig(order_cost=100.0), action_unit="scale")
    base = s_S_policy(env, 0.5, 2.0)
    net = ActorCritic(env.obs_dim)
    assert np.isclose(run_policy(env, make_policy(net, base))["total_cost"],
                      run_policy(env, base)["total_cost"])


def test_ppo_runs_and_never_returns_a_worse_policy_on_its_window():
    torch.set_num_threads(1)
    demand, mu, alpha = _data(T=120)
    env = InventoryEnv(demand, mu, alpha, EnvConfig(order_cost=100.0), action_unit="scale")
    base = s_S_policy(env, 0.5, 2.0)
    net, log = train_ppo(env, PPOConfig(iterations=6, horizon=20, eval_every=3, minibatch=256), base_policy=base)
    assert min(log.eval_cost) <= log.eval_cost[0]
    final = run_policy(env, make_policy(net, base))["total_cost"]
    assert final <= log.eval_cost[0] + 1e-6
