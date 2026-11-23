"""Proximal Policy Optimization (PPO) for the inventory environment, in PyTorch.

One shared actor-critic network controls every store-item series. The actor outputs the
mean of a Gaussian over the residual safety-factor action ``a`` (see ``rl_env.py``); the
critic estimates the discounted future reward. Training uses generalised advantage
estimation (GAE), the clipped PPO objective, and episodes of fixed length that start at
random days of the training window. Episodes that end because of the time limit are
bootstrapped with the critic's value rather than treated as terminal.

The output layer of the actor starts at zero, so before training the policy *is* the
base-stock rule with safety factor ``z0``; PPO then learns where to deviate from it.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .rl_env import InventoryEnv, run_policy


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, init_log_std: float = -1.2):
        super().__init__()

        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
        self.actor = mlp(1)
        self.critic = mlp(1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor([init_log_std]))

    def dist(self, obs):
        mean = self.actor(obs).squeeze(-1)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        return self.actor(torch.from_numpy(obs)).squeeze(-1).numpy().astype(np.float64)


@dataclass
class PPOConfig:
    iterations: int = 150
    horizon: int = 42            # days per training episode
    gamma: float = 0.9           # an order's effect fades within about a week
    lam: float = 0.9
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    lr: float = 5e-4
    critic_warmup: int = 5       # iterations that train only the critic at the start
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    eval_every: int = 10
    init_log_std: float = -1.2
    seed: int = 0


@dataclass
class PPOLog:
    iteration: list[int] = field(default_factory=list)
    train_cost: list[float] = field(default_factory=list)     # mean cost per series-day in rollouts
    eval_iteration: list[int] = field(default_factory=list)
    eval_cost: list[float] = field(default_factory=list)      # full-window deterministic cost
    best_iteration: int = 0
    seconds: float = 0.0


def make_policy(net: ActorCritic, base_policy=None):
    """Deterministic policy: the actor's mean, added to ``base_policy`` if given."""
    if base_policy is None:
        return net.act_deterministic
    return lambda obs: base_policy(obs) + net.act_deterministic(obs)


def train_ppo(env: InventoryEnv, cfg: PPOConfig, base_policy=None,
              verbose: bool = False) -> tuple[ActorCritic, PPOLog]:
    """Train on ``env``'s window; keep the checkpoint with the lowest deterministic cost
    over the whole window (the same data the agent trains on; the test window is not used).

    With ``base_policy`` (a function obs -> action) the agent learns a *residual*: the
    action sent to the environment is base_policy(obs) + the agent's output, so an
    untrained agent reproduces the base policy exactly."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = ActorCritic(env.obs_dim, init_log_std=cfg.init_log_std)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    N, H = env.N, cfg.horizon
    max_start = env.T - H - 1
    log = PPOLog()

    def evaluate():
        return run_policy(env, make_policy(net, base_policy))["total_cost"]

    # rewards are rescaled so a typical day's reward is about -1 (stabilises the critic)
    probe = env.reset(rng.integers(0, max_start + 1, size=N))
    r_scale = 0.0
    for _ in range(10):
        _, r0, _ = env.step(np.zeros(N) if base_policy is None else base_policy(probe))
        probe = env.observe()
        r_scale += np.abs(r0).mean() / 10
    r_scale = max(r_scale, 1e-6)

    best_cost = evaluate()
    best_state = copy.deepcopy(net.state_dict())
    log.eval_iteration.append(0)
    log.eval_cost.append(best_cost)
    t0 = time.time()
    for it in range(1, cfg.iterations + 1):
        # ---- rollout: one episode per series, random start days ----------------------
        obs = env.reset(rng.integers(0, max_start + 1, size=N))
        O, A, LP, R, V = [], [], [], [], []
        costs = 0.0
        for _ in range(H):
            o = torch.from_numpy(obs)
            with torch.no_grad():
                d = net.dist(o)
                a = d.sample()
                lp = d.log_prob(a)
                v = net.value(o)
            act = a.numpy().astype(np.float64)
            if base_policy is not None:
                act = act + base_policy(obs)
            obs, r, info = env.step(act)
            O.append(o); A.append(a); LP.append(lp); V.append(v)
            R.append(torch.from_numpy((r / r_scale).astype(np.float32)))
            costs += info["cost"].mean()
        with torch.no_grad():
            v_last = net.value(torch.from_numpy(obs))      # bootstrap: time limit, not terminal
        # ---- GAE -------------------------------------------------------------------------
        adv = torch.zeros(H, N)
        gae = torch.zeros(N)
        for k in reversed(range(H)):
            nxt = v_last if k == H - 1 else V[k + 1]
            delta = R[k] + cfg.gamma * nxt - V[k]
            gae = delta + cfg.gamma * cfg.lam * gae
            adv[k] = gae
        ret = adv + torch.stack(V)
        O_, A_, LP_ = torch.cat(O), torch.cat(A), torch.cat(LP)
        adv_, ret_ = adv.reshape(-1), ret.reshape(-1)
        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
        # ---- PPO update ------------------------------------------------------------------
        n = len(A_)
        for _ in range(cfg.epochs):
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, cfg.minibatch):
                j = perm[i:i + cfg.minibatch]
                d = net.dist(O_[j])
                ratio = torch.exp(d.log_prob(A_[j]) - LP_[j])
                pg = -torch.min(ratio * adv_[j],
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_[j]).mean()
                vl = ((net.value(O_[j]) - ret_[j]) ** 2).mean()
                if it <= cfg.critic_warmup:
                    pg = pg * 0.0
                loss = pg + cfg.value_coef * vl - cfg.entropy_coef * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        log.iteration.append(it)
        log.train_cost.append(costs / H)
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            c = evaluate()
            log.eval_iteration.append(it)
            log.eval_cost.append(c)
            if c < best_cost:
                best_cost, best_state, log.best_iteration = c, copy.deepcopy(net.state_dict()), it
            if verbose:
                print(f"iter {it:3d}  rollout cost/series-day {costs / H:8.2f}  "
                      f"window cost {c:,.0f}  std {net.log_std.exp().item():.3f}")
    net.load_state_dict(best_state)
    log.seconds = time.time() - t0
    return net, log
