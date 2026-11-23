"""Probabilistic demand forecaster in PyTorch.

A single global network forecasts every store-item series. For each (series, day) it
outputs a full **negative-binomial** demand distribution, not only a point forecast:

    mean      mu    = scale * exp(f(x))
    variance  mu + alpha * mu^2,    alpha = softplus(g(x))

where ``scale`` is the series' recent level (28-day-lagged 91-day mean), so the network
learns relative patterns shared across series. Store, item, weekday and month enter
through learned embeddings; lag and rolling features enter as ratios to ``scale``.
Training minimises the negative-binomial negative log-likelihood, which rewards both an
accurate mean and an honest spread.

The model uses exactly the leakage-safe features from ``features.py``: every input for
date t comes from sales on or before t - 28.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import LAG_FEATURES, ROLL_FEATURES

COUNT_FEATURES = LAG_FEATURES + ROLL_FEATURES  # divided by the series scale


def _scale(df: pd.DataFrame, anchor: str = "roll_mean_365") -> np.ndarray:
    """Series level the network forecasts relative to (falls back to the 91-day mean)."""
    level = df[anchor].to_numpy(dtype=np.float64)
    level = np.where(np.isnan(level), df["roll_mean_91"].to_numpy(dtype=np.float64), level)
    return np.maximum(level, 1.0)


def make_inputs(df: pd.DataFrame, anchor: str = "roll_mean_365") -> dict[str, np.ndarray]:
    """Turn rows of the feature frame into network inputs (numpy arrays).

    ``anchor`` is the level that count features are divided by and that the output is
    multiplied by. The 365-day mean is free of seasonality, so the network learns the
    seasonal shape itself instead of inheriting the last quarter's level (chosen with a
    Q1-2016 backtest; see the notebook)."""
    s = _scale(df, anchor)
    counts = df[COUNT_FEATURES].to_numpy(dtype=np.float64)
    missing = np.isnan(counts)
    ratios = np.where(missing, 1.0, counts / s[:, None])
    ratios = np.log1p(np.clip(ratios, 0.0, 10.0))           # tame the rare spikes
    doy = df["dayofyear"].to_numpy(dtype=np.float64)
    dom = df["dom"].to_numpy(dtype=np.float64)
    cyc = np.column_stack([
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * dom / 31.0), np.cos(2 * np.pi * dom / 31.0),
    ])
    num = np.column_stack([ratios, missing.any(axis=1, keepdims=True).astype(float),
                           np.log(s)[:, None] / 5.0, cyc]).astype(np.float32)
    return {
        "num": num,
        "store": (df["store"].to_numpy() - 1).astype(np.int64),
        "item": (df["item"].to_numpy() - 1).astype(np.int64),
        "dow": df["dow"].to_numpy().astype(np.int64),
        "month": (df["month"].to_numpy() - 1).astype(np.int64),
        "log_scale": np.log(s).astype(np.float32),
    }


class NBNet(nn.Module):
    def __init__(self, n_num: int, n_store: int = 10, n_item: int = 50, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.emb_store = nn.Embedding(n_store, 4)
        self.emb_item = nn.Embedding(n_item, 12)
        self.emb_dow = nn.Embedding(7, 4)
        self.emb_month = nn.Embedding(12, 4)
        d_in = n_num + 4 + 12 + 4 + 4
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.head = nn.Linear(hidden // 2, 2)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([b["num"], self.emb_store(b["store"]), self.emb_item(b["item"]),
                       self.emb_dow(b["dow"]), self.emb_month(b["month"])], dim=1)
        out = self.head(self.body(x))
        log_mu = b["log_scale"] + out[:, 0]
        alpha = nn.functional.softplus(out[:, 1] - 3.0) + 1e-4   # starts near Poisson
        return log_mu, alpha


def nb_nll(y: torch.Tensor, log_mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of counts y under NB(mean=mu, var=mu+alpha*mu^2)."""
    r = 1.0 / alpha
    mu = torch.exp(log_mu)
    log_p = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
             + r * torch.log(r / (r + mu)) + y * (log_mu - torch.log(r + mu)))
    return -log_p.mean()


@dataclass
class TrainLog:
    epochs: list[int] = field(default_factory=list)
    train_nll: list[float] = field(default_factory=list)
    val_nll: list[float] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def _to_tensors(inp: dict[str, np.ndarray], idx=None) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v if idx is None else v[idx]) for k, v in inp.items()}


def train_nb(train_inp, y_train, val_inp=None, y_val=None, epochs: int = 12,
             batch_size: int = 2048, lr: float = 2e-3, weight_decay: float = 1e-5,
             seed: int = 0, verbose: bool = False) -> tuple[NBNet, TrainLog]:
    """Train with AdamW and a cosine schedule. With validation data, keep the best epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NBNet(train_inp["num"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(y_train)
    steps = epochs * int(np.ceil(n / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(y_train.astype(np.float32))
    tensors = _to_tensors(train_inp)
    log, best, best_state = TrainLog(), np.inf, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch_size):
            j = perm[i:i + batch_size]
            b = {k: v[j] for k, v in tensors.items()}
            loss = nb_nll(yt[j], *model(b))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        log.epochs.append(ep)
        log.train_nll.append(tot / n)
        if val_inp is not None:
            v = evaluate_nll(model, val_inp, y_val)
            log.val_nll.append(v)
            if v < best:
                best, best_state, log.best_epoch = v, copy.deepcopy(model.state_dict()), ep
        if verbose:
            print(f"epoch {ep:2d}  train NLL {log.train_nll[-1]:.4f}"
                  + (f"  val NLL {log.val_nll[-1]:.4f}" if val_inp is not None else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log.best_epoch = epochs
    log.seconds = time.time() - t0
    model.eval()
    return model, log


@torch.no_grad()
def predict_nb(model: NBNet, inp: dict[str, np.ndarray], batch_size: int = 65536):
    """Return (mu, alpha) numpy arrays."""
    model.eval()
    mus, alphas = [], []
    n = len(inp["num"])
    for i in range(0, n, batch_size):
        b = _to_tensors(inp, slice(i, i + batch_size))
        lm, a = model(b)
        mus.append(torch.exp(lm).numpy())
        alphas.append(a.numpy())
    return np.concatenate(mus).astype(np.float64), np.concatenate(alphas).astype(np.float64)


@torch.no_grad()
def evaluate_nll(model: NBNet, inp, y) -> float:
    mu, alpha = predict_nb(model, inp)
    return float(nb_nll(torch.from_numpy(y.astype(np.float64)), torch.from_numpy(np.log(mu)),
                        torch.from_numpy(alpha)).item())


def ensemble_predict(models: list[NBNet], inp) -> tuple[np.ndarray, np.ndarray]:
    """Average several seeds as a mixture, moment-matched back to one NB per row."""
    preds = [predict_nb(m, inp) for m in models]
    mu = np.mean([p[0] for p in preds], axis=0)
    second = np.mean([p[0] + (1 + p[1]) * p[0] ** 2 for p in preds], axis=0)  # E[Y^2]
    var = np.maximum(second - mu ** 2, mu * (1 + 1e-6))
    alpha = (var - mu) / np.maximum(mu ** 2, 1e-9)
    return mu, np.maximum(alpha, 1e-6)
