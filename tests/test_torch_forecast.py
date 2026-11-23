import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo_data import make_demo_data  # noqa: E402
from src.features import build_features, to_wide  # noqa: E402
from src.torch_forecast import NBNet, ensemble_predict, make_inputs, nb_nll, predict_nb, train_nb  # noqa: E402


def test_nb_nll_matches_scipy():
    y = torch.tensor([0.0, 3.0, 17.0, 40.0], dtype=torch.float64)
    mu = torch.tensor([2.0, 5.0, 20.0, 35.0], dtype=torch.float64)
    alpha = torch.tensor([0.5, 0.1, 0.02, 0.001], dtype=torch.float64)
    ours = nb_nll(y, torch.log(mu), alpha).item()
    n = 1 / alpha.numpy()
    p = n / (n + mu.numpy())
    ref = -stats.nbinom.logpmf(y.numpy(), n, p).mean()
    assert abs(ours - ref) < 1e-8


def _demo_frame():
    feat = build_features(to_wide(make_demo_data()))
    return feat[feat["roll_mean_91"].notna()].reset_index(drop=True)


def test_inputs_are_finite_and_use_only_lagged_features():
    feat = _demo_frame()
    inp = make_inputs(feat.iloc[:5000])
    assert np.isfinite(inp["num"]).all()
    # the target column must not influence the inputs
    changed = feat.iloc[:5000].copy()
    changed["y"] = changed["y"] * 10 + 7
    inp2 = make_inputs(changed)
    for k in inp:
        assert np.array_equal(inp[k], inp2[k])


def test_training_reduces_loss_and_predictions_are_valid():
    torch.set_num_threads(1)
    feat = _demo_frame().sample(20000, random_state=0)
    inp = make_inputs(feat)
    y = feat["y"].to_numpy()
    model, log = train_nb(inp, y, epochs=2, batch_size=1024, seed=0)
    assert log.train_nll[-1] < log.train_nll[0]
    mu, alpha = predict_nb(model, inp)
    assert (mu > 0).all() and (alpha > 0).all()


def test_ensemble_moment_matching_preserves_mean_and_variance():
    torch.manual_seed(0)
    feat = _demo_frame().iloc[:3000]
    inp = make_inputs(feat)
    models = [NBNet(inp["num"].shape[1]).eval() for _ in range(3)]
    preds = [predict_nb(m, inp) for m in models]
    mu, alpha = ensemble_predict(models, inp)
    mix_mean = np.mean([p[0] for p in preds], axis=0)
    mix_var = np.mean([p[0] + p[1] * p[0] ** 2 + p[0] ** 2 for p in preds], axis=0) - mix_mean ** 2
    assert np.allclose(mu, mix_mean)
    assert np.allclose(mu + alpha * mu ** 2, np.maximum(mix_var, mu), rtol=1e-6)
