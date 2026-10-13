import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo_data import make_demo_data  # noqa: E402
from src.features import FEATURES, MIN_LAG, build_features, to_wide  # noqa: E402


def _small_wide():
    df = make_demo_data(n_stores=2, n_items=3, start="2014-01-01", end="2016-12-31")
    return to_wide(df)


def test_features_ignore_data_newer_than_min_lag():
    """Changing sales on/after a cutoff must not change features for dates
    earlier than cutoff + MIN_LAG."""
    wide = _small_wide()
    cutoff = wide.index[800]

    corrupted = wide.copy()
    corrupted.loc[corrupted.index >= cutoff] = 99999

    f_clean = build_features(wide)
    f_bad = build_features(corrupted)

    safe = f_clean["date"] < cutoff + np.timedelta64(MIN_LAG, "D")
    a = f_clean.loc[safe, FEATURES].to_numpy(dtype=float)
    b = f_bad.loc[safe, FEATURES].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_features_do_use_older_data():
    """Sanity check that the test above can fail: features must react to
    changes in data that is at least MIN_LAG days old."""
    wide = _small_wide()
    cutoff = wide.index[800]
    changed = wide.copy()
    changed.loc[changed.index == cutoff] = 12345

    f_clean = build_features(wide)
    f_changed = build_features(changed)
    target_date = cutoff + np.timedelta64(MIN_LAG, "D")
    rows = f_clean["date"] == target_date
    assert not np.allclose(
        f_clean.loc[rows, "lag_28"].to_numpy(dtype=float),
        f_changed.loc[rows, "lag_28"].to_numpy(dtype=float),
    )
