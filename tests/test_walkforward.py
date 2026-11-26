import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from backtest.features import level_features
from backtest.settings import Settings
from backtest.simulator import simulate
from backtest.walkforward import triple_barrier_labels, walk_forward

S = Settings()
SMALL = dataclasses.replace(S, train_rows=120, test_rows=35, step=35, primary_n_estimators=5, primary_max_depth=2)


def prices(n, seed=0):
    """Seeded random-walk OHLCV on business days."""
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.015, n))
    o = c * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * 1.005, "Low": np.minimum(o, c) * 0.995, "Close": c,
         "Volume": rng.integers(1_000, 10_000, n).astype(float)},
        index=pd.bdate_range("2020-01-01", periods=n))


def run(p, settings=SMALL, **kw):
    return walk_forward(p, level_features, seed=42, settings=settings, **kw)


def test_a1_same_inputs_give_identical_series_trades_and_fingerprints():
    p = prices(300)
    a, b = run(p), run(p)
    pd.testing.assert_series_equal(a.returns, b.returns)
    pd.testing.assert_series_equal(a.predictions, b.predictions)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.windows, b.windows)
    assert len(a.windows) > 1 and not a.trades.empty


def corrupt_after(p, row):
    """Every OHLCV value after position `row` alternately halved and doubled."""
    out = p.copy()
    out.iloc[row + 1:] = out.iloc[row + 1:].mul(np.resize([0.5, 2.0], len(out) - row - 1), axis=0)
    return out


# Enough trees that predictions vary inside a test window, so a leaked future feature moves one.
SHARP = dataclasses.replace(SMALL, primary_n_estimators=30, primary_max_depth=4, primary_learning_rate=0.5)


@pytest.mark.parametrize("offset", [10, 40, 60, 75, 100])  # rows into the series, inside test windows
def test_a2_corrupting_prices_after_t_leaves_the_series_up_to_t_unchanged(offset):
    p = prices(300)
    base = run(p, SHARP)
    t = p.index.get_loc(base.predictions.index[0]) + offset
    bad, T = run(corrupt_after(p, t), SHARP), p.index[t]
    assert base.predictions[:T].nunique() > 1
    pd.testing.assert_series_equal(bad.returns[:T], base.returns[:T])
    pd.testing.assert_series_equal(bad.predictions[:T], base.predictions[:T])
    assert not bad.predictions.equals(base.predictions)  # the corruption did reach later rows


def test_a3_corrupting_prices_after_a_training_window_leaves_its_fingerprint_unchanged():
    p = prices(300)
    base = run(p)
    k = 1
    bad = corrupt_after(p, p.index.get_loc(base.windows.train_end[k]))
    assert run(bad).windows.fingerprint[k] == base.windows.fingerprint[k]


@pytest.mark.parametrize("horizon", [4, 6])
def test_a3_purge_is_exactly_horizon_plus_one_rows(horizon):
    # Flat prices and a price-free feature: only labels can move the fingerprint, and any corrupted
    # close a kept label reads turns its timeout into a barrier hit.
    s = dataclasses.replace(TINY, horizon=horizon)
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=60))
    fp = lambda q: walk_forward(q, row_feature, seed=42, settings=s, model=Stub()).windows.fingerprint[0]
    last = s.train_rows - 1
    assert fp(corrupt_after(p, last)) == fp(p)  # purge long enough
    assert fp(corrupt_after(p, last - 1)) != fp(p)  # and no longer: the last kept label reads the window's last close


class Stub(BaseEstimator):
    """Predicts +1 everywhere if trained on rows up to `last_long_t` only, else 0 everywhere."""

    def __init__(self, last_long_t=-1):
        self.last_long_t = last_long_t

    def fit(self, X, y, sample_weight=None):  # noqa: V107 (sklearn signature)
        self.long_ = X["t"].max() <= self.last_long_t
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), 3))
        proba[:, 2 if self.long_ else 1] = 1
        return proba


def row_feature(p, settings):
    return pd.DataFrame({"t": np.arange(len(p), dtype=float)}, index=p.index)


TINY = dataclasses.replace(S, horizon=4, train_rows=30, test_rows=12, step=12)


def test_a4_position_open_at_a_window_boundary_carries_over_and_next_window_decides_exit():
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=54))  # flat: reviews are timeouts
    # Window 0 trains on rows up to 24 (30 minus the 5-row purge) and calls +1; window 1 calls 0.
    res = walk_forward(p, row_feature, seed=42, settings=TINY, model=Stub(last_long_t=24))
    boundary = 42
    assert res.windows.test_start[1] == p.index[boundary]
    assert (res.predictions[:p.index[boundary - 1]] == 1).all() and (res.predictions[p.index[boundary]:] == 0).all()
    # Signal on row 30 -> entry 31; timeout reviews every 5 rows: 35 and 40 reset (+1), 45 is in window 1 (0) -> exit 46.
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert (t.entry_date, t.exit_date, t.resets) == (p.index[31], p.index[46], 2)
    assert (res.position[p.index[31]:p.index[45]] == 1).all() and res.position[p.index[46]] == 0


@pytest.mark.parametrize("step", [6, 18])
def test_step_other_than_test_rows_is_rejected(step):
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=66))
    with pytest.raises(ValueError, match="step"):
        walk_forward(p, row_feature, seed=42, settings=dataclasses.replace(TINY, step=step), model=Stub())


def test_a5_series_runs_from_after_first_training_window_to_last_row_with_short_final_window():
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=66))

    def warm_features(p, settings):  # warm-up: 7 rows for one feature, 3 for the other
        f = row_feature(p, settings)
        f["u"] = f["t"]
        f.iloc[:7, 0], f.iloc[:3, 1] = np.nan, np.nan
        return f

    res = walk_forward(p, warm_features, seed=42, settings=TINY, model=Stub())
    assert res.windows.train_start[0] == p.index[7]
    for s in (res.returns, res.position, res.predictions):
        assert s.index.equals(p.index[7 + 30:])
    # 29 test rows: 12 + 12 + a short final window of 5 ending on the last row.
    assert list(res.windows.test_start) == list(p.index[[37, 49, 61]])
    assert list(res.windows.test_end) == list(p.index[[48, 60, 65]])


def test_label_equals_simulator_outcome_of_the_same_single_trade():
    close = prices(120)["Close"]
    labels = triple_barrier_labels(close, S)
    h = S.horizon
    assert labels.iloc[-(h + 1):].isna().all() and labels.iloc[:-(h + 1)].notna().all()
    for i in range(len(close) - h - 2):  # leave room for the exit fill after the review day
        preds = pd.Series(0, index=close.index)
        preds.iloc[i] = 1
        t = simulate(close, preds, S).trades.iloc[0]
        entry, review = close.index.get_loc(t.entry_date), close.index.get_loc(t.exit_date) - 1
        move = close.iloc[review] / close.iloc[entry] - 1
        outcome = 1 if move >= S.upper_barrier else -1 if move <= -S.lower_barrier else 0
        assert entry == i + 1 and labels.iloc[i] == outcome
