import dataclasses
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from backtest.baselines import baselines
from backtest.features import (
    CLUSTERED_LEVEL,
    CLUSTERED_STATIONARY,
    STATIONARY_TWIN,
    V1_STYLE,
    Arm,
    stationary_features,
)
from backtest.settings import Settings
from backtest.simulator import simulate
from backtest.walkforward import triple_barrier_labels, walk_forward

S = Settings()
SMALL = dataclasses.replace(S, train_rows=120, test_rows=35, step=35, primary_n_estimators=5, primary_max_depth=2,
                            n_splits=2, n_permutation_repeats=1)


def prices(n, seed=0):
    """Seeded random-walk OHLCV on business days."""
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.015, n))
    o = c * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame(
        {"Open": o, "High": np.maximum(o, c) * 1.005, "Low": np.minimum(o, c) * 0.995, "Close": c,
         "Volume": rng.integers(1_000, 10_000, n).astype(float)},
        index=pd.bdate_range("2020-01-01", periods=n))


def run(p, settings=SMALL, seed=42, **kw):
    return walk_forward(p, V1_STYLE, seed=seed, settings=settings, **kw)


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


def test_a3_corrupting_prices_after_a_training_window_leaves_its_fingerprint_and_selection_unchanged():
    p = prices(300)
    base = run(p)
    k = 1
    bad = run(corrupt_after(p, p.index.get_loc(base.windows.train_end[k]))).windows
    assert bad.fingerprint[k] == base.windows.fingerprint[k]
    assert bad.features[k] == base.windows.features[k]
    assert len(base.windows.features[k]) == SMALL.n_features


@pytest.mark.parametrize("horizon", [4, 6])
def test_a3_purge_is_exactly_horizon_plus_one_rows(horizon):
    # Flat prices and a price-free feature: only labels can move the fingerprint, and any corrupted
    # close a kept label reads turns its timeout into a barrier hit.
    s = dataclasses.replace(TINY, horizon=horizon)
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=60))
    fp = lambda q: walk_forward(q, ROW_ARM, seed=42, settings=s, model=Stub()).windows.fingerprint[0]
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


ROW_ARM = Arm(row_feature, lags_outside_clustering=False)


TINY = dataclasses.replace(S, horizon=4, train_rows=30, test_rows=12, step=12)


def test_a4_position_open_at_a_window_boundary_carries_over_and_next_window_decides_exit():
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=54))  # flat: reviews are timeouts
    # Window 0 trains on rows up to 24 (30 minus the 5-row purge) and calls +1; window 1 calls 0.
    res = walk_forward(p, ROW_ARM, seed=42, settings=TINY, model=Stub(last_long_t=24))
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

    res = walk_forward(p, Arm(warm_features, lags_outside_clustering=False), seed=42, settings=TINY, model=Stub())
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


@pytest.mark.parametrize("arm", [V1_STYLE, STATIONARY_TWIN, CLUSTERED_LEVEL, CLUSTERED_STATIONARY])
def test_each_arm_runs_end_to_end_and_selects_per_window(arm):
    res = walk_forward(prices(300), arm, seed=42, settings=SMALL)
    assert len(res.windows) > 1 and res.returns.notna().all()
    assert all(0 < len(f) <= SMALL.n_features for f in res.windows.features)
    lags = {f"lag_{k}" for k in range(1, SMALL.n_lags + 1)}
    if arm.lags_outside_clustering:  # 30 unclustered lags alone fill the 15 slots' candidate pool
        assert all(len(f) == SMALL.n_features for f in res.windows.features)
    else:  # lags cluster: on a random walk they are near-collinear, so few survive
        assert all(len(lags & set(f)) < SMALL.n_features for f in res.windows.features)


def test_selection_seed_changes_selected_features_and_same_seed_reproduces_them():
    p = prices(300)
    sel = lambda seed: tuple(run(p, seed=seed).windows.features)
    assert sel(42) == sel(42)
    assert len({sel(seed) for seed in SMALL.selection_seeds}) > 1


def test_v1_style_arm_and_stationary_twin_differ_only_in_feature_form():
    assert dataclasses.replace(V1_STYLE, features=stationary_features) == STATIONARY_TWIN
    assert dataclasses.replace(CLUSTERED_LEVEL, features=stationary_features) == CLUSTERED_STATIONARY


class FoldSpy(BaseEstimator):
    """Primary stub: always calls +1, and its probabilities carry the last row it trained on."""

    def fit(self, X, y, sample_weight=None):  # noqa: V107 (sklearn signature)
        self.last_t_ = X["t"].max()
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), self.last_t_), X["t"], np.full(len(X), 1e9)])


class MetaStub(BaseEstimator):
    """Meta stub: constant meta probability; records the rows each fitted copy trained on."""
    seen: ClassVar[list] = []

    def __init__(self, p=0.4):
        self.p = p

    def fit(self, X, y, sample_weight=None):  # noqa: V107 (sklearn signature)
        MetaStub.seen.append(np.asarray(X)[np.asarray(sample_weight) > 0])
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def test_meta_trains_only_on_out_of_fold_predictions():
    MetaStub.seen.clear()
    s = dataclasses.replace(TINY, train_rows=65)
    p = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=80))
    walk_forward(p, ROW_ARM, seed=42, settings=s, model=FoldSpy(), meta=True, meta_model=MetaStub())
    first = MetaStub.seen[0]  # window 0: 60 purged training rows; five validation folds of 10 cover rows 10-59
    assert first.shape == (50, 3)
    last_trained, row = first[:, 0], first[:, 1]
    assert list(row) == list(range(10, 60))
    assert (last_trained < row - s.purge_length).all()  # no row's prediction came from a model that saw it


def test_meta_vetoes_calls_below_threshold_and_keeps_calls_at_it():
    p = prices(300)
    base = run(p)
    assert not base.trades.empty
    vetoed = run(p, meta=True, meta_model=MetaStub(0.4))
    assert vetoed.trades.empty and not (vetoed.predictions == 1).any()
    kept = run(p, meta=True, meta_model=MetaStub(0.5))
    pd.testing.assert_series_equal(kept.predictions, base.predictions)
    pd.testing.assert_frame_equal(kept.trades, base.trades)


def test_meta_off_is_the_base_run():
    p = prices(300)
    a, b = run(p), run(p, meta=False, meta_model=MetaStub(0.0))
    pd.testing.assert_series_equal(a.returns, b.returns)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_meta_ablation_runs_end_to_end_with_default_models():
    s = dataclasses.replace(SMALL, meta_n_estimators=5, meta_max_depth=2)
    res = walk_forward(prices(300), V1_STYLE, seed=42, settings=s, meta=True)
    assert res.returns.notna().all() and res.predictions.isin([-1, 0, 1]).all()


def test_meta_skips_a_window_with_no_out_of_fold_long_calls_and_keeps_the_base_calls():
    p = prices(300)
    p[["Open", "High", "Low", "Close"]] = p[["Open", "High", "Low", "Close"]].mul(0.998 ** np.arange(300), axis=0)
    s = dataclasses.replace(SMALL, meta_n_estimators=5, meta_max_depth=2)
    base, meta = run(p, settings=s), run(p, settings=s, meta=True)
    pd.testing.assert_series_equal(meta.predictions, base.predictions)
    assert (meta.windows["oof_long_calls"] == 0).any()
    assert base.windows["oof_long_calls"].isna().all()


def test_baselines_cover_the_same_dates_as_the_walk_forward_series():
    p = prices(300)
    idx = run(p).returns.index
    for r in baselines(p, idx, SMALL).values():
        assert (r.returns.index[0], r.returns.index[-1]) == (idx[0], idx[-1])
