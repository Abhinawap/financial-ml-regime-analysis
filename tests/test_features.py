import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest.data import load_prices
from backtest.features import level_features, stationary_features
from backtest.settings import Settings

S = Settings()

PRICE_COLS = ["Open", "High", "Low", "Close"]
LAGS = [f"lag_{i}" for i in range(1, 31)]
PRICE_LEVEL = LAGS + ["Open", "High", "Low", "bb_mid", "bb_hi", "bb_lo",
                      "kc_upper", "kc_Lower", "ema_8", "ema_21", "sma_50", "vwap"]
SPREADS = ["macd", "macd_signal", "macd_hist", "atr"]
CUMULATIVE = ["obv", "ad"]
VOLUME_LEVEL = ["Volume", "vol20"]
STATIONARY = ["rsi", "stoch_k", "stoch_d", "cci", "roc", "natr", "bb_width", "cmf", "mfi",
              "ema_ratio", "adx", "dm_plus", "dm_minus", "aroon_up", "aroon_down"]
LEVEL_COLUMNS = (["Open", "High", "Low", "Close", "Volume"] + SPREADS + CUMULATIVE + STATIONARY
                 + ["bb_mid", "bb_hi", "bb_lo", "kc_upper", "kc_Lower", "ema_8", "ema_21",
                    "sma_50", "vwap", "vol20"] + LAGS)


@pytest.fixture(scope="module")
def prices():
    return load_prices()


@pytest.fixture(scope="module")
def level(prices):
    return level_features(prices, S)


@pytest.fixture(scope="module")
def stat(prices):
    return stationary_features(prices, S)


def test_exact_column_sets(level, stat):
    assert set(level.columns) == set(LEVEL_COLUMNS)
    assert len(level.columns) == 66
    assert set(stat.columns) == set(LEVEL_COLUMNS) - {"Close"}
    assert len(stat.columns) == 65
    assert level.index.equals(stat.index)


def test_level_form_is_v1_indicators(prices, level):
    # Independent check against v1's calls, not our own code path.
    import pandas_ta_classic as ta
    c = prices["Close"]
    pd.testing.assert_series_equal(level["rsi"], ta.rsi(c, length=14), check_names=False)
    pd.testing.assert_series_equal(level["Close"], c, check_names=False)
    pd.testing.assert_series_equal(level["lag_30"], c.shift(30), check_names=False)
    assert level["vol20"].equals(prices["Volume"].rolling(20).mean())


def scaled(prices, price=1.0, volume=1.0):
    p = prices.copy()
    p[PRICE_COLS] = p[PRICE_COLS] * price
    p["Volume"] = p["Volume"] * volume
    return p


def test_price_scaling_leaves_stationary_price_features_unchanged(prices, stat):
    s2 = stationary_features(scaled(prices, price=2.0 ** 10), S)
    for col in PRICE_LEVEL + SPREADS + CUMULATIVE + STATIONARY:
        pd.testing.assert_series_equal(s2[col], stat[col], check_exact=False, rtol=1e-12, atol=0,
                                       obj=col)


def test_price_scaling_changes_level_price_features(prices, level):
    l2 = level_features(scaled(prices, price=2.0 ** 10), S)
    for col in PRICE_LEVEL + ["Close"] + SPREADS:
        assert not np.allclose(l2[col].dropna(), level[col].dropna()), col


def test_volume_scaling_leaves_stationary_volume_features_unchanged(prices, stat):
    s2 = stationary_features(scaled(prices, volume=2.0 ** 10), S)
    for col in VOLUME_LEVEL + CUMULATIVE:
        pd.testing.assert_series_equal(s2[col], stat[col], check_exact=False, rtol=1e-12, atol=0,
                                       obj=col)


def test_atr_stationary_equals_natr_over_100(stat):
    pd.testing.assert_series_equal(stat["atr"], stat["natr"] / 100, check_names=False, rtol=1e-9)


def test_ad_stationary_equals_cmf(stat):
    pd.testing.assert_series_equal(stat["ad"], stat["cmf"], check_names=False, rtol=1e-9,
                                   atol=1e-12)


def test_volume_and_vol20_stationary_are_duplicates(prices, stat):
    pd.testing.assert_series_equal(stat["Volume"], stat["vol20"], check_names=False)
    expected = prices["Volume"] / prices["Volume"].rolling(20).mean()
    pd.testing.assert_series_equal(stat["Volume"], expected, check_names=False)


def test_price_level_transform_is_log_ratio_to_close(prices, stat):
    c = prices["Close"]
    pd.testing.assert_series_equal(stat["Open"], np.log(prices["Open"] / c), check_names=False)
    pd.testing.assert_series_equal(stat["lag_3"], np.log(c.shift(3) / c), check_names=False)


@pytest.mark.parametrize("form", [level_features, stationary_features])
def test_no_lookahead(prices, form):
    t = 400
    base = form(prices, S)
    changed = prices.astype("float64")
    rng = np.random.default_rng(0)
    n = len(prices) - t - 1
    changed.iloc[t + 1:, :] = changed.iloc[t + 1:, :].mul(rng.uniform(0.95, 1.05, n), axis=0)
    altered = form(changed, S)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], altered.iloc[: t + 1])


def test_n_lags_setting_controls_lag_columns(prices):
    f = level_features(prices, dataclasses.replace(S, n_lags=5))
    assert [c for c in f.columns if c.startswith("lag_")] == [f"lag_{k}" for k in range(1, 6)]
