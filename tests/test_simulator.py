import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest.data import load_prices
from backtest.settings import Settings
from backtest.simulator import buy_and_hold, first_barrier, simulate

S = Settings()


def test_barrier_first_upper_crossing_on_closes():
    assert first_barrier([100, 101, 102.5, 97, 105], 0, S) == (2, 1)


def test_barrier_first_lower_crossing_on_closes():
    assert first_barrier([100, 99, 97.9, 105], 0, S) == (2, -1)


def test_barrier_measured_from_given_entry_close():
    # From entry 2 (close 110), 112.2 is exactly +2%.
    assert first_barrier([100, 120, 110, 111, 112.2], 2, S) == (4, 1)


def test_barrier_timeout_at_horizon():
    s = dataclasses.replace(S, horizon=3)
    assert first_barrier([100, 101, 100.5, 101.5, 103], 0, s) == (3, 0)


def test_barrier_none_when_data_ends_before_crossing_or_timeout():
    assert first_barrier([100, 101, 100.5], 0, S) is None


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def run(closes, preds, settings=S):
    close = series(closes)
    return simulate(close, pd.Series(preds, index=close.index), settings)


def test_b1_single_upper_barrier_trade_matches_hand_calculation():
    # Signal day 0 -> buy at close 1 (100). Close 2 (102) is +2%: review, prediction 0 -> sell at close 3.
    res = run([100, 100, 102, 105.06, 105.06], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(res.returns.to_numpy(), [0, -0.0045, 0.02, 0.03 - 0.0055, 0])
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 1, 1, 0, 0])
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t.entry_date == res.returns.index[1] and t.exit_date == res.returns.index[3]
    assert t.resets == 0
    assert t.gross_return == pytest.approx(0.0506)
    assert t.net_return == pytest.approx(0.9955 * 1.02 * 1.0245 - 1)


def test_b2_signal_moves_position_only_from_next_close():
    # Signal on day 2; the day-2 jump must not be earned, and the position starts at close 3.
    res = run([100, 100, 150, 150, 150, 150], [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 0, 0, 1, 1, 0])
    assert res.returns.iloc[2] == 0
    assert res.returns.iloc[3] == pytest.approx(-0.0045)


def test_b2_exit_decided_on_review_day_fills_next_close():
    # Review on day 3 (lower barrier), prediction 0: still long through close 3, flat from close 4.
    res = run([100, 100, 100, 90, 80, 80], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 1, 1, 1, 0, 0])
    assert res.returns.iloc[4] == pytest.approx(80 / 90 - 1 - 0.0055)
    assert res.trades.iloc[0].exit_date == res.returns.index[4]


def test_b3_reset_keeps_position_free_and_restarts_barriers_from_next_close():
    # Review day 2 (+2%) with +1: reset, new barriers from close 3 (102.5).
    # Close 4 (100.4) is below -2% of 102.5 only, so it is the next review; prediction 0 -> sell at close 5.
    res = run([100, 100, 102, 102.5, 100.4, 100, 100], [1, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 1, 1, 1, 1, 0, 0])
    assert res.returns.iloc[2] == pytest.approx(0.02)
    assert res.returns.iloc[3] == pytest.approx(102.5 / 102 - 1)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t.resets == 1
    assert (t.entry_date, t.exit_date) == (res.returns.index[1], res.returns.index[5])


def test_b3_reset_restarts_the_horizon_from_next_close():
    s = dataclasses.replace(S, horizon=2)
    # Flat prices: timeout review on day 3 (+1 -> reset from close 4), next timeout day 6 (0 -> sell at close 7).
    res = run([100] * 9, [1, 0, 0, 1, 0, 0, 0, 0, 0], s)
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 1, 1, 1, 1, 1, 1, 0, 0])
    assert res.trades.iloc[0].resets == 1


def test_b4_lower_crossing_with_plus_one_held_keeps_position():
    res = run([100, 100, 97, 96, 95, 95], [1, 0, 1, 0, 0, 0])
    assert res.position.iloc[3] == 1
    assert res.returns.iloc[3] == pytest.approx(96 / 97 - 1)


def test_b5_plus_one_on_exit_fill_day_reenters_next_close_with_full_buy_cost():
    # Review day 2, prediction 0 -> sell at close 3. The +1 on day 3 is an ordinary signal -> buy at close 4.
    res = run([100, 100, 102, 103, 103, 103, 103], [1, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(res.position.to_numpy(), [0, 1, 1, 0, 1, 1, 0])
    assert res.returns.iloc[3] == pytest.approx(103 / 102 - 1 - 0.0055)
    assert res.returns.iloc[4] == pytest.approx(-0.0045)
    assert len(res.trades) == 2
    assert res.trades.entry_date.iloc[1] == res.returns.index[4]


def test_b6_open_position_closes_at_final_close_and_final_signal_never_fills():
    res = run([100, 100, 101, 101.5], [1, 0, 0, 1])
    assert res.returns.iloc[-1] == pytest.approx(101.5 / 101 - 1 - 0.0055)
    assert res.position.iloc[-1] == 0
    assert len(res.trades) == 1 and res.trades.exit_date.iloc[0] == res.returns.index[-1]

    flat = run([100, 100, 101, 101.5], [0, 0, 0, 1])
    assert flat.trades.empty and (flat.returns == 0).all()


def test_b7_costs_charged_once_per_end_and_never_move_trade_dates():
    closes = [100, 100, 102, 102.5, 100.4, 100, 100, 101, 99, 99]
    preds = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    base = run(closes, preds)
    gross = base.position.shift(fill_value=0) * series(closes).pct_change().fillna(0)
    costs = (base.returns - gross).round(12)
    expected = pd.Series(0.0, index=costs.index)
    expected[base.trades.entry_date] = -0.0045
    expected[base.trades.exit_date] = -0.0055
    pd.testing.assert_series_equal(costs, expected)

    buy, sell = S.cost_stresses[1]
    stressed = run(closes, preds, dataclasses.replace(S, buy_cost_bps=buy, sell_cost_bps=sell))
    cols = ["entry_date", "exit_date", "resets", "gross_return"]
    pd.testing.assert_frame_equal(stressed.trades[cols], base.trades[cols])
    pd.testing.assert_series_equal(stressed.position, base.position)
    assert (stressed.trades.net_return < base.trades.net_return).all()


def test_b9_minus_one_or_zero_never_gives_negative_position():
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    res = run(closes, rng.choice([-1, 0, 1], 300))
    assert set(res.position.unique()) <= {0, 1}
    assert (run(closes, rng.choice([-1, 0], 300)).position == 0).all()


def test_barrier_overrides_change_simulator_behaviour():
    closes = [100, 100, 102, 103, 106, 106, 106]
    preds = [1, 0, 0, 0, 0, 0, 0]
    assert run(closes, preds).trades.exit_date.iloc[0] == series(closes).index[3]
    wide = run(closes, preds, dataclasses.replace(S, upper_barrier=0.05))
    assert wide.trades.exit_date.iloc[0] == series(closes).index[5]

    falling = [100, 100, 98.5, 97, 97, 97]
    assert run(falling, preds[:6]).trades.exit_date.iloc[0] == series(falling).index[4]
    loose = run(falling, preds[:6], dataclasses.replace(S, lower_barrier=0.01))
    assert loose.trades.exit_date.iloc[0] == series(falling).index[3]


def test_b8_buy_and_hold_is_start_to_end_return_minus_costs():
    res = buy_and_hold(series([100, 110, 121]), S)
    np.testing.assert_allclose(res.returns.to_numpy(), [-0.0045, 0.1, 0.1 - 0.0055])
    t = res.trades.iloc[0]
    assert len(res.trades) == 1 and t.resets >= 0
    assert t.gross_return == pytest.approx(0.21)
    assert t.net_return == pytest.approx(0.9955 * 1.1 * 1.0945 - 1)
    assert t.net_return == pytest.approx(0.21 - 0.0045 - 0.0055, abs=0.002)  # equal up to compounding


def test_buy_and_hold_runs_on_development_rows():
    close = load_prices()["Close"]
    res = buy_and_hold(close, S)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert (t.entry_date, t.exit_date) == (close.index[0], close.index[-1])
    assert t.gross_return == pytest.approx(close.iloc[-1] / close.iloc[0] - 1)
    assert (1 + res.returns).prod() - 1 == pytest.approx(t.net_return)

