import numpy as np
import pandas as pd
import pytest

from backtest import stats


def _returns(n=300, seed=1):
    return pd.Series(np.random.default_rng(seed).normal(0.001, 0.02, n),
                     index=pd.bdate_range("2023-01-02", periods=n))


def test_c4_scaling_a_series_leaves_sharpe_unchanged():
    r = _returns()
    assert stats.sharpe(r * 0.37) == pytest.approx(stats.sharpe(r))


def test_sharpe_is_daily_mean_over_std_times_root_252():
    # mean 0.01, sample std 0.01 -> sqrt(252)
    assert stats.sharpe(pd.Series([0.0, 0.01, 0.02])) == pytest.approx(np.sqrt(252))


SMALL = stats.Settings(bootstrap_resamples=500)


def test_c1_series_paired_with_itself_gives_ci_exactly_zero():
    r = _returns()
    assert stats.paired_bootstrap_ci(r, r, settings=SMALL) == (0.0, 0.0)


def test_c2_same_seed_gives_same_interval_twice():
    r = _returns()
    first = stats.bootstrap_ci(r, stats.sharpe, settings=SMALL)
    assert stats.bootstrap_ci(r, stats.sharpe, settings=SMALL) == first
    assert first != stats.bootstrap_ci(r, stats.sharpe, settings=stats.Settings(bootstrap_resamples=500, bootstrap_seed=1))


def test_bootstrap_ci_brackets_the_point_estimate():
    r = _returns(n=1000)
    lo, hi = stats.bootstrap_ci(r, settings=SMALL)
    assert lo < r.mean() < hi


def test_paired_ci_resamples_the_same_days_for_both_series():
    # b = a + constant: any shared-day resampling gives exactly the constant difference
    a = _returns()
    lo, hi = stats.paired_bootstrap_ci(a + 0.001, a, settings=SMALL)
    assert lo == pytest.approx(0.001) and hi == pytest.approx(0.001)


def test_c3_strategy_equal_to_scaled_benchmark_has_zero_excess():
    bench = _returns(seed=2)
    pos = pd.Series(np.r_[np.ones(100), np.zeros(200)], index=bench.index)  # average 1/3
    strategy = bench / 3
    assert stats.excess_returns(strategy, bench, pos).abs().max() == pytest.approx(0.0)


def test_per_window_benchmark_scales_each_window_by_its_own_average_position():
    idx = pd.bdate_range("2023-01-02", periods=4)
    bench = pd.Series([0.01, 0.01, 0.02, 0.02], index=idx)
    pos = pd.Series([1.0, 0.0, 1.0, 1.0], index=idx)
    windows = pd.Series([1, 1, 2, 2], index=idx)
    out = stats.exposure_matched_benchmark(bench, pos, windows)
    assert out.tolist() == pytest.approx([0.005, 0.005, 0.02, 0.02])
    assert stats.exposure_matched_benchmark(bench, pos).tolist() == pytest.approx([0.0075, 0.0075, 0.015, 0.015])


def test_c5_seed_averaging_gives_the_daily_mean():
    idx = pd.bdate_range("2023-01-02", periods=3)
    runs = [pd.Series(v, index=idx) for v in ([0.0, 0.03, -0.01], [0.02, 0.01, 0.01], [0.01, 0.02, 0.03])]
    assert stats.seed_average(runs).tolist() == pytest.approx([0.01, 0.02, 0.01])


def _flat_benchmark_case():
    idx = pd.bdate_range("2023-01-02", periods=10)
    gross = pd.Series(0.001, index=idx)  # mean gross 0.001, benchmark zero
    bench = pd.Series(0.0, index=idx)
    pos = pd.Series(0.5, index=idx)
    trades = [(idx[0], idx[3]), (idx[5], idx[9])]
    return gross, bench, pos, trades


def test_break_even_known_value():
    # excess before costs sums to 0.01 over 2 round trips -> 0.005 per round trip = 50 bps
    gross, bench, pos, trades = _flat_benchmark_case()
    assert stats.break_even_cost(gross, bench, pos, trades) == pytest.approx(50.0)


def test_break_even_is_none_when_strategy_trails_benchmark_gross():
    gross, bench, pos, trades = _flat_benchmark_case()
    assert stats.break_even_cost(-gross, bench, pos, trades) is None


def test_recomputing_excess_at_break_even_gives_zero():
    gross, bench = _returns(seed=3), _returns(seed=4) * 0.5
    pos = pd.Series(0.6, index=gross.index)
    d = gross.index
    trades = [(d[i], d[i + 7]) for i in range(0, 280, 20)]
    c = stats.break_even_cost(gross, bench, pos, trades)
    net = stats.net_returns(gross, trades, c)
    assert stats.excess_returns(net, bench, pos).mean() == pytest.approx(0.0, abs=1e-15)


def test_scaled_costs_keep_the_10bps_sell_side_gap():
    gross, _, _, trades = _flat_benchmark_case()
    cost = gross - stats.net_returns(gross, trades, 150.0)
    d = gross.index
    assert cost[d[0]] == pytest.approx(0.0070) and cost[d[3]] == pytest.approx(0.0080)
    assert cost[d[1]] == 0.0


def test_seed_averaged_costs_are_divided_by_the_number_of_runs():
    gross, bench, pos, trades = _flat_benchmark_case()
    # same two trades in each of two runs -> each run's cost weighs 1/2, break-even unchanged
    assert stats.break_even_cost(gross, bench, pos, trades * 2, n_runs=2) == pytest.approx(50.0)
