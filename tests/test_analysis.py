import dataclasses
import datetime

import numpy as np
import pandas as pd
import pytest

from backtest.analysis import (
    load,
    per_seed,
    per_window,
    price_only,
    reading1,
    reading3,
    summary,
)
from backtest.results import main
from backtest.settings import Settings
from tests.test_results import TINY

FAST = dataclasses.replace(Settings(), bootstrap_resamples=500)
DAYS = pd.bdate_range("2022-06-06", periods=2000)


def noisy(mean, std, seed):
    """Five seeds' daily returns (date x seed) with the given mean and standard deviation."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(mean, std, (len(DAYS), 5)), index=DAYS, columns=range(42, 47))


def test_reading1_names_the_v1_style_arm_better_when_both_gates_favour_it():
    r = reading1(noisy(0.004, 0.01, 1), noisy(0.0, 0.01, 2), FAST)
    assert r["gate1"] and r["gate2"]
    assert r["outcome"] == "v1-style arm better: the level-feature theory is rejected"


def test_reading1_names_the_stationary_twin_better_when_both_gates_favour_it():
    r = reading1(noisy(0.0, 0.01, 1), noisy(0.004, 0.01, 2), FAST)
    assert r["outcome"] == "stationary twin better: raw price levels hurt the strategy"


def test_reading1_is_inconclusive_when_one_seed_widens_the_selection_spread_past_the_gap():
    v1 = noisy(0.004, 0.01, 1)
    v1[42] = noisy(-0.01, 0.01, 3)[42]  # one seed's features picked badly
    r = reading1(v1, noisy(0.0, 0.01, 2), FAST)
    assert r["gate1"] and not r["gate2"] and r["outcome"] == "inconclusive"


def test_reading1_is_inconclusive_when_the_gates_favour_different_arms():
    # v1-style: higher mean daily return, so gate 1 favours it; far noisier, so its per-seed Sharpes are lower.
    r = reading1(noisy(0.003, 0.05, 1), noisy(0.001, 0.002, 2), FAST)
    assert r["diff_ci"][0] > 0 and r["sharpe_gap"] < 0
    assert r["gate1"] and r["gate2"] and r["outcome"] == "inconclusive"


# Three 4-day test windows; window 2 has 3 of its 4 days before the boundary, so it is early.
WDAYS = pd.bdate_range("2023-01-02", periods=12)
WINDOWS = pd.DataFrame({"test_start": WDAYS[::4], "test_end": WDAYS[3::4]})
SPLIT = dataclasses.replace(FAST, reading3_boundary=datetime.date(2023, 1, 11))


def test_per_window_scales_each_window_benchmark_by_its_own_average_position_and_sides_by_most_days():
    returns = pd.Series([0.01, 0, 0, 0] + [0] * 8, index=WDAYS)
    positions = pd.Series([1, 1, 0, 0] + [1] * 8, index=WDAYS)
    benchmark = pd.Series([0.04, 0, 0, 0] + [0.01] * 8, index=WDAYS)
    t = per_window(returns, positions, benchmark, WINDOWS, SPLIT)
    assert t.side.tolist() == ["early", "early", "late"]
    assert np.isclose(t.strategy.iloc[0], 0.01) and np.isclose(t.benchmark.iloc[0], 0.02)  # 0.04 at position 0.5
    assert np.isclose(t.excess.iloc[0], -0.01)
    assert np.isclose(t.benchmark.iloc[1], 1.01 ** 4 - 1)


def test_reading3_holds_when_an_arm_beats_its_benchmark_early_and_trails_it_late():
    benchmark, positions = pd.Series(0.0, index=WDAYS), pd.Series(1, index=WDAYS)
    early_then_late = pd.Series([0.01] * 8 + [-0.01] * 4, index=WDAYS)
    assert reading3(early_then_late, positions, benchmark, WINDOWS, SPLIT)["holds"]
    assert not reading3(-early_then_late, positions, benchmark, WINDOWS, SPLIT)["holds"]


# Two seeds, and a boundary inside the development rows so both reading-3 sides exist.
TINY_RUN = dataclasses.replace(TINY, selection_seeds=(42, 43), reading3_boundary=datetime.date(2022, 1, 1))


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    main(["--out", str(out)], settings=TINY_RUN)
    return out


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return load(tiny_dir)


def test_load_restores_the_settings_the_run_used(tiny):
    assert tiny[3] == TINY_RUN


def test_per_seed_returns_rebuilt_from_gross_returns_and_trade_dates_match_the_saved_net_returns(tiny):
    series, trades, _, settings = tiny
    for run in set(series.run):
        saved = series[series.run == run].pivot(index="date", columns="seed", values="return")
        pd.testing.assert_frame_equal(per_seed(series, trades, run, settings), saved, check_names=False)


def test_excess_return_at_each_reported_break_even_cost_is_zero(tiny):
    # B10 cross-check.
    series, trades, _, settings = tiny
    checked = 0
    for run in set(series.run) - {"benchmark"}:
        be = summary(series, trades, run, settings)["break_even"]
        if be is not None and np.isfinite(be):
            assert abs(summary(series, trades, run, settings, round_trip_bps=be)["excess"]) < 1e-12
            checked += 1
    assert checked > 0


def test_price_only_splits_buy_and_hold_at_the_boundary_and_adds_the_boundary_calendar_year():
    days = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-12-31", "2025-01-02"])
    series = pd.DataFrame({"date": days, "run": "benchmark", "return": [0.0, 0.1, -0.1, -0.2, 0.2]})
    t = price_only(series, FAST)
    assert list(t.index) == ["early", "late", "calendar 2024"]
    assert np.allclose(t["return"], [0.1, 0.9 * 0.8 * 1.2 - 1, 0.72 - 1])
    assert np.isclose(t.loc["early", "volatility"], np.std([0.0, 0.1], ddof=1) * np.sqrt(252))


def test_reading1_refuses_a_seed_that_never_trades_instead_of_failing_gate_2_silently():
    v1 = noisy(0.004, 0.01, 1)
    v1[42] = 0.0  # no Sharpe, so no selection spread
    with pytest.raises(ValueError, match="never trades"):
        reading1(v1, noisy(0.0, 0.01, 2), FAST)


def test_a_window_split_evenly_by_the_boundary_has_no_side_and_stops_the_analysis():
    even = dataclasses.replace(FAST, reading3_boundary=datetime.date(2023, 1, 10))  # window 2: 2 days each side
    flat = pd.Series(0.0, index=WDAYS)
    with pytest.raises(ValueError, match="evenly"):
        per_window(flat, flat + 1, flat, WINDOWS, even)


def test_the_rehearsal_computes_every_reading_and_table_and_draws_the_readme_charts(tiny_dir):
    # Anything that reads results must run on rehearsal outputs before the official-run tag.
    from scripts.rehearsal import analyse
    analyse(tiny_dir)
    assert {p.name for p in (tiny_dir / "charts").iterdir()} == {"excess_ci.svg", "headline.svg"}
