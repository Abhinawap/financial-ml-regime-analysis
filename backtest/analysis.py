"""v2 analysis: the notebook's tables and reading outcomes, from the results command's files. Fits no model."""
import datetime
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.settings import Settings
from backtest.stats import (
    bootstrap_ci,
    break_even_cost,
    excess_returns,
    exposure_matched_benchmark,
    net_returns,
    paired_bootstrap_ci,
    sharpe,
)


def load(results: Path):
    """series, trades and windows from the results command's files, and the Settings its run.json records.

    Baselines and the benchmark are a single run with a blank seed; it is read as seed 0.
    """
    s = json.loads((results / "run.json").read_text())["settings"]
    settings = Settings(**{**s, "cost_stresses": tuple(map(tuple, s["cost_stresses"])),
                           "selection_seeds": tuple(s["selection_seeds"]),
                           "reading3_boundary": datetime.date.fromisoformat(s["reading3_boundary"])})
    series = pd.read_csv(results / "series.csv", parse_dates=["date"]).fillna({"seed": 0}).astype({"seed": int})
    trades = pd.read_csv(results / "trades.csv", parse_dates=["entry_date", "exit_date"])
    trades = trades.fillna({"seed": 0}).astype({"seed": int})
    windows = pd.read_csv(results / "windows.csv", parse_dates=["train_start", "train_end", "test_start", "test_end"])
    return series, trades, windows, settings


def per_seed(series, trades, run, settings: Settings, round_trip_bps=None) -> pd.DataFrame:
    """A run's daily net returns (date x seed) at round_trip_bps (default: the settings' cost), rebuilt from gross
    returns and trade dates: costs never change a decision, so no model reruns."""
    bps = settings.buy_cost_bps + settings.sell_cost_bps if round_trip_bps is None else round_trip_bps
    rows, t = series[series.run == run], trades[trades.run == run]
    return pd.DataFrame({
        seed: net_returns(g.set_index("date").gross_return,
                          list(zip(t.entry_date[t.seed == seed], t.exit_date[t.seed == seed])), bps, settings)
        for seed, g in rows.groupby("seed")
    })


def position(series, run) -> pd.Series:
    """Seed-averaged daily position."""
    return series[series.run == run].groupby("date").position.mean()


def benchmark(series) -> pd.Series:
    """Unscaled, cost-free buy-and-hold daily returns."""
    return series[series.run == "benchmark"].set_index("date")["return"]


def summary(series, trades, run, settings: Settings, round_trip_bps=None) -> dict:
    """One table row for a run's seed-averaged series at round_trip_bps (default: the settings' cost)."""
    seeds = per_seed(series, trades, run, settings, round_trip_bps)
    r, pos, bench = seeds.mean(axis=1), position(series, run), benchmark(series)
    gross = series[series.run == run].groupby("date").gross_return.mean()
    t = trades[trades.run == run]
    with np.errstate(invalid="ignore"):  # a run that never trades has no Sharpe: NaN
        seed_sharpes = sharpe(seeds, settings, axis=0)
        headline_sharpe = sharpe(r, settings), bootstrap_ci(r, settings, statistic=partial(sharpe, settings=settings))
    return {
        "strategy_return": (1 + r).prod() - 1,
        "sharpe": headline_sharpe[0],
        "sharpe_ci": headline_sharpe[1],
        "excess": excess_returns(r, bench, pos).mean(),
        "excess_ci": paired_bootstrap_ci(r, exposure_matched_benchmark(bench, pos), settings),
        "break_even": break_even_cost(gross, bench, pos, list(zip(t.entry_date, t.exit_date)), settings,
                                      n_runs=seeds.shape[1]),
        "trades_per_seed": len(t) / seeds.shape[1],
        "seed_sharpe_mean": seed_sharpes.mean(), "seed_sharpe_min": seed_sharpes.min(),
        "seed_sharpe_max": seed_sharpes.max(),
    }


def drawdown(returns) -> dict:
    """Max drawdown and longest time under water of daily returns (Amendment 7).

    Equity starts at 1 before the first day, which counts as a peak, and compounds the returns. A day is under
    water while equity is below its previous running peak; equalling or exceeding the peak ends the run. The
    longest run is in trading days, and `recovered` is False when it is still open on the final day (the later
    run wins a tie).
    """
    equity = np.cumprod(1 + np.asarray(returns, dtype="float64"))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[:-1]  # the running peak before each day
    longest = run = 0
    for under in equity < peak:
        run = run + 1 if under else 0
        longest = max(longest, run)
    return {"max_drawdown": float(np.max(1 - equity / peak, initial=0.0)),
            "days_under_water": longest, "recovered": run == 0 or run < longest}


def reading1(v1_style, stationary_twin, settings: Settings) -> dict:
    """Reading 1 on the primary pair, from each arm's per-seed daily returns (date x seed).

    Gate 1: the paired CI on the seed-averaged daily return difference excludes zero. Gate 2: the gap between
    the mean per-seed Sharpes exceeds the selection spread. The better arm is the one gate 1 favours, and the
    reading passes only if gate 2's gap points the same way (Amendment 4); otherwise it is inconclusive.
    """
    ci = paired_bootstrap_ci(v1_style.mean(axis=1), stationary_twin.mean(axis=1), settings)
    with np.errstate(invalid="ignore"):
        sa, sb = sharpe(v1_style, settings, axis=0), sharpe(stationary_twin, settings, axis=0)
    if np.isnan(sa).any() or np.isnan(sb).any():
        raise ValueError("a seed never trades, so it has no Sharpe and gate 2 has no selection spread")
    gap, spread = sa.mean() - sb.mean(), max(np.ptp(sa), np.ptp(sb))
    gate1, gate2 = ci[0] > 0 or ci[1] < 0, abs(gap) > spread
    if gate1 and gate2 and np.sign(ci[0]) == np.sign(gap):
        outcome = ("v1-style arm better: the level-feature theory is rejected" if gap > 0
                   else "stationary twin better: raw price levels hurt the strategy")
    else:
        outcome = "inconclusive"
    return {"diff_ci": ci, "sharpe_gap": gap, "selection_spread": spread, "gate1": gate1, "gate2": gate2,
            "outcome": outcome}



def _window_sides(index, windows, settings: Settings):
    """Per-day window number (1-based) and each window's side: early if most of its test days fall before the
    reading-3 boundary. The spec gives an even split no side, so that raises."""
    numbers = pd.Series(np.searchsorted(windows.test_start.sort_values(), index, side="right"), index=index)
    early = pd.Series(index < pd.Timestamp(settings.reading3_boundary), index=index).groupby(numbers).mean()
    if (early == 0.5).any():
        raise ValueError(f"windows {list(early.index[early == 0.5])} straddle the reading-3 boundary evenly")
    return numbers, (early > 0.5).map({True: "early", False: "late"})


def per_window(returns, positions, benchmark, windows, settings: Settings) -> pd.DataFrame:
    """Compounded strategy, exposure-matched benchmark (at each window's average position) and excess return per
    test window. windows: one run's test_start and test_end."""
    numbers, sides = _window_sides(returns.index, windows, settings)
    t = windows.sort_values("test_start").reset_index(drop=True)[["test_start", "test_end"]]
    t.index += 1
    t["side"] = sides
    t["strategy"] = (1 + returns).groupby(numbers).prod() - 1
    t["benchmark"] = (1 + exposure_matched_benchmark(benchmark, positions, numbers)).groupby(numbers).prod() - 1
    t["excess"] = t.strategy - t.benchmark
    return t


def reading3(returns, positions, benchmark, windows, settings: Settings) -> dict:
    """Descriptive: mean daily excess return over the early and the late windows, each window's benchmark at its
    own average position. Holds if the arm beats its benchmark early and trails it late. Not a test."""
    numbers, sides = _window_sides(returns.index, windows, settings)
    excess = excess_returns(returns, benchmark, positions, numbers).groupby(numbers.map(sides)).mean()
    return {"early": excess["early"], "late": excess["late"], "holds": bool(excess["early"] > 0 > excess["late"])}


def price_only(series, settings: Settings) -> pd.DataFrame:
    """Buy-and-hold return, annualised volatility and Sharpe before and after the reading-3 boundary, and over the
    boundary's calendar year. Uses prices only, whatever the models show."""
    b = benchmark(series)
    boundary = pd.Timestamp(settings.reading3_boundary)
    periods = {"early": b[b.index < boundary], "late": b[b.index >= boundary],
               f"calendar {boundary.year}": b[b.index.year == boundary.year]}
    return pd.DataFrame({name: {"start": r.index[0].date(), "end": r.index[-1].date(), "return": (1 + r).prod() - 1,
                                "volatility": r.std() * np.sqrt(settings.trading_days), "sharpe": sharpe(r, settings)}
                         for name, r in periods.items()}).T.infer_objects()
