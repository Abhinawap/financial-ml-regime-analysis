"""Seam C, statistics: functions that take return series and return numbers."""
import numpy as np
import pandas as pd

from backtest.settings import Settings


def sharpe(returns, settings: Settings, axis=None) -> float:
    """Annualised Sharpe of daily returns, risk-free rate zero."""
    r = np.asarray(returns, dtype=float)
    return np.mean(r, axis=axis) / np.std(r, axis=axis, ddof=1) * np.sqrt(settings.trading_days)


def _resample_indices(n: int, settings: Settings) -> np.ndarray:
    """Stationary block bootstrap (Politis-Romano): (resamples, n) day indices, wrapping at the end."""
    rng = np.random.default_rng(settings.bootstrap_seed)
    shape = (settings.bootstrap_resamples, n)
    starts = rng.integers(0, n, shape)
    new_block = rng.random(shape) < 1.0 / settings.bootstrap_block_length
    idx = np.empty(shape, dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(returns, settings: Settings, statistic=np.mean) -> tuple[float, float]:
    """Percentile CI of statistic(returns). statistic must accept axis= (np.mean, functools.partial(sharpe, settings=settings))."""
    r = np.asarray(returns, dtype=float)
    return _percentile_ci(statistic(r[_resample_indices(len(r), settings)], axis=1), settings.ci_alpha)


def paired_bootstrap_ci(a, b, settings: Settings, statistic=np.mean) -> tuple[float, float]:
    """Percentile CI of statistic(a) - statistic(b), resampling the same days for both series."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired series must have the same length")
    idx = _resample_indices(len(a), settings)
    return _percentile_ci(statistic(a[idx], axis=1) - statistic(b[idx], axis=1), settings.ci_alpha)


def exposure_matched_benchmark(benchmark_returns, positions, windows=None):
    """Buy-and-hold daily returns scaled by the average position, no costs.

    windows: optional per-day window labels; each window is then scaled by its own average position.
    """
    avg = positions.mean() if windows is None else positions.groupby(windows).transform("mean")
    return benchmark_returns * avg


def excess_returns(strategy_returns, benchmark_returns, positions, windows=None):
    """Daily strategy return minus the exposure-matched benchmark's daily return."""
    return strategy_returns - exposure_matched_benchmark(benchmark_returns, positions, windows)


def seed_average(series):
    """Day-by-day mean of several return series."""
    return pd.concat(list(series), axis=1).mean(axis=1)



def net_returns(gross_returns, trades, round_trip_bps: float, settings: Settings, n_runs: int = 1):
    """Gross daily returns minus costs at round_trip_bps, keeping the settings' sell-side gap.

    trades: (entry_date, exit_date) pairs; buy cost is charged on the entry day, sell cost on the exit day.
    n_runs: runs averaged into gross_returns (seed-averaged series pass all runs' trades and n_runs=5).
    """
    gap = settings.sell_cost_bps - settings.buy_cost_bps
    buy, sell = (round_trip_bps - gap) / 2 / 1e4, (round_trip_bps + gap) / 2 / 1e4
    costs = pd.Series(
        [buy] * len(trades) + [sell] * len(trades),
        index=[e for e, _ in trades] + [x for _, x in trades],
        dtype=float,
    )
    costs = costs.groupby(level=0).sum().reindex(gross_returns.index, fill_value=0.0) / n_runs
    return gross_returns - costs


def break_even_cost(gross_returns, benchmark_returns, positions, trades,
                    settings: Settings, n_runs: int = 1, windows=None) -> float | None:
    """Round-trip cost in bps at which mean excess return is zero; None if the strategy trails the benchmark gross."""
    gross_excess = excess_returns(gross_returns, benchmark_returns, positions, windows).mean()
    if gross_excess < 0:
        return None
    if not trades:
        return float("inf")
    # Mean excess falls linearly with cost: each round trip costs c / n_runs, spread over len(series) days.
    return float(gross_excess / (len(trades) / n_runs / len(gross_returns)) * 1e4)
