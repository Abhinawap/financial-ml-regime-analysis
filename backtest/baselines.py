"""Momentum and mean-reversion baselines, run through the trade simulator."""
from dataclasses import dataclass

import pandas as pd
import pandas_ta_classic as ta

from backtest.settings import Settings
from backtest.simulator import simulate


@dataclass(frozen=True)
class BaselineResult:
    returns: pd.Series  # daily strategy returns over the out-of-sample dates, net of costs
    position: pd.Series  # position after each close, same index
    trades: pd.DataFrame  # simulator trade log
    n_trades: int


def baselines(prices: pd.DataFrame, index: pd.Index, settings: Settings) -> dict[str, BaselineResult]:
    """Run both baselines over `index` (the walk-forward series' dates), simulating like the arms."""
    c = prices["Close"]
    signals = {
        "momentum": c.pct_change(settings.momentum_lookback) > 0,
        "mean_reversion": ta.rsi(c, length=settings.rsi_length) < settings.rsi_entry_below,
    }
    out = {}
    for name, long in signals.items():
        sim = simulate(c, long.astype(int)[index], settings)  # off-index days count as "not +1"
        out[name] = BaselineResult(sim.returns[index], sim.position[index], sim.trades, len(sim.trades))
    return out
