"""Seam B: the trade simulator, plus the single barrier function labels also use."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.settings import Settings


def first_barrier(close, entry: int, settings: Settings) -> tuple[int, int] | None:
    """First review of a long trade entered at close position `entry`.

    Returns (position, label): label +1 for the first close at or above the upper
    barrier, -1 at or below the lower barrier, 0 for the timeout `horizon` closes
    after entry. None if the data ends before any of these happens.
    """
    c = np.asarray(close, dtype=float)
    path = c[entry + 1 : entry + settings.horizon + 1] / c[entry] - 1
    hits = np.flatnonzero((path >= settings.upper_barrier) | (path <= -settings.lower_barrier))
    if hits.size:
        return entry + 1 + int(hits[0]), 1 if path[hits[0]] > 0 else -1
    if entry + settings.horizon < len(c):
        return entry + settings.horizon, 0
    return None


@dataclass(frozen=True)
class SimResult:
    returns: pd.Series  # daily strategy returns, net of costs
    position: pd.Series  # position held after each close: 0 or 1
    trades: pd.DataFrame  # entry_date, exit_date, resets, gross_return, net_return


def simulate(close: pd.Series, predictions: pd.Series, settings: Settings) -> SimResult:
    """Run the long-or-flat trading rules on daily closes and per-day predictions.

    A prediction is read on its own day's close; missing predictions count as "not +1".
    """
    return _simulate(close, predictions, settings, buy_at_start=False)


def buy_and_hold(close: pd.Series, settings: Settings) -> SimResult:
    """Buy at the first close, hold (every review resets), sell at the final close."""
    return _simulate(close, pd.Series(1, index=close.index), settings, buy_at_start=True)


def _simulate(close, predictions, settings, buy_at_start):
    c = close.to_numpy(dtype=float)
    long_signal = predictions.reindex(close.index).eq(1).to_numpy()
    buy, sell = settings.buy_cost_bps / 1e4, settings.sell_cost_bps / 1e4
    n = len(c)
    ret, pos = np.zeros(n), np.zeros(n, dtype=int)
    trades = []
    held, pending, entry, review, resets = 0, ("buy" if buy_at_start else None), 0, None, 0

    def close_trade(t):
        trades.append((close.index[entry], close.index[t], resets, c[t] / c[entry] - 1,
                       np.prod(1 + ret[entry : t + 1]) - 1))

    for t in range(n):
        if t:
            ret[t] = held * (c[t] / c[t - 1] - 1)
        if pending == "buy":
            ret[t] -= buy
            held, entry, resets = 1, t, 0
            review = first_barrier(c, t, settings)
        elif pending == "sell":
            ret[t] -= sell
            held = 0
            close_trade(t)
        pending = None
        # End of data: a final-day signal never fills; a buy that fills on the final close
        # is sold at that same close, paying both costs (rule-literal, spec "End of data").
        if t == n - 1:
            if held:
                ret[t] -= sell
                close_trade(t)
                held = 0
        elif not held:
            if long_signal[t]:
                pending = "buy"
        elif review is not None and review[0] == t:
            if long_signal[t]:  # reset: free, new horizon and barriers from the next close
                resets += 1
                review = first_barrier(c, t + 1, settings)
            else:
                pending = "sell"
        pos[t] = held

    cols = ["entry_date", "exit_date", "resets", "gross_return", "net_return"]
    return SimResult(pd.Series(ret, index=close.index), pd.Series(pos, index=close.index),
                     pd.DataFrame(trades, columns=cols))
