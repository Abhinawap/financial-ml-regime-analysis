"""Seam A: the walk-forward run, plus the triple-barrier labels it trains on."""
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.selection import fit_all_classes, primary_model
from backtest.settings import Settings
from backtest.simulator import first_barrier, simulate

_CLASSES = np.array([-1, 0, 1])  # label of each fixed class code


def triple_barrier_labels(close: pd.Series, settings: Settings) -> pd.Series:
    """Label of each signal row: the barrier outcome of a long trade entered at the next close.

    Rows without a complete horizon after the entry close are NaN; nothing is padded.
    """
    n = len(close)
    labels = [first_barrier(close, i + 1, settings)[1] if i + 1 + settings.horizon < n else np.nan
              for i in range(n)]
    return pd.Series(labels, index=close.index, dtype=float)


@dataclass(frozen=True)
class WalkForwardResult:
    returns: pd.Series  # out-of-sample daily strategy returns, series start to last row
    position: pd.Series  # position after each close, same index
    predictions: pd.Series  # continuous prediction series (-1/0/+1), same index
    trades: pd.DataFrame  # simulator trade log
    windows: pd.DataFrame  # train_start, train_end (last window row, before purge), test_start, test_end, fingerprint


def walk_forward(prices: pd.DataFrame, features: Callable[[pd.DataFrame, Settings], pd.DataFrame],
                 seed: int, settings: Settings, model=None) -> WalkForwardResult:
    """Retrain on each purged training window, predict its test window, simulate one continuous series.

    Every column of `features(prices, settings)` is used. `seed` is accepted for per-window feature
    selection (issue #8) and unused for now. `model` defaults to the primary XGBoost classifier.
    """
    model = primary_model(settings) if model is None else model
    X = features(prices, settings)
    y = triple_barrier_labels(prices["Close"], settings)
    n, first = len(prices), int(np.argmax(X.notna().all(axis=1).to_numpy()))
    preds, windows = [], []
    for start in range(first, n - settings.train_rows, settings.step):
        test = start + settings.train_rows
        end = min(test + settings.test_rows, n)
        # Purge: drop the last rows, whose labels read closes past the window's last row.
        Xt, yt = X.iloc[start : test - settings.purge_length], y.iloc[start : test - settings.purge_length]
        m = fit_all_classes(model, Xt, np.searchsorted(_CLASSES, yt), len(_CLASSES))
        preds.append(pd.Series(_CLASSES[m.predict_proba(X.iloc[test:end]).argmax(axis=1)], index=X.index[test:end]))
        fingerprint = hashlib.sha256(str(list(Xt.columns)).encode() + Xt.to_numpy().tobytes() + yt.to_numpy().tobytes())
        windows.append((X.index[start], X.index[test - 1], X.index[test], X.index[end - 1], fingerprint.hexdigest()))
    predictions = pd.concat(preds)
    sim = simulate(prices["Close"], predictions, settings)
    return WalkForwardResult(sim.returns[predictions.index], sim.position[predictions.index], predictions, sim.trades,
                             pd.DataFrame(windows, columns=["train_start", "train_end", "test_start", "test_end",
                                                            "fingerprint"]))
