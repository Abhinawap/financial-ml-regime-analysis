"""Seam A: the walk-forward run, plus the triple-barrier labels it trains on."""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtest.features import Arm, lag_columns
from backtest.selection import (
    CLASSES,
    fit_all_classes,
    primary_model,
    purged_splitter,
    select_features,
)
from backtest.settings import Settings
from backtest.simulator import first_barrier, simulate


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
    # train_start, train_end (last window row, before purge), test_start, test_end, fingerprint,
    # features (tuple of the window's selected columns), oof_long_calls (out-of-fold +1 calls the meta-model
    # trained on; 0 means the meta step was skipped; NaN with meta off)
    windows: pd.DataFrame


def walk_forward(prices: pd.DataFrame, arm: Arm, seed: int, settings: Settings, model=None, meta: bool = False,
                 meta_model=None) -> WalkForwardResult:
    """Retrain on each purged training window, predict its test window, simulate one continuous series.

    Each window selects its features from `arm`'s feature form on its own purged rows; `seed` drives the
    MDA shuffle. `model` defaults to the primary XGBoost classifier and is used for selection and prediction.
    `meta` turns on the meta-labelling ablation: a +1 call trades only if `meta_model` (default: the meta
    XGBoost classifier), trained on the window's out-of-fold +1 calls, gives it probability >= meta_threshold.
    """
    if settings.step != settings.test_rows:  # otherwise test windows overlap or leave gaps in the series
        raise ValueError(f"step ({settings.step}) must equal test_rows ({settings.test_rows})")
    model = primary_model(settings) if model is None else model
    if meta_model is None:
        meta_model = XGBClassifier(n_estimators=settings.meta_n_estimators, max_depth=settings.meta_max_depth,
                                   random_state=settings.meta_seed)
    X = arm.features(prices, settings)
    outside = lag_columns(settings) if arm.lags_outside_clustering else []
    y = triple_barrier_labels(prices["Close"], settings)
    n, first = len(prices), int(np.argmax(X.notna().all(axis=1).to_numpy()))
    preds, windows = [], []
    for start in range(first, n - settings.train_rows, settings.step):
        test = start + settings.train_rows
        end = min(test + settings.test_rows, n)
        # Purge: drop the last rows, whose labels read closes past the window's last row.
        Xt, yt = X.iloc[start : test - settings.purge_length], y.iloc[start : test - settings.purge_length]
        cols = select_features(Xt, yt, settings, seed, outside, model)
        Xs = Xt[cols]
        codes = np.searchsorted(CLASSES, yt)
        m = fit_all_classes(model, Xs, codes, len(CLASSES))
        proba = m.predict_proba(X.iloc[test:end][cols])
        pred = CLASSES[proba.argmax(axis=1)]
        oof_long_calls = np.nan
        if meta:
            # Out-of-fold primary probabilities: each validation fold is predicted by a model that never saw it.
            # Limit: feature selection ran on the whole window (labels included) before these predictions, so
            # selection, not the fold models' parameters, has seen the predicted rows.
            folds = list(purged_splitter(settings).split(Xs))
            oof = np.vstack([fit_all_classes(model, Xs.iloc[tr], codes[tr], len(CLASSES)).predict_proba(Xs.iloc[va])
                             for tr, va in folds])
            val = np.concatenate([va for _, va in folds])
            calls = CLASSES[oof.argmax(axis=1)] == 1
            oof_long_calls = int(calls.sum())
            if calls.any():  # no out-of-fold +1 calls: skip the meta step, keep the window's +1 calls as the base run
                meta_fit = fit_all_classes(meta_model, pd.DataFrame(oof[calls]),
                                           (yt.to_numpy()[val][calls] == 1).astype(int), 2)
                pred[(pred == 1) & (meta_fit.predict_proba(pd.DataFrame(proba))[:, 1] < settings.meta_threshold)] = 0
        preds.append(pd.Series(pred, index=X.index[test:end]))
        fingerprint = hashlib.sha256(str(list(Xt.columns)).encode() + Xt.to_numpy().tobytes() + yt.to_numpy().tobytes())
        windows.append((X.index[start], X.index[test - 1], X.index[test], X.index[end - 1], fingerprint.hexdigest(),
                        tuple(cols), oof_long_calls))
    predictions = pd.concat(preds)
    sim = simulate(prices["Close"], predictions, settings)
    return WalkForwardResult(sim.returns[predictions.index], sim.position[predictions.index], predictions, sim.trades,
                             pd.DataFrame(windows, columns=["train_start", "train_end", "test_start", "test_end",
                                                            "fingerprint", "features", "oof_long_calls"]))
