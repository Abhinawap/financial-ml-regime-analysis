"""Per-window feature selection: purged splitter, medoid clustering, seeded MDA."""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from backtest.settings import Settings

CLASSES = np.array([-1, 0, 1])  # triple-barrier label of each fixed class code


def purged_splitter(settings: Settings) -> TimeSeriesSplit:
    """Time-series splitter that drops `purge_length` training rows before each validation fold."""
    return TimeSeriesSplit(n_splits=settings.n_splits, gap=settings.purge_length)


def cluster_representatives(X: pd.DataFrame, settings: Settings, outside: Sequence[str] = ()) -> list[str]:
    """Medoid of each Ward cluster on 1 - |corr|, plus `outside` columns untouched, in column order."""
    outside = set(outside)
    cols = [c for c in X.columns if c not in outside]
    keep = set(outside)
    if len(cols) == 1:
        keep.update(cols)
    elif cols:
        corr = X[cols].corr().abs().fillna(0.0).clip(upper=1.0).to_numpy()
        np.fill_diagonal(corr, 1.0)
        tree = linkage(squareform(1.0 - corr, checks=False), method=settings.linkage)
        labels = fcluster(tree, t=settings.distance_threshold, criterion="distance")
        for k in np.unique(labels):
            idx = np.flatnonzero(labels == k)
            if len(idx) == 1:
                keep.add(cols[idx[0]])
                continue
            sub = corr[np.ix_(idx, idx)]
            score = (sub.sum(axis=1) - 1.0) / (len(idx) - 1)
            # round away float noise so exact-in-theory ties resolve to the earlier column
            keep.add(cols[idx[np.argmax(np.round(score, 12))]])
    return [c for c in X.columns if c in keep]


def primary_model(settings: Settings) -> XGBClassifier:
    return XGBClassifier(
        objective=settings.primary_objective,
        n_estimators=settings.primary_n_estimators,
        max_depth=settings.primary_max_depth,
        learning_rate=settings.primary_learning_rate,
        random_state=settings.model_seed,
    )


def fit_all_classes(model, X: pd.DataFrame, codes: np.ndarray, n_classes: int):
    """Fit a clone of `model` on class codes 0..n_classes-1, even if some are missing from `codes`.

    XGBoost needs every class present to size its output, so one zero-weight row is added per missing class.
    """
    missing = np.setdiff1d(np.arange(n_classes), codes)
    Xt = pd.concat([X, X.iloc[np.zeros(len(missing), dtype=int)]])
    yt = np.concatenate([codes, missing])
    w = np.concatenate([np.ones(len(codes)), np.zeros(len(missing))])
    return clone(model).fit(Xt, yt, sample_weight=w)


def mda_importance(
    X: pd.DataFrame, y: pd.Series, settings: Settings, seed: int, model=None
) -> pd.Series:
    """Mean log-loss increase from permuting each feature in the validation fold, sorted descending.

    Uses the purged splitter; the permutation shuffle is drawn from `seed`. `model` defaults to the
    primary XGBoost classifier (override with a small one in tests).
    """
    model = primary_model(settings) if model is None else model
    rng = np.random.default_rng(seed)
    codes = np.searchsorted(CLASSES, np.asarray(y))
    drops = np.zeros(X.shape[1])
    n_runs = 0
    for train, val in purged_splitter(settings).split(X):
        m = fit_all_classes(model, X.iloc[train], codes[train], len(CLASSES))

        def loss(Xv):
            return log_loss(codes[val], m.predict_proba(Xv), labels=range(len(CLASSES)))  # noqa: B023 (used only this iteration)

        Xv = X.iloc[val].copy()
        base = loss(Xv)
        for j, col in enumerate(X.columns):
            orig = Xv[col].to_numpy().copy()
            for _ in range(settings.n_permutation_repeats):
                Xv[col] = rng.permutation(orig)
                drops[j] += loss(Xv) - base
            Xv[col] = orig
        n_runs += settings.n_permutation_repeats
    # stable sort: ties keep column order
    return pd.Series(drops / n_runs, index=X.columns).sort_values(ascending=False, kind="stable")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
    seed: int,
    outside: Sequence[str] = (),
    model=None,
) -> list[str]:
    """Cluster to medoids (keeping `outside` columns), then keep the top `n_features` by MDA."""
    cand = cluster_representatives(X, settings, outside)
    if len(cand) <= settings.n_features:
        return cand
    imp = mda_importance(X[cand], y, settings, seed, model)
    return list(imp.index[: settings.n_features])
