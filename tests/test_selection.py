import dataclasses

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtest.selection import cluster_representatives, mda_importance, purged_splitter, select_features
from backtest.settings import Settings

S = Settings()


def test_splitter_purges_horizon_plus_one_rows_before_each_validation_fold():
    folds = list(purged_splitter(S).split(np.arange(120)))
    assert len(folds) == 5
    for train, val in folds:
        assert val.min() - train.max() - 1 == 11  # frozen horizon 10 -> purge 11


def test_overriding_horizon_changes_purge():
    s = dataclasses.replace(S, horizon=4)
    for train, val in purged_splitter(s).split(np.arange(120)):
        assert val.min() - train.max() - 1 == 5


def test_validation_folds_cover_five_sixths_of_data():
    val = np.concatenate([v for _, v in purged_splitter(S).split(np.arange(120))])
    assert len(val) == 100 and len(set(val)) == 100


def _frame(**cols):
    return pd.DataFrame(cols)


def test_two_feature_cluster_medoid_tie_goes_to_earlier_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = _frame(z=rng.normal(size=200), a=a, b=a * 2 + 0.01 * rng.normal(size=200))
    assert cluster_representatives(X, S) == ["z", "a"]
    # swapping the two columns' order flips the winner: it is order, not name
    assert cluster_representatives(X[["z", "b", "a"]], S) == ["z", "b"]


def test_single_feature_cluster_keeps_its_feature():
    rng = np.random.default_rng(1)
    X = _frame(**{c: rng.normal(size=200) for c in "pqr"})
    assert cluster_representatives(X, S) == ["p", "q", "r"]


def test_medoid_is_most_correlated_member_not_first():
    rng = np.random.default_rng(2)
    base = rng.normal(size=500)
    # b is the hub: close to both a and c; a and c are further from each other
    X = _frame(
        a=base + 0.6 * rng.normal(size=500),
        b=base + 0.05 * rng.normal(size=500),
        c=base + 0.6 * rng.normal(size=500),
        z=rng.normal(size=500),
    )
    assert cluster_representatives(X, S) == ["b", "z"]


def test_features_outside_clustering_pass_through_untouched():
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    X = _frame(lag1=a, a=a, b=a + 0.01 * rng.normal(size=200))
    # lag1 duplicates a, but stays; a and b cluster and a is the earlier column
    assert cluster_representatives(X, S, outside=["lag1"]) == ["lag1", "a"]


def _signal_data(n=300, n_noise=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_noise + 1)), columns=["sig"] + [f"n{i}" for i in range(n_noise)])
    y = pd.Series((X["sig"] + 0.3 * rng.normal(size=n) > 0).astype(int) * 2 - 1)  # labels in {-1, +1}
    return X, y


def _small():
    return XGBClassifier(n_estimators=10, max_depth=2, n_jobs=1)


def test_mda_ranks_informative_feature_first():
    X, y = _signal_data()
    imp = mda_importance(X, y, S, seed=42, model=_small())
    assert imp.index[0] == "sig" and imp.iloc[0] > 0


def test_mda_same_seed_same_ranking_different_seed_can_differ():
    X, y = _signal_data(n_noise=8)
    rank = lambda seed: list(mda_importance(X, y, S, seed=seed, model=_small()).index)
    assert rank(42) == rank(42)
    assert len({tuple(rank(s)) for s in S.selection_seeds}) > 1


def test_select_features_caps_at_n_features_and_keeps_signal():
    X, y = _signal_data(n_noise=20)
    out = select_features(X, y, S, seed=42, model=_small())
    assert len(out) == S.n_features and "sig" in out


def test_select_features_keeps_all_candidates_below_cap():
    X, y = _signal_data(n_noise=4)
    out = select_features(X, y, S, seed=42, model=_small())
    assert sorted(out) == sorted(X.columns)


def test_select_features_clusters_then_ranks_and_keeps_outside_columns():
    X, y = _signal_data(n_noise=3)
    X["sig_dup"] = X["sig"] * 3  # clusters with sig, sig is the earlier column
    X["lag1"] = X["sig"]  # outside clustering: survives despite duplicating sig
    out = select_features(X, y, S, seed=42, model=_small(), outside=["lag1"])
    assert "sig_dup" not in out and {"sig", "lag1"} <= set(out)


def test_mda_default_model_handles_a_training_fold_missing_a_class():
    rng = np.random.default_rng(5)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.choice([-1, 0], size=n))
    y.iloc[60:] = rng.choice([-1, 0, 1], size=n - 60)  # class +1 first appears after the early folds' train rows
    s = dataclasses.replace(S, primary_n_estimators=5)
    imp = mda_importance(X, y, s, seed=42)
    assert set(imp.index) == {"a", "b", "c"} and np.isfinite(imp).all()
