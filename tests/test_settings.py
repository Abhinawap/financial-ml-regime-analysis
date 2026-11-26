import dataclasses
import datetime

import pytest

from backtest.settings import Settings

EXPECTED = {
    "upper_barrier": 0.02,
    "lower_barrier": 0.02,
    "horizon": 10,
    "buy_cost_bps": 45.0,
    "sell_cost_bps": 55.0,
    "cost_stresses": ((70.0, 80.0), (95.0, 105.0)),
    "train_rows": 600,
    "test_rows": 120,
    "step": 120,
    "linkage": "ward",
    "distance_threshold": 0.6,
    "n_features": 15,
    "n_splits": 5,
    "n_permutation_repeats": 3,
    "selection_seeds": (42, 43, 44, 45, 46),
    "primary_objective": "multi:softprob",
    "primary_n_estimators": 200,
    "primary_max_depth": 6,
    "primary_learning_rate": 0.05,
    "model_seed": 42,
    "meta_n_estimators": 100,
    "meta_max_depth": 4,
    "meta_threshold": 0.5,
    "meta_seed": 42,
    "n_lags": 30,
    "cum_window": 20,
    "trading_days": 252,
    "ci_alpha": 0.05,
    "bootstrap_resamples": 10_000,
    "bootstrap_block_length": 10,
    "bootstrap_seed": 0,
    "momentum_lookback": 60,
    "rsi_length": 14,
    "rsi_entry_below": 30.0,
    "reading3_boundary": datetime.date(2024, 1, 1),
}


def test_defaults_are_the_frozen_values():
    assert dataclasses.asdict(Settings()) == EXPECTED


def test_purge_length_is_derived_from_horizon():
    assert Settings().purge_length == 11 == Settings().horizon + 1
    assert dataclasses.replace(Settings(), horizon=3).purge_length == 4
    assert "purge_length" not in {f.name for f in dataclasses.fields(Settings)}


def test_settings_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().horizon = 5
