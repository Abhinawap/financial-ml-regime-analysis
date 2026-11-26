"""Frozen settings: every ticket reads values from here and defines no constants of its own."""
import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Labels and trading
    # Triple-barrier label: upper barrier is +upper_barrier, lower barrier is -lower_barrier.
    upper_barrier: float = 0.02
    lower_barrier: float = 0.02
    horizon: int = 10
    # Cost model, bps per side; cost_stresses are (buy, sell) = 150 and 200 bps round trip.
    buy_cost_bps: float = 45.0
    sell_cost_bps: float = 55.0
    cost_stresses: tuple[tuple[float, float], ...] = ((70.0, 80.0), (95.0, 105.0))

    # Walk-forward window
    train_rows: int = 600
    test_rows: int = 120
    step: int = 120

    # Feature selection
    linkage: str = "ward"
    distance_threshold: float = 0.6
    n_features: int = 15
    n_splits: int = 5
    n_permutation_repeats: int = 3
    selection_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)

    # Primary model (no class_weight: XGBoost ignores it, so it is deliberately absent)
    primary_objective: str = "multi:softprob"
    primary_n_estimators: int = 200
    primary_max_depth: int = 6
    primary_learning_rate: float = 0.05
    model_seed: int = 42

    # Meta-model
    meta_n_estimators: int = 100
    meta_max_depth: int = 4
    meta_threshold: float = 0.5
    meta_seed: int = 42

    # Features
    n_lags: int = 30  # close lags in the feature universe
    cum_window: int = 20  # rolling window for vol20 and the obv/ad differencing

    # Statistics
    trading_days: int = 252  # Sharpe annualisation
    ci_alpha: float = 0.05  # two-sided bootstrap CI level

    # Bootstrap (block length is the mean block length, in days)
    bootstrap_resamples: int = 10_000
    bootstrap_block_length: int = 10
    bootstrap_seed: int = 0

    # Baselines
    momentum_lookback: int = 60
    rsi_length: int = 14
    rsi_entry_below: float = 30.0

    # Reading 3
    reading3_boundary: datetime.date = datetime.date(2024, 1, 1)

    @property
    def purge_length(self) -> int:
        """Rows purged from a training window: labels start at the entry close, so horizon + 1."""
        return self.horizon + 1
