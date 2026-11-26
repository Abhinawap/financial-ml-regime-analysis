"""Results command: runs every arm and seed, the meta-labelling ablation and the baselines, and writes files.

    python -m backtest.results [--official-run] [--out results]

The notebook reads only these files, so it never fits a model. Without --official-run the guarded loader
returns development rows only.
"""
import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from backtest.baselines import baselines
from backtest.data import load_prices
from backtest.features import (
    CLUSTERED_LEVEL,
    CLUSTERED_STATIONARY,
    STATIONARY_TWIN,
    V1_STYLE,
)
from backtest.settings import Settings
from backtest.simulator import buy_and_hold
from backtest.walkforward import walk_forward

ARMS = {"v1_style": V1_STYLE, "stationary_twin": STATIONARY_TWIN,
        "clustered_level": CLUSTERED_LEVEL, "clustered_stationary": CLUSTERED_STATIONARY}
PRIMARY_PAIR = ("v1_style", "stationary_twin")
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pandas-ta-classic"]


def run_results(prices: pd.DataFrame, settings: Settings, out: Path, official_run: bool) -> None:
    """Write series.csv (daily return and position per run and seed), trades.csv, windows.csv and run.json."""
    series, trades, windows = [], [], []

    def add(run, seed, returns, position, trade_log=None):
        series.append(pd.DataFrame({"date": returns.index, "run": run, "seed": seed,
                                    "return": returns.to_numpy(), "position": position.to_numpy()}))
        if trade_log is not None and not trade_log.empty:  # an empty log has object dtypes that break concat
            trades.append(trade_log.assign(run=run, seed=seed)[["run", "seed", *trade_log.columns]])

    runs = [(name, arm, False) for name, arm in ARMS.items()]
    runs += [(f"{name}_meta", ARMS[name], True) for name in PRIMARY_PAIR]
    for run, arm, meta in runs:
        for seed in settings.selection_seeds:
            r = walk_forward(prices, arm, seed, settings, meta=meta)
            add(run, seed, r.returns, r.position, r.trades)
            w = r.windows.assign(run=run, seed=seed, features=r.windows.features.map(";".join))
            windows.append(w[["run", "seed", *r.windows.columns]])

    index = r.returns.index  # every run covers the same out-of-sample dates
    close = prices["Close"][index]
    bh = buy_and_hold(close, settings)
    add("buy_and_hold", pd.NA, bh.returns, bh.position, bh.trades)
    for name, b in baselines(prices, index, settings).items():
        add(name, pd.NA, b.returns, b.position, b.trades)
    # Gross buy-and-hold, no costs: the exposure-matched benchmark before scaling.
    add("benchmark", pd.NA, close.pct_change().fillna(0.0), pd.Series(1, index=index))

    out.mkdir(parents=True, exist_ok=True)
    for name, frames in [("series", series), ("trades", trades), ("windows", windows)]:
        pd.concat(frames, ignore_index=True).astype({"seed": "Int64"}).to_csv(out / f"{name}.csv", index=False)
    git = subprocess.run(["git", "describe", "--always", "--dirty"], check=True, capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    info = {
        "official_run": official_run,
        "rows": len(prices), "first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date()),
        "prices_sha256": hashlib.sha256(pd.util.hash_pandas_object(prices).to_numpy().tobytes()).hexdigest(),
        "git": git,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "settings": dataclasses.asdict(settings),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2, default=str) + "\n")


def main(argv=None, settings=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official-run", action="store_true", help="read every row, out-of-sample included")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    prices = load_prices(official_run=args.official_run)
    run_results(prices, Settings() if settings is None else settings, args.out, args.official_run)


if __name__ == "__main__":
    main()
