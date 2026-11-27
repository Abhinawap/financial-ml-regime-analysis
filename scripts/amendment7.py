"""Amendment 7's figures (Amendment 8: computed here, not in the notebook), from a results directory.

    uv run python -m scripts.amendment7 RESULTS_DIR

Prints, for each primary-pair arm, max drawdown and longest time under water of the seed-averaged series net of
default costs and of its headline exposure-matched benchmark (no costs), then the 95% interval on mean daily
excess return at mean block lengths of 5, 10 and 20 days. Stops if the 10-day row differs from summary()'s
interval, which is what the notebook published.
"""
import dataclasses
import sys
from pathlib import Path

from backtest import analysis as a
from backtest.results import PRIMARY_PAIR
from backtest.stats import exposure_matched_benchmark


def main(results: Path) -> None:
    series, trades, _, settings = a.load(results)
    for run in PRIMARY_PAIR:
        r, pos = a.per_seed(series, trades, run, settings).mean(axis=1), a.position(series, run)
        model, bench = a.drawdown(r), a.drawdown(exposure_matched_benchmark(a.benchmark(series), pos))
        print(f"{run}: drawdown {model}, benchmark {bench}")
        published = a.summary(series, trades, run, settings)["excess_ci"]
        rows = {k: a.summary(series, trades, run, dataclasses.replace(settings, bootstrap_block_length=k))["excess_ci"]
                for k in (5, 10, 20)}
        if rows[10] != published:
            raise SystemExit(f"{run}: 10-day interval {rows[10]} differs from the published {published}")
        for k, (lo, hi) in rows.items():
            print(f"{run}: block {k:>2} days, excess CI [{lo * 1e4:+.2f}, {hi * 1e4:+.2f}] bps/day")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
