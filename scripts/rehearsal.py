"""Full-settings rehearsal: the results command at frozen Settings on 1,465 rows of synthetic prices, then the analysis.

    OMP_NUM_THREADS=1 uv run python -m scripts.rehearsal [out]   # default out: results-rehearsal/

Catches crashes and measures runtime before the official run (#11: 922 s single-threaded). It then computes every
reading and table the v2 notebook shows and draws the README charts, so anything that reads results and can't be
computed fails here, before the tag. The outputs are synthetic, so they go in an ignored directory, never in
results/ where the official run writes.
"""
import sys
import time
from pathlib import Path

from backtest import analysis as a
from backtest.results import ARMS, PRIMARY_PAIR, run_results
from backtest.settings import Settings
from docs.img.make_charts import main as make_charts
from tests.test_walkforward import prices  # the tests' seeded random walk


def analyse(out: Path) -> None:
    """The notebook's readings, tables and B10 cross-check, and the README charts (into out/charts), on out."""
    series, trades, windows, settings = a.load(out)
    seeds = {run: a.per_seed(series, trades, run, settings) for run in PRIMARY_PAIR}
    print("reading 1:", a.reading1(*seeds.values(), settings)["outcome"])
    rows = {run: a.summary(series, trades, run, settings) for run in sorted(set(series.run) - {"benchmark"})}
    print("reading 2 holds:", not any(rows[run]["excess_ci"][0] > 0 for run in PRIMARY_PAIR))
    for run, row in rows.items():  # B10: excess return at each reported break-even cost is zero
        if row["break_even"] is not None and row["break_even"] != float("inf"):
            assert abs(a.summary(series, trades, run, settings, row["break_even"])["excess"]) < 1e-12, run
    test_windows = windows[windows.run == PRIMARY_PAIR[0]].drop_duplicates("test_start")
    for run in ARMS:
        r, pos = a.per_seed(series, trades, run, settings).mean(axis=1), a.position(series, run)
        a.per_window(r, pos, a.benchmark(series), test_windows, settings)
        if run in PRIMARY_PAIR:
            print(f"reading 3 pattern, {run}:", a.reading3(r, pos, a.benchmark(series), test_windows, settings)["holds"])
    for buy, sell in settings.cost_stresses:
        for run in PRIMARY_PAIR:
            a.summary(series, trades, run, settings, buy + sell)
    a.price_only(series, settings)
    make_charts(out, out / "charts")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results-rehearsal")
    start = time.perf_counter()
    run_results(prices(1465), Settings(), out, official_run=False)
    print(f"results command {time.perf_counter() - start:.1f} s -> {out}")
    analyse(out)
    print(f"analysis and charts done -> {out / 'charts'}")
