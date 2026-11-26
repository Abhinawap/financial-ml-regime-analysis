"""Full-settings rehearsal: the results command at frozen Settings on 1,465 rows of synthetic prices.

    OMP_NUM_THREADS=1 uv run python -m scripts.rehearsal [out]   # default out: results-rehearsal/

Catches crashes and measures runtime before the official run (#11: 922 s single-threaded). The outputs are
synthetic, so they go in an ignored directory, never in results/ where the official run writes.
"""
import sys
import time
from pathlib import Path

from backtest.results import run_results
from backtest.settings import Settings
from tests.test_walkforward import prices  # the tests' seeded random walk

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results-rehearsal")
start = time.perf_counter()
run_results(prices(1465), Settings(), out, official_run=False)
print(f"runtime {time.perf_counter() - start:.1f} s -> {out}")
