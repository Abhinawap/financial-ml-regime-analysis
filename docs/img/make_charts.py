"""README charts: excess_ci.svg and headline.svg.

    uv run python docs/img/make_charts.py [results_dir] [out_dir]   # default: results/ -> docs/img/

Reads the results command's files through backtest.analysis, the same functions the v2 notebook uses, so no
model is fitted. scripts/rehearsal.py runs it on rehearsal outputs, so a chart that can't be drawn fails there.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.dates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backtest import analysis as a
from backtest.stats import exposure_matched_benchmark

TEXT, CONTEXT, GRID, ACCENT = "#374151", "#9CA3AF", "#E5E7EB", "#0072B2"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "path",
    "svg.hashsalt": "make_charts",  # fixed glyph and clip IDs, so an unchanged chart rebuilds byte for byte
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def titles(fig, title, subtitle, source):
    fig.text(0.02, 0.97, title, fontsize=17, fontweight="semibold", va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=11, color="#4B5563", va="top")
    fig.text(0.02, 0.02, source, fontsize=9.5, color="#4B5563")


def save(fig, path):
    fig.savefig(path, metadata={"Date": None})  # no timestamp in the file
    plt.close(fig)


def excess_ci(series, trades, settings, source, out):
    runs = {"v1_style": "v1-style arm (headline)", "stationary_twin": "Stationary twin",
            "v1_style_meta": "v1-style + meta", "stationary_twin_meta": "Stationary twin + meta",
            "clustered_level": "Clustered level", "clustered_stationary": "Clustered stationary",
            "momentum": "Momentum baseline", "mean_reversion": "Mean-reversion baseline"}
    # Buy-and-hold is left out: its excess return only reflects scaling by its 815/816 average position.
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    fig.subplots_adjust(left=0.29, right=0.76, top=0.79, bottom=0.15)
    for i, (run, label) in enumerate(runs.items()):
        s = a.summary(series, trades, run, settings)
        lo, hi = (x * 1e4 for x in s["excess_ci"])
        primary = run in ("v1_style", "stationary_twin")
        color = ACCENT if primary else CONTEXT
        ax.plot([lo, hi], [i, i], color=color, lw=2.4 if primary else 1.6, solid_capstyle="butt")
        ax.plot(s["excess"] * 1e4, i, "o", color=color, ms=8 if primary else 6,
                mfc=color if primary else "white", mew=1.6)
        label_text = f"{s['excess'] * 1e4:+.1f}  [{lo:+.1f}, {hi:+.1f}]".replace("-", "\u2212")
        ax.text(1.03, i, label_text, transform=ax.get_yaxis_transform(), va="center", fontsize=9.5,
                color=TEXT if primary else "#4B5563")  # one aligned column, clear of every interval
    ax.set_yticks(range(len(runs)), list(runs.values()))
    for tick, run in zip(ax.get_yticklabels(), runs):
        tick.set_fontweight("semibold" if run in ("v1_style", "stationary_twin") else "normal")
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT, lw=1, ls=(0, (4, 3)))
    ax.axhline(1.5, color=GRID, lw=1)
    ax.text(1.03, -0.75, "mean  [95% CI]", transform=ax.get_yaxis_transform(), fontsize=9.5, color="#4B5563")
    ax.set_xlabel("Mean daily excess return, bps (dashed line: no edge)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    start, end = series.date.min().date(), series.date.max().date()
    titles(fig, "No run's excess return is clearly above zero",
           f"Mean daily return minus the exposure-matched benchmark's, default costs, {start} to {end}.\n"
           "Primary pair in blue (filled); every interval includes zero or lies below it.", source)
    save(fig, out / "excess_ci.svg")


def headline(series, trades, settings, source, out):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    fig.subplots_adjust(left=0.08, right=0.76, top=0.80, bottom=0.12)
    ends, low = [], 1.0
    for run, label, color, lw in [("v1_style", "v1-style arm", ACCENT, 2.2), ("stationary_twin", "Stationary twin", TEXT, 1.6)]:
        r = a.per_seed(series, trades, run, settings).mean(axis=1)
        matched = (1 + exposure_matched_benchmark(a.benchmark(series), a.position(series, run))).cumprod()
        equity = (1 + r).cumprod()
        ax.plot(equity, color=color, lw=lw)
        ax.plot(matched, color=color, lw=1.1, ls=(0, (4, 3)), alpha=0.55)
        ends += [(equity.iloc[-1], label, color), (matched.iloc[-1], f"{label}'s benchmark", color)]
        low = min(low, equity.min(), matched.min())
    bottom = math.floor(low * 10) / 10  # 0.8 on the official outputs
    ax.set_ylim(bottom, None)
    x, placed = series.date.max() + pd.DateOffset(days=14), []
    for y, label, color in sorted(ends):  # direct labels at line ends, pushed apart where they would collide
        placed.append(max(y, placed[-1] + 0.09) if placed else y)
        ax.text(x, placed[-1], f"{label}  {y - 1:+.1%}".replace("-", "\u2212"), va="center", fontsize=9.5, color=color,
                clip_on=False)
    boundary = settings.reading3_boundary
    ax.axvline(boundary, color=CONTEXT, lw=1, ls=(0, (2, 2)))
    ax.text(boundary, ax.get_ylim()[1], f" {boundary}: early | late", fontsize=9, color="#4B5563", va="top")
    ax.axhline(1, color=GRID, lw=1)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_ylabel(f"Growth of 1, net of costs (axis starts at {bottom:g})")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    start, end = series.date.min().date(), series.date.max().date()
    titles(fig, "Neither arm keeps pace with its exposure-matched benchmark",
           f"Seed-averaged out-of-sample equity, default costs, {start} to {end}.\n"
           "Dashed: buy-and-hold scaled by each arm's average position, no costs.", source)
    save(fig, out / "headline.svg")


def main(results: Path, out: Path) -> None:
    series, trades, _, settings = a.load(results)
    run = json.loads((results / "run.json").read_text())
    source = (f"Source: {results.name}/ ({'tag ' if run['official_run'] else 'rehearsal at '}{run['git']}), "
              "via backtest/analysis.py.")
    out.mkdir(parents=True, exist_ok=True)
    excess_ci(series, trades, settings, source, out)
    headline(series, trades, settings, source, out)


if __name__ == "__main__":
    args = [Path(p) for p in sys.argv[1:]]
    main(args[0] if args else ROOT / "results", args[1] if len(args) > 1 else ROOT / "docs/img")
