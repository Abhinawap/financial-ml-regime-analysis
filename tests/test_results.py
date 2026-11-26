import dataclasses

import pandas as pd
import pytest

from backtest.results import main, run_results
from backtest.settings import Settings
from tests.test_walkforward import prices

# Two windows on the development rows, one seed: the whole command in a few seconds.
TINY = dataclasses.replace(Settings(), train_rows=400, test_rows=100, step=100, primary_n_estimators=5,
                           primary_max_depth=2, n_splits=2, n_permutation_repeats=1, meta_n_estimators=5,
                           meta_max_depth=2, selection_seeds=(42,))


def test_without_the_official_flag_it_writes_every_run_from_development_rows_only(tmp_path):
    main(["--out", str(tmp_path)], settings=TINY)
    series = pd.read_csv(tmp_path / "series.csv", parse_dates=["date"])
    assert set(series.run) == {"v1_style", "stationary_twin", "clustered_level", "clustered_stationary",
                               "v1_style_meta", "stationary_twin_meta",
                               "buy_and_hold", "momentum", "mean_reversion", "benchmark"}
    assert series.date.max() == pd.Timestamp("2022-06-03")  # row 648, the last development row
    windows = pd.read_csv(tmp_path / "windows.csv")
    assert set(windows.run) == {"v1_style", "stationary_twin", "clustered_level", "clustered_stationary",
                                "v1_style_meta", "stationary_twin_meta"}
    assert windows.fingerprint.str.len().eq(64).all() and windows.features.str.len().gt(0).all()
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert {"run", "seed", "entry_date", "exit_date", "net_return"} <= set(trades.columns)
    assert (tmp_path / "run.json").exists()


def test_rerunning_with_the_same_inputs_writes_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(["--out", str(a)], settings=TINY)
    main(["--out", str(b)], settings=TINY)
    names = sorted(p.name for p in a.iterdir())
    assert names == ["run.json", "series.csv", "trades.csv", "windows.csv"]
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_frozen_settings_on_development_rows_fail_clearly_before_writing(tmp_path):
    # The default path: 649 development rows cannot hold one 600-row window after warm-up.
    with pytest.raises(ValueError, match="too few rows"):
        main(["--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_gross_returns_differ_from_net_by_one_round_trip_cost_per_trade(tmp_path):
    # Cost stresses and break-even are recomputed from gross returns and trade dates (spec), never by refitting.
    main(["--out", str(tmp_path)], settings=TINY)
    series = pd.read_csv(tmp_path / "series.csv")
    n_trades = pd.read_csv(tmp_path / "trades.csv").groupby("run").size()
    cost = (series.gross_return - series["return"]).groupby(series.run).sum()
    expected = n_trades.reindex(cost.index, fill_value=0) * (45 + 55) / 1e4
    pd.testing.assert_series_equal(cost, expected.astype(float), check_names=False)
    assert n_trades["v1_style"] > 0 and cost["benchmark"] == 0


def test_a_non_finite_feature_after_warm_up_stops_the_run_before_any_fit_or_file(tmp_path):
    # A trading suspension: 25 zero-volume days make volume-scaled features 0/0.
    p = prices(600)
    p.iloc[300:325, p.columns.get_loc("Volume")] = 0.0
    with pytest.raises(ValueError, match="non-finite"):
        run_results(p, TINY, tmp_path / "out", official_run=False)
    assert not (tmp_path / "out").exists()


def test_official_run_refuses_a_dirty_tree_before_reading_any_price(tmp_path, monkeypatch):
    # run.json's commit is the record of the official run; uncommitted changes would make it meaningless.
    from backtest import results
    monkeypatch.setattr(results, "git_describe", lambda: "v1-36-gdb41825-dirty")
    monkeypatch.setattr(results, "load_prices", lambda **_: pytest.fail("read prices on a dirty tree"))
    with pytest.raises(ValueError, match="uncommitted"):
        main(["--official-run", "--out", str(tmp_path / "out")])
