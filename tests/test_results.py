import dataclasses

import pandas as pd
import pytest

from backtest.results import main
from backtest.settings import Settings

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
