import hashlib
import inspect

import pandas as pd
import pytest

from backtest import data
from backtest.data import DATA_PATH, DATA_SHA256, load_prices

FROZEN_SHA256 = "4a90304fa95b3e8adaf736e4876dfd03713585f7599e5d34bcd04a570d27f71d"


def test_data_file_matches_frozen_hash():
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == FROZEN_SHA256
    assert DATA_SHA256 == FROZEN_SHA256


def test_default_returns_only_development_rows():
    assert inspect.signature(load_prices).parameters["official_run"].default is False
    prices = load_prices()
    assert len(prices) == 649
    assert prices.index[0] == pd.Timestamp("2019-10-01")
    assert prices.index[-1] == pd.Timestamp("2022-06-03")
    assert prices.index[-1] < pd.Timestamp("2022-06-06")


def test_official_run_returns_all_rows():
    # The ONLY test allowed to pass official_run=True.
    assert len(load_prices(official_run=True)) == 1465


def test_hash_mismatch_raises(monkeypatch):
    monkeypatch.setattr(data, "DATA_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        load_prices()
