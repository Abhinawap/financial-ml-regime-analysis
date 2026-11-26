"""Guarded loader: the only way code reads KEEN prices."""
import hashlib
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "KEEN_2019_2025.csv"
DATA_SHA256 = "4a90304fa95b3e8adaf736e4876dfd03713585f7599e5d34bcd04a570d27f71d"

# Row 649 is the first out-of-sample row (2022-06-06); rows 0-648 are development rows.
DEV_ROWS = 649


def load_prices(*, official_run: bool = False) -> pd.DataFrame:
    """Return KEEN OHLCV indexed by Date.

    By default only the development rows (before the out-of-sample series).
    Pass official_run=True, as a keyword, to get every row.
    """
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError(f"{DATA_PATH} does not match the frozen sha256 {DATA_SHA256}")
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return prices if official_run else prices.iloc[:DEV_ROWS]
