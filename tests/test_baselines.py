import numpy as np
import pandas as pd

from backtest.baselines import baselines
from backtest.settings import Settings

S = Settings()


def prices(n=250, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.015, n))
    return pd.DataFrame({"Close": c}, index=pd.bdate_range("2020-01-01", periods=n))


def test_both_baselines_cover_the_oos_dates_and_report_trade_counts():
    p = prices()
    oos = p.index[100:220]
    out = baselines(p, oos, S)
    assert set(out) == {"momentum", "mean_reversion"}
    for r in out.values():
        pd.testing.assert_index_equal(r.returns.index, oos)
        assert r.n_trades == len(r.trades) > 0
        assert r.trades["entry_date"].min() >= oos[0]


def test_signals_use_only_data_up_to_the_signal_day():
    p = prices()
    oos = p.index[100:220]
    changed = p.copy()
    changed.iloc[150:] *= 3  # rewrite the future
    a, b = baselines(p, oos, S), baselines(changed, oos, S)
    for k in a:
        cut = oos[oos < p.index[150]]
        pd.testing.assert_series_equal(a[k].position[cut], b[k].position[cut])


def test_momentum_goes_long_after_a_lookback_rise():
    n = 2 * S.momentum_lookback
    close = pd.Series(np.linspace(100, 200, n), index=pd.bdate_range("2020-01-01", periods=n))
    r = baselines(close.to_frame("Close"), close.index[S.momentum_lookback:], S)["momentum"]
    assert r.position.iloc[1] == 1
