"""Feature universe in level form (as v1 used it) and stationary form (scale-free)."""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta

N_LAGS = 30
CUM_WINDOW = 20


def level_features(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLCV, v1's technical indicators and 30 close lags, computed from rows up to each row.

    Warm-up rows are NaN; callers drop them (v1 did dropna).
    """
    p = prices[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    o, h, l, c, v = (p[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    f = p.copy()
    f["rsi"] = ta.rsi(c, length=14)
    macd = ta.macd(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = (
        macd["MACD_12_26_9"], macd["MACDs_12_26_9"], macd["MACDh_12_26_9"])
    stoch = ta.stoch(h, l, c)
    f["stoch_k"], f["stoch_d"] = stoch["STOCHk_14_3_3"], stoch["STOCHd_14_3_3"]
    f["cci"] = ta.cci(h, l, c, length=20)
    f["roc"] = ta.roc(c, length=12)
    f["atr"] = ta.atr(h, l, c)
    f["natr"] = ta.natr(h, l, c, length=14)
    bb = ta.bbands(c, length=20)
    f["bb_mid"], f["bb_hi"], f["bb_lo"], f["bb_width"] = (
        bb["BBM_20_2.0"], bb["BBU_20_2.0"], bb["BBL_20_2.0"], bb["BBB_20_2.0"])
    kc = ta.kc(h, l, c)
    f["kc_upper"], f["kc_Lower"] = kc["KCUe_20_2"], kc["KCLe_20_2"]
    f["obv"] = ta.obv(c, v)
    f["cmf"] = ta.cmf(h, l, c, v)
    f["mfi"] = ta.mfi(h, l, c, v)
    f["vol20"] = v.rolling(window=20).mean()
    f["ad"] = ta.ad(h, l, c, v)
    f["vwap"] = ta.vwap(h, l, c, v)
    f["ema_8"] = ta.ema(c, length=8)
    f["ema_21"] = ta.ema(c, length=21)
    f["ema_ratio"] = f["ema_8"] / f["ema_21"]
    f["sma_50"] = ta.sma(c, length=50)
    adx = ta.adx(h, l, c, length=14)
    f["adx"], f["dm_plus"], f["dm_minus"] = adx["ADX_14"], adx["DMP_14"], adx["DMN_14"]
    aroon = ta.aroon(h, l, length=25)
    f["aroon_up"], f["aroon_down"] = aroon["AROONU_25"], aroon["AROOND_25"]
    for k in range(1, N_LAGS + 1):
        f[f"lag_{k}"] = c.shift(k)
    return f


_PRICE_LEVEL = (["Open", "High", "Low", "bb_mid", "bb_hi", "bb_lo", "kc_upper", "kc_Lower",
                 "ema_8", "ema_21", "sma_50", "vwap"] + [f"lag_{k}" for k in range(1, N_LAGS + 1)])
_SPREADS = ["macd", "macd_signal", "macd_hist", "atr"]


def stationary_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Level form with every price and volume level replaced by a scale-free version."""
    f = level_features(prices)
    c, v = f["Close"], f["Volume"]
    s = f.drop(columns="Close")
    s[_PRICE_LEVEL] = np.log(f[_PRICE_LEVEL].div(c, axis=0))
    s[_SPREADS] = f[_SPREADS].div(c, axis=0)
    vol_sum = v.rolling(CUM_WINDOW).sum()
    for col in ("obv", "ad"):
        # Cumulative series start at 0 before the first row, so the first full window is defined.
        s[col] = (f[col] - f[col].shift(CUM_WINDOW, fill_value=0.0)) / vol_sum
    s["Volume"] = v / f["vol20"]
    s["vol20"] = s["Volume"]
    return s
