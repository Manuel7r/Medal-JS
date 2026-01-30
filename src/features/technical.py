"""Technical indicators: RSI, MACD, Bollinger Bands, ATR."""

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        series: Price series (typically close).
        period: Lookback period.

    Returns:
        RSI values (0-100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns:
        DataFrame with columns [macd, signal, histogram].
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns:
        DataFrame with columns [bb_upper, bb_middle, bb_lower, bb_position].
        bb_position: 0 = at lower band, 1 = at upper band.
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    position = (series - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_position": position,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.

    Returns:
        ATR values.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume.
    Adds indicator columns in-place and returns the DataFrame.
    """
    df = df.copy()

    # RSI
    df["rsi_14"] = rsi(df["close"], 14)

    # MACD
    macd_df = macd(df["close"])
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["signal"]
    df["macd_histogram"] = macd_df["histogram"]

    # Bollinger Bands
    bb_df = bollinger_bands(df["close"])
    df["bb_upper"] = bb_df["bb_upper"]
    df["bb_middle"] = bb_df["bb_middle"]
    df["bb_lower"] = bb_df["bb_lower"]
    df["bb_position"] = bb_df["bb_position"]

    # ATR
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    # Volume ratio
    df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df
