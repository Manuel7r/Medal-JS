"""Statistical features: z-score, correlations, cointegration."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint


def z_score(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling z-score of a series.

    Args:
        series: Input series.
        lookback: Rolling window size.

    Returns:
        Z-score values.
    """
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    return (series - mean) / std


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return series_a.rolling(lookback).corr(series_b)


def cointegration_test(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict:
    """Engle-Granger cointegration test.

    Returns:
        Dict with keys: t_stat, p_value, critical_values, is_cointegrated.
    """
    clean_a = series_a.dropna()
    clean_b = series_b.dropna()
    common_idx = clean_a.index.intersection(clean_b.index)
    clean_a = clean_a.loc[common_idx]
    clean_b = clean_b.loc[common_idx]

    t_stat, p_value, critical_values = coint(clean_a, clean_b)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "critical_values": {
            "1%": float(critical_values[0]),
            "5%": float(critical_values[1]),
            "10%": float(critical_values[2]),
        },
        "is_cointegrated": bool(p_value < 0.05),
    }


def hedge_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
) -> float:
    """OLS hedge ratio: series_a = beta * series_b + alpha.

    Returns:
        beta (hedge ratio).
    """
    clean_a = series_a.dropna()
    clean_b = series_b.dropna()
    common_idx = clean_a.index.intersection(clean_b.index)

    slope, _intercept, _r, _p, _se = stats.linregress(
        clean_b.loc[common_idx], clean_a.loc[common_idx]
    )
    return float(slope)


def spread(
    series_a: pd.Series,
    series_b: pd.Series,
    beta: float | None = None,
) -> pd.Series:
    """Compute spread between two series.

    spread = series_a - beta * series_b

    If beta is None, it is computed via OLS.
    """
    if beta is None:
        beta = hedge_ratio(series_a, series_b)
    return series_a - beta * series_b


def spread_z_score(
    series_a: pd.Series,
    series_b: pd.Series,
    lookback: int = 168,
    beta: float | None = None,
) -> pd.Series:
    """Z-score of the spread between two cointegrated series.

    Args:
        series_a: First price series.
        series_b: Second price series.
        lookback: Rolling window for z-score.
        beta: Hedge ratio. Computed if None.

    Returns:
        Z-score of the spread.
    """
    s = spread(series_a, series_b, beta)
    return z_score(s, lookback)


def returns_stats(series: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Compute rolling return statistics.

    Returns:
        DataFrame with columns: returns_mean, returns_std, returns_skew, returns_kurt.
    """
    returns = series.pct_change()
    return pd.DataFrame({
        "returns_mean": returns.rolling(lookback).mean(),
        "returns_std": returns.rolling(lookback).std(),
        "returns_skew": returns.rolling(lookback).skew(),
        "returns_kurt": returns.rolling(lookback).kurt(),
    })


def compute_all(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute all statistical features on an OHLCV DataFrame.

    Adds columns for z-score, return stats, and lagged returns.
    """
    df = df.copy()
    close = df["close"]

    df["z_score_20"] = z_score(close, lookback)

    ret_stats = returns_stats(close, lookback)
    df["returns_skew"] = ret_stats["returns_skew"]
    df["returns_kurt"] = ret_stats["returns_kurt"]

    # Multi-period returns
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)

    # Lagged returns
    df["returns_lag_1"] = df["returns_1d"].shift(1)
    df["returns_lag_2"] = df["returns_1d"].shift(2)
    df["returns_lag_5"] = df["returns_1d"].shift(5)

    return df
