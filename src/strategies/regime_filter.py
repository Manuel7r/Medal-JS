"""Hurst-based regime filter for strategy selection.

Computes the Hurst exponent on price data and determines which
strategy type should be active:
    - Hurst < 0.45: mean-reverting regime → MeanReversion active
    - Hurst > 0.55: trending regime → Momentum active
    - 0.45 <= Hurst <= 0.55: ambiguous → both suppressed or reduced weight

Used as a wrapper around existing strategies to dynamically enable/disable
them based on market conditions.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_hurst(prices: pd.Series, max_lag: int = 100) -> float:
    """Compute Hurst exponent using R/S analysis.

    Args:
        prices: Price series (close prices).
        max_lag: Maximum lag for R/S computation.

    Returns:
        Hurst exponent (float). Returns 0.5 on failure.
    """
    if len(prices) < max_lag + 10:
        return 0.5

    try:
        log_prices = np.log(prices.dropna().values)
        returns = np.diff(log_prices)

        lags = range(2, min(max_lag, len(returns) // 2))
        rs_values = []
        lag_values = []

        for lag in lags:
            chunks = [returns[i:i + lag] for i in range(0, len(returns) - lag, lag)]
            if len(chunks) < 2:
                continue
            rs_list = []
            for chunk in chunks:
                mean = chunk.mean()
                deviations = np.cumsum(chunk - mean)
                r = deviations.max() - deviations.min()
                s = chunk.std(ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)

        if len(lag_values) < 3:
            return 0.5

        log_lags = np.log(lag_values)
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))
    except Exception:
        return 0.5


def classify_regime(hurst: float) -> str:
    """Classify market regime from Hurst exponent.

    Returns:
        'mean_reverting', 'trending', or 'ambiguous'
    """
    if hurst < 0.45:
        return "mean_reverting"
    elif hurst > 0.55:
        return "trending"
    return "ambiguous"


def get_strategy_weights(
    hurst: float,
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Adjust strategy weights based on Hurst regime.

    Args:
        hurst: Current Hurst exponent.
        base_weights: Original strategy weights.

    Returns:
        Adjusted weights dict.
    """
    if base_weights is None:
        base_weights = {
            "mean_reversion": 0.35,
            "momentum": 0.35,
            "ml_ensemble": 0.30,
        }

    regime = classify_regime(hurst)
    adjusted = dict(base_weights)

    if regime == "trending":
        # Boost momentum, suppress mean reversion
        adjusted["momentum"] = adjusted.get("momentum", 0.0) * 1.5
        adjusted["mean_reversion"] = adjusted.get("mean_reversion", 0.0) * 0.2
        logger.debug("Regime: trending (H={:.2f}), boosting momentum", hurst)
    elif regime == "mean_reverting":
        # Boost mean reversion, suppress momentum
        adjusted["mean_reversion"] = adjusted.get("mean_reversion", 0.0) * 1.5
        adjusted["momentum"] = adjusted.get("momentum", 0.0) * 0.2
        logger.debug("Regime: mean-reverting (H={:.2f}), boosting MR", hurst)
    else:
        # Ambiguous: reduce both directional strategies, keep ML
        adjusted["mean_reversion"] = adjusted.get("mean_reversion", 0.0) * 0.7
        adjusted["momentum"] = adjusted.get("momentum", 0.0) * 0.7
        logger.debug("Regime: ambiguous (H={:.2f}), reducing weights", hurst)

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted
