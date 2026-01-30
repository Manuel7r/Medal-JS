"""Triangular Arbitrage strategy for 3-asset cointegration.

Extends pairs trading to 3 assets. When the synthetic spread of
three cointegrated assets deviates, enter a market-neutral position
expecting reversion.

Example: BTC/USDT, ETH/USDT, ETH/BTC
    - If ETH/BTC implied by ETH/USDT / BTC/USDT deviates from actual
      ETH/BTC, there's an arbitrage opportunity.

This is a simplified statistical version (not pure exchange-level arb).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class TriangularArbParams(StrategyParams):
    """Parameters for Triangular Arbitrage strategy."""

    name: str = "TriangularArb"
    lookback: int = 72             # window for z-score
    entry_zscore: float = 2.0      # entry threshold
    exit_zscore: float = 0.3       # exit threshold
    stop_zscore: float = 4.0       # stop loss
    max_hold: int = 48             # max bars (2 days)


class TriangularArbStrategy(BaseStrategy):
    """Statistical triangular arbitrage on 3 assets.

    Expects data with columns: close_a, close_b, close_c
    where the theoretical relationship is: close_c ≈ close_b / close_a
    (e.g., ETH/BTC ≈ ETH/USDT / BTC/USDT)

    The spread = close_c - (close_b / close_a) and we trade
    the z-score of this spread.
    """

    def __init__(self, params: TriangularArbParams | None = None) -> None:
        self.params = params or TriangularArbParams()
        self._position_bar: int | None = None

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute triangular spread and z-score.

        Expects: close_a (base, e.g. BTC/USDT), close_b (quote, e.g. ETH/USDT),
                 close_c (cross, e.g. ETH/BTC)
        """
        data = data.copy()
        p = self.params

        # Implied cross rate
        data["tri_implied"] = data["close_b"] / data["close_a"]

        # Spread: actual - implied
        data["tri_spread"] = data["close_c"] - data["tri_implied"]

        # Z-score
        data["tri_spread_mean"] = data["tri_spread"].rolling(p.lookback).mean()
        data["tri_spread_std"] = data["tri_spread"].rolling(p.lookback).std()
        data["tri_zscore"] = (data["tri_spread"] - data["tri_spread_mean"]) / data["tri_spread_std"]

        # Spread in percentage terms
        data["tri_spread_pct"] = data["tri_spread"] / data["tri_implied"] * 100

        return data

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate triangular arb signal based on spread z-score."""
        p = self.params

        if index < p.lookback + 5:
            return Signal.HOLD

        z = data["tri_zscore"].iloc[index]
        if np.isnan(z):
            return Signal.HOLD

        # Max hold
        if self._position_bar is not None:
            if index - self._position_bar >= p.max_hold:
                self._position_bar = None
                return Signal.EXIT

        # Stop loss
        if self._position_bar is not None and abs(z) > p.stop_zscore:
            self._position_bar = None
            return Signal.EXIT

        # Exit on mean reversion
        if self._position_bar is not None and abs(z) < p.exit_zscore:
            self._position_bar = None
            return Signal.EXIT

        # Entry
        if self._position_bar is None:
            if z > p.entry_zscore:
                self._position_bar = index
                return Signal.SELL  # Cross rate too high -> sell cross, buy implied
            if z < -p.entry_zscore:
                self._position_bar = index
                return Signal.BUY   # Cross rate too low -> buy cross, sell implied

        return Signal.HOLD

    def generate_engine_signal(self, data: pd.DataFrame, index: int) -> SignalType:
        """Adapter for BacktestEngine."""
        signal = self.generate_signal(data, index)
        mapping = {
            Signal.BUY: SignalType.BUY,
            Signal.SELL: SignalType.SELL,
            Signal.EXIT: SignalType.EXIT,
            Signal.HOLD: SignalType.HOLD,
            Signal.LONG_SPREAD: SignalType.BUY,
            Signal.SHORT_SPREAD: SignalType.SELL,
        }
        return mapping[signal]

    def reset(self) -> None:
        """Reset internal state."""
        self._position_bar = None


def build_triangular_dataframe(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
) -> pd.DataFrame:
    """Merge three OHLCV DataFrames into a triangular arb DataFrame.

    df_a: base (e.g. BTC/USDT)
    df_b: quote (e.g. ETH/USDT)
    df_c: cross (e.g. ETH/BTC)
    """
    a = df_a[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    a.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    a["close_a"] = a["close"]

    b = df_b[["timestamp", "close"]].copy().rename(columns={"close": "close_b"})
    c = df_c[["timestamp", "close"]].copy().rename(columns={"close": "close_c"})

    merged = a.merge(b, on="timestamp").merge(c, on="timestamp")
    return merged.sort_values("timestamp").reset_index(drop=True)
