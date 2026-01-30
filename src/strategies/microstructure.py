"""Microstructure strategy: order flow and volume analysis.

Analyzes intra-bar volume patterns and price microstructure for
short-term alpha:
    - Volume Profile: abnormal volume detection
    - VWAP deviation
    - Volume-weighted momentum
    - Buying/selling pressure estimation
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class MicrostructureParams(StrategyParams):
    """Parameters for Microstructure strategy."""

    name: str = "Microstructure"
    vwap_lookback: int = 24       # VWAP rolling window
    volume_lookback: int = 48     # Volume spike detection window
    volume_spike: float = 2.5     # Spike threshold (x mean volume)
    vwap_entry_std: float = 1.5   # Entry when price deviates from VWAP
    vwap_exit_std: float = 0.3    # Exit when near VWAP
    max_hold: int = 24            # Max hold (1 day)
    use_volume_filter: bool = True


class MicrostructureStrategy(BaseStrategy):
    """Volume and price microstructure strategy.

    Signal logic:
        - BUY: Price below VWAP - entry_std*σ AND volume spike (buying pressure)
        - SELL: Price above VWAP + entry_std*σ AND volume spike (selling pressure)
        - EXIT: Price crosses back to VWAP OR max_hold
        - HOLD: otherwise

    Volume spike detection: current volume > volume_spike * mean volume
    Buying pressure: close > open (bullish candle) with high volume
    Selling pressure: close < open (bearish candle) with high volume
    """

    def __init__(self, params: MicrostructureParams | None = None) -> None:
        self.params = params or MicrostructureParams()
        self._position_bar: int | None = None
        self._position_side: Signal | None = None

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute VWAP, volume profile, and pressure indicators."""
        data = data.copy()
        p = self.params
        close = data["close"]
        volume = data["volume"]

        # VWAP (rolling)
        typical_price = (data["high"] + data["low"] + close) / 3
        cum_tp_vol = (typical_price * volume).rolling(p.vwap_lookback).sum()
        cum_vol = volume.rolling(p.vwap_lookback).sum()
        data["micro_vwap"] = cum_tp_vol / cum_vol

        # VWAP deviation
        data["micro_vwap_std"] = (close - data["micro_vwap"]).rolling(p.vwap_lookback).std()
        data["micro_vwap_zscore"] = (close - data["micro_vwap"]) / data["micro_vwap_std"]

        # Volume analysis
        data["micro_vol_mean"] = volume.rolling(p.volume_lookback).mean()
        data["micro_vol_ratio"] = volume / data["micro_vol_mean"]

        # Buying/selling pressure (simplified)
        data["micro_pressure"] = np.where(close >= data["open"], 1.0, -1.0)
        data["micro_weighted_pressure"] = data["micro_pressure"] * data["micro_vol_ratio"]
        data["micro_pressure_ma"] = data["micro_weighted_pressure"].rolling(6).mean()

        return data

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate microstructure signal."""
        p = self.params

        if index < max(p.vwap_lookback, p.volume_lookback) + 5:
            return Signal.HOLD

        vwap_z = data["micro_vwap_zscore"].iloc[index]
        vol_ratio = data["micro_vol_ratio"].iloc[index]
        pressure = data["micro_pressure_ma"].iloc[index]

        if np.isnan(vwap_z) or np.isnan(vol_ratio):
            return Signal.HOLD

        # Max hold check
        if self._position_bar is not None:
            bars_held = index - self._position_bar
            if bars_held >= p.max_hold:
                self._position_bar = None
                self._position_side = None
                return Signal.EXIT

        # Exit on mean reversion to VWAP
        if self._position_bar is not None and abs(vwap_z) < p.vwap_exit_std:
            self._position_bar = None
            self._position_side = None
            return Signal.EXIT

        # Entry
        if self._position_bar is None:
            has_volume = not p.use_volume_filter or vol_ratio > p.volume_spike

            # Oversold with buying pressure
            if vwap_z < -p.vwap_entry_std and has_volume and pressure > 0:
                self._position_bar = index
                self._position_side = Signal.BUY
                return Signal.BUY

            # Overbought with selling pressure
            if vwap_z > p.vwap_entry_std and has_volume and pressure < 0:
                self._position_bar = index
                self._position_side = Signal.SELL
                return Signal.SELL

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
        self._position_side = None


def backtest_microstructure(
    data: pd.DataFrame,
    params: MicrostructureParams | None = None,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run microstructure strategy backtest."""
    params = params or MicrostructureParams()
    strategy = MicrostructureStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        position_size_pct=position_size_pct,
    )

    result = engine.run(
        data=prepared,
        signal_fn=strategy.generate_engine_signal,
        symbol="MICRO",
        stop_loss_pct=0.03,
        take_profit_pct=0.04,
    )

    logger.info(
        "Microstructure backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )
    return result
