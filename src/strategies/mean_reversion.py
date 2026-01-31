"""Mean Reversion strategy for single assets.

Buys oversold conditions and sells overbought conditions based on
z-score deviation from a rolling mean. Includes a volatility regime
filter to avoid trading in strong trends.

Crypto parameters (1h timeframe):
    - lookback: 72 periods (3 days)
    - entry_std: 3.0
    - exit_std: 0.2
    - max_hold: 72 bars (3 days)
    - vol_lookback: 100
    - vol_threshold: 2.0 (skip if current vol > 2.0x average vol)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class MeanReversionParams(StrategyParams):
    """Parameters for Mean Reversion strategy."""

    name: str = "MeanReversion"
    lookback: int = 72            # rolling window for mean/std (3 days at 1h)
    entry_std: float = 3.0        # entry z-score threshold (higher = fewer but better trades)
    exit_std: float = 0.2         # exit z-score threshold (closer to mean = fuller reversion)
    max_hold: int = 72            # max bars to hold (3 days)
    vol_lookback: int = 100       # volatility regime lookback
    vol_threshold: float = 2.0    # skip if vol ratio > this (trending market)
    use_vol_filter: bool = True   # enable/disable volatility filter


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion on a single asset.

    Signal logic:
        - BUY:  z-score < -entry_std (oversold)
        - SELL: z-score >  entry_std (overbought)
        - EXIT: |z-score| < exit_std or max_hold reached
        - HOLD: otherwise

    Volatility filter:
        If enabled, entry signals are suppressed when the current rolling
        volatility exceeds vol_threshold × long-term average volatility.
        This avoids trading during strong trends where mean reversion fails.
    """

    def __init__(self, params: MeanReversionParams | None = None) -> None:
        self.params = params or MeanReversionParams()
        self._position_bar: int | None = None
        self._position_side: Signal | None = None

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Precompute z-score and volatility features.

        Expects column: close.
        Adds: mr_mean, mr_std, mr_zscore, mr_vol, mr_vol_avg, mr_vol_ratio.
        """
        data = data.copy()
        p = self.params

        close = data["close"]

        # Rolling mean and std
        data["mr_mean"] = close.rolling(p.lookback).mean()
        data["mr_std"] = close.rolling(p.lookback).std()
        data["mr_zscore"] = (close - data["mr_mean"]) / data["mr_std"]

        # Volatility regime filter
        returns = close.pct_change()
        data["mr_vol"] = returns.rolling(p.lookback).std()
        data["mr_vol_avg"] = returns.rolling(p.vol_lookback).std()
        data["mr_vol_ratio"] = data["mr_vol"] / data["mr_vol_avg"]

        return data

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate mean reversion signal."""
        p = self.params

        if index < max(p.lookback, p.vol_lookback):
            return Signal.HOLD

        z = data["mr_zscore"].iloc[index]
        if np.isnan(z):
            return Signal.HOLD

        # Check max hold
        if self._position_bar is not None:
            bars_held = index - self._position_bar
            if bars_held >= p.max_hold:
                self._position_bar = None
                self._position_side = None
                return Signal.EXIT

        # Exit on mean reversion
        if self._position_bar is not None and abs(z) < p.exit_std:
            self._position_bar = None
            self._position_side = None
            return Signal.EXIT

        # Volatility filter: block entries in trending markets
        if self._position_bar is None and p.use_vol_filter:
            vol_ratio = data["mr_vol_ratio"].iloc[index]
            if not np.isnan(vol_ratio) and vol_ratio > p.vol_threshold:
                return Signal.HOLD

        # Entry signals
        if self._position_bar is None:
            if z < -p.entry_std:
                self._position_bar = index
                self._position_side = Signal.BUY
                return Signal.BUY
            if z > p.entry_std:
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
        """Reset internal state for a new backtest run."""
        self._position_bar = None
        self._position_side = None


def backtest_mean_reversion(
    data: pd.DataFrame,
    params: MeanReversionParams | None = None,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run a full mean reversion backtest.

    Args:
        data: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
        params: Strategy parameters.
        initial_capital: Starting equity.
        commission_rate: Commission per side.
        slippage_rate: Slippage per trade.
        position_size_pct: Fraction of equity per trade.

    Returns:
        BacktestResult with metrics, equity curve, trades.
    """
    params = params or MeanReversionParams()
    strategy = MeanReversionStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        position_size_pct=position_size_pct,
    )

    # Compute median ATR-based stop percentages
    if "atr_14" in prepared.columns:
        median_atr = prepared["atr_14"].dropna().median()
        median_close = prepared["close"].dropna().median()
        atr_pct = median_atr / median_close if median_close > 0 else 0.02
        stop_loss = 2.5 * atr_pct
        take_profit = 4.0 * atr_pct
    else:
        stop_loss = 0.05
        take_profit = 0.08

    result = engine.run(
        data=prepared,
        signal_fn=strategy.generate_engine_signal,
        symbol="MR",
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    logger.info(
        "MeanReversion backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )

    return result
