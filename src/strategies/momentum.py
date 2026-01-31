"""Momentum / Trend Following strategy.

Complementary to Mean Reversion. Captures directional trends using
a combination of:
    - Dual moving average crossover (fast/slow EMA)
    - ADX trend strength filter
    - RSI momentum confirmation
    - ATR-based dynamic sizing

Crypto parameters (1h):
    - fast_period: 12 EMA
    - slow_period: 48 EMA
    - adx_period: 14, min_adx: 25
    - rsi_period: 14, rsi_oversold: 35, rsi_overbought: 65
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class MomentumParams(StrategyParams):
    """Parameters for Momentum strategy."""

    name: str = "Momentum"
    fast_period: int = 12
    slow_period: int = 48
    adx_period: int = 14
    min_adx: float = 25.0       # minimum ADX for trend confirmation
    rsi_period: int = 14
    rsi_oversold: float = 35.0  # RSI below this in uptrend = buy
    rsi_overbought: float = 65.0  # RSI above this in downtrend = sell
    max_hold: int = 120         # max bars (5 days)
    use_adx_filter: bool = True
    use_rsi_filter: bool = True


class MomentumStrategy(BaseStrategy):
    """Trend-following momentum strategy.

    Signal logic:
        - BUY: fast EMA > slow EMA AND (ADX > min_adx or disabled) AND (RSI < overbought or disabled)
        - SELL: fast EMA < slow EMA AND (ADX > min_adx or disabled) AND (RSI > oversold or disabled)
        - EXIT: crossover in opposite direction OR max_hold reached
        - HOLD: otherwise
    """

    def __init__(self, params: MomentumParams | None = None) -> None:
        self.params = params or MomentumParams()
        self._position_bar: int | None = None
        self._position_side: Signal | None = None

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Precompute EMA crossover, ADX, and RSI indicators."""
        data = data.copy()
        p = self.params
        close = data["close"]

        # EMAs
        data["mom_fast_ema"] = close.ewm(span=p.fast_period, adjust=False).mean()
        data["mom_slow_ema"] = close.ewm(span=p.slow_period, adjust=False).mean()
        data["mom_ema_diff"] = data["mom_fast_ema"] - data["mom_slow_ema"]

        # ADX (simplified: using directional movement)
        high = data["high"]
        low = data["low"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        plus_dm = np.where(
            (high - prev_high) > (prev_low - low),
            np.maximum(high - prev_high, 0),
            0,
        )
        minus_dm = np.where(
            (prev_low - low) > (high - prev_high),
            np.maximum(prev_low - low, 0),
            0,
        )

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / p.adx_period, min_periods=p.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / p.adx_period, min_periods=p.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / p.adx_period, min_periods=p.adx_period).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        data["mom_adx"] = dx.ewm(alpha=1 / p.adx_period, min_periods=p.adx_period).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / p.rsi_period, min_periods=p.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1 / p.rsi_period, min_periods=p.rsi_period).mean()
        rs = avg_gain / avg_loss
        data["mom_rsi"] = 100 - (100 / (1 + rs))

        return data

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate momentum signal."""
        p = self.params

        if index < p.slow_period + 5:
            return Signal.HOLD

        ema_diff = data["mom_ema_diff"].iloc[index]
        prev_ema_diff = data["mom_ema_diff"].iloc[index - 1]

        if np.isnan(ema_diff) or np.isnan(prev_ema_diff):
            return Signal.HOLD

        # Check max hold
        if self._position_bar is not None:
            bars_held = index - self._position_bar
            if bars_held >= p.max_hold:
                self._position_bar = None
                self._position_side = None
                return Signal.EXIT

        # Exit on crossover reversal
        if self._position_bar is not None:
            if self._position_side == Signal.BUY and ema_diff < 0:
                self._position_bar = None
                self._position_side = None
                return Signal.EXIT
            if self._position_side == Signal.SELL and ema_diff > 0:
                self._position_bar = None
                self._position_side = None
                return Signal.EXIT

        # Entry: require crossover (sign change)
        if self._position_bar is None:
            crossover_up = prev_ema_diff <= 0 and ema_diff > 0
            crossover_down = prev_ema_diff >= 0 and ema_diff < 0

            # ADX filter
            if p.use_adx_filter:
                adx = data["mom_adx"].iloc[index]
                if np.isnan(adx) or adx < p.min_adx:
                    return Signal.HOLD

            # RSI filter
            rsi = data["mom_rsi"].iloc[index]

            if crossover_up:
                if p.use_rsi_filter and not np.isnan(rsi) and rsi > p.rsi_overbought:
                    return Signal.HOLD  # Overbought, skip buy
                self._position_bar = index
                self._position_side = Signal.BUY
                return Signal.BUY

            if crossover_down:
                if p.use_rsi_filter and not np.isnan(rsi) and rsi < p.rsi_oversold:
                    return Signal.HOLD  # Oversold, skip sell
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


def backtest_momentum(
    data: pd.DataFrame,
    params: MomentumParams | None = None,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run momentum strategy backtest."""
    params = params or MomentumParams()
    strategy = MomentumStrategy(params)
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
        symbol="MOM",
        atr_stop_multiplier=2.5,
        atr_tp_multiplier=4.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
    )

    logger.info(
        "Momentum backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )

    return result
