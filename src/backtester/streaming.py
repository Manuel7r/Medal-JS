"""Streaming backtest runner — processes bars one-by-one with WebSocket progress."""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, Side, SignalType, Position, Trade
from src.backtester.metrics import compute_metrics, BacktestMetrics


@dataclass
class StreamingState:
    """Current state of a streaming backtest."""

    strategy: str = ""
    symbol: str = ""
    bar_index: int = 0
    total_bars: int = 0
    equity: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    running: bool = False


class StreamingBacktestRunner:
    """Runs a backtest bar-by-bar, broadcasting progress via a callback."""

    def __init__(
        self,
        broadcast_fn: Callable[[str, dict], Awaitable[None]],
        initial_capital: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
        position_size_pct: float = 0.03,
        broadcast_every: int = 5,
        delay_ms: int = 50,
    ) -> None:
        self.broadcast = broadcast_fn
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size_pct = position_size_pct
        self.broadcast_every = broadcast_every
        self.delay_ms = delay_ms
        self.state = StreamingState()

        # Reuse BacktestEngine helpers for position management
        self._engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            position_size_pct=position_size_pct,
        )

    async def run_streaming(
        self,
        data: pd.DataFrame,
        signal_fn: Callable,
        symbol: str,
        strategy: str,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        atr_stop_multiplier: float | None = None,
        atr_tp_multiplier: float | None = None,
    ) -> dict:
        """Run backtest bar-by-bar with async progress broadcasting.

        Returns final metrics dict.
        """
        data = data.reset_index(drop=True).copy()
        n = len(data)
        has_atr = "atr_14" in data.columns and atr_stop_multiplier is not None

        self.state = StreamingState(
            strategy=strategy,
            symbol=symbol,
            total_bars=n,
            equity=self.initial_capital,
            running=True,
        )

        equity = self.initial_capital
        cash = self.initial_capital
        position: Position | None = None
        trades: list[Trade] = []
        equity_values: list[float] = []

        for i in range(n):
            bar = data.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            timestamp = bar["timestamp"]

            # Dynamic ATR-based stops
            if has_atr:
                atr_val = float(bar["atr_14"]) if not pd.isna(bar["atr_14"]) else 0
                sl_pct = (atr_stop_multiplier * atr_val / close) if close > 0 and atr_val > 0 else stop_loss_pct
                tp_pct = (atr_tp_multiplier * atr_val / close) if close > 0 and atr_val > 0 and atr_tp_multiplier else take_profit_pct
            else:
                sl_pct = stop_loss_pct
                tp_pct = take_profit_pct

            # Check stops
            if position is not None:
                closed = self._engine._check_stops(position, high, low, close, timestamp, cash, trades)
                if closed:
                    cash = closed
                    position = None

            # Generate signal
            signal = signal_fn(data, i)

            # Execute
            if position is None and signal in (SignalType.BUY, SignalType.SELL):
                side = Side.LONG if signal == SignalType.BUY else Side.SHORT
                position, cost = self._engine._open_position(
                    symbol, side, close, timestamp, cash,
                    sl_pct, tp_pct,
                )
                cash -= cost
            elif position is not None and signal == SignalType.EXIT:
                cash = self._engine._close_position(position, close, timestamp, cash, trades)
                position = None
            elif position is not None and (
                (position.side == Side.LONG and signal == SignalType.SELL)
                or (position.side == Side.SHORT and signal == SignalType.BUY)
            ):
                cash = self._engine._close_position(position, close, timestamp, cash, trades)
                position = None
                side = Side.LONG if signal == SignalType.BUY else Side.SHORT
                position, cost = self._engine._open_position(
                    symbol, side, close, timestamp, cash,
                    sl_pct, tp_pct,
                )
                cash -= cost

            # Mark to market
            if position is not None:
                equity = cash + self._engine._mark_to_market(position, close)
            else:
                equity = cash

            equity_values.append(equity)

            # Update state
            self.state.bar_index = i + 1
            self.state.equity = equity
            self.state.equity_curve.append(equity)

            # Track new trades
            if len(trades) > len(self.state.trades):
                new_trade = trades[-1]
                trade_dict = {
                    "side": new_trade.side.value,
                    "entry_price": round(new_trade.entry_price, 2),
                    "exit_price": round(new_trade.exit_price, 2),
                    "pnl": round(new_trade.pnl, 2),
                    "return_pct": round(new_trade.return_pct * 100, 2),
                }
                self.state.trades.append(trade_dict)

                # Broadcast trade immediately
                await self.broadcast("backtest_trade", {
                    "strategy": strategy,
                    "symbol": symbol,
                    "trade": trade_dict,
                    "total_trades": len(trades),
                })

            # Broadcast progress every N bars
            if (i + 1) % self.broadcast_every == 0 or i == n - 1:
                await self.broadcast("backtest_progress", {
                    "strategy": strategy,
                    "symbol": symbol,
                    "bar_index": i + 1,
                    "total_bars": n,
                    "progress_pct": round((i + 1) / n * 100, 1),
                    "equity": round(equity, 2),
                    "total_trades": len(trades),
                    "equity_curve": equity_values[-100:],  # last 100 points
                })

                # Yield control to event loop
                await asyncio.sleep(self.delay_ms / 1000)

        # Close remaining position
        if position is not None:
            last_close = float(data.iloc[-1]["close"])
            last_ts = data.iloc[-1]["timestamp"]
            cash = self._engine._close_position(position, last_close, last_ts, cash, trades)
            equity_values[-1] = cash

        # Compute final metrics
        equity_curve = pd.Series(equity_values, name="equity")
        trade_returns = pd.Series([t.return_pct for t in trades], dtype=float)
        metrics = compute_metrics(equity_curve, trade_returns)

        metrics_dict = {
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "total_return": round(metrics.total_return * 100, 2),
            "max_drawdown": round(metrics.max_drawdown * 100, 2),
            "win_rate": round(metrics.win_rate * 100, 1),
            "total_trades": metrics.total_trades,
            "profit_factor": round(metrics.profit_factor, 2),
        }
        self.state.metrics = metrics_dict
        self.state.running = False

        # Broadcast completion
        await self.broadcast("backtest_complete", {
            "strategy": strategy,
            "symbol": symbol,
            "metrics": metrics_dict,
            "final_equity": round(equity_values[-1] if equity_values else self.initial_capital, 2),
            "total_trades": len(trades),
        })

        logger.info(
            "Streaming backtest complete: {} {} — Sharpe={:.2f}, Return={:.2%}, Trades={}",
            strategy, symbol, metrics.sharpe_ratio, metrics.total_return, len(trades),
        )

        return metrics_dict

    def get_status(self) -> dict:
        """Current streaming state."""
        return {
            "strategy": self.state.strategy,
            "symbol": self.state.symbol,
            "bar_index": self.state.bar_index,
            "total_bars": self.state.total_bars,
            "progress_pct": round(self.state.bar_index / max(self.state.total_bars, 1) * 100, 1),
            "equity": round(self.state.equity, 2),
            "total_trades": len(self.state.trades),
            "running": self.state.running,
            "metrics": self.state.metrics,
        }
