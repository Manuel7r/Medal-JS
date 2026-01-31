"""Backtesting engine with commission and slippage modeling."""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from loguru import logger

from src.backtester.metrics import BacktestMetrics, compute_metrics


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    commission: float
    slippage_cost: float
    pnl: float
    return_pct: float


@dataclass
class Position:
    """Open position tracker."""

    symbol: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class BacktestResult:
    """Full backtest output."""

    metrics: BacktestMetrics
    equity_curve: pd.Series
    trades: list[Trade]
    signals: pd.DataFrame


class BacktestEngine:
    """Event-driven backtest engine for a single symbol.

    Processes OHLCV bars sequentially, applies a signal function,
    and simulates execution with commissions and slippage.

    Args:
        initial_capital: Starting equity.
        commission_rate: Commission per side (e.g. 0.001 = 0.1%).
        slippage_rate: Slippage per trade (e.g. 0.001 = 0.1%).
        position_size_pct: Fraction of equity per trade.
        periods_per_year: For annualization (8760 = hourly 24/7).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
        position_size_pct: float = 0.03,
        periods_per_year: float = 8760,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size_pct = position_size_pct
        self.periods_per_year = periods_per_year

    def run(
        self,
        data: pd.DataFrame,
        signal_fn,
        symbol: str = "SYM",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        atr_stop_multiplier: float | None = None,
        atr_tp_multiplier: float | None = None,
    ) -> BacktestResult:
        """Run backtest on OHLCV data.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
                  and any extra feature columns the signal function needs.
            signal_fn: Callable(df, index) -> SignalType. Receives the full DataFrame
                       and current bar index, returns a signal.
            symbol: Symbol name for trade records.
            stop_loss_pct: Optional fixed stop loss as fraction (e.g. 0.05 = 5%).
            take_profit_pct: Optional fixed take profit as fraction.

        Returns:
            BacktestResult with metrics, equity curve, trades, and signals.
        """
        data = data.reset_index(drop=True).copy()
        n = len(data)
        has_atr = "atr_14" in data.columns and atr_stop_multiplier is not None

        equity = self.initial_capital
        cash = self.initial_capital
        position: Position | None = None
        trades: list[Trade] = []
        equity_values: list[float] = []
        signal_records: list[dict] = []

        for i in range(n):
            bar = data.iloc[i]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            timestamp = bar["timestamp"]

            # Compute dynamic ATR-based stop/tp for this bar
            if has_atr:
                atr_val = float(bar["atr_14"]) if not pd.isna(bar["atr_14"]) else 0
                sl_pct = (atr_stop_multiplier * atr_val / close) if close > 0 and atr_val > 0 else stop_loss_pct
                tp_pct = (atr_tp_multiplier * atr_val / close) if close > 0 and atr_val > 0 and atr_tp_multiplier else take_profit_pct
            else:
                sl_pct = stop_loss_pct
                tp_pct = take_profit_pct

            # Check stops on open position
            if position is not None:
                closed = self._check_stops(position, high, low, close, timestamp, cash, trades)
                if closed:
                    cash = closed
                    position = None

            # Generate signal
            signal = signal_fn(data, i)
            signal_records.append({"timestamp": timestamp, "signal": signal.value if isinstance(signal, SignalType) else str(signal)})

            # Execute signal
            if position is None and signal in (SignalType.BUY, SignalType.SELL):
                side = Side.LONG if signal == SignalType.BUY else Side.SHORT
                position, cost = self._open_position(
                    symbol, side, close, timestamp, cash,
                    sl_pct, tp_pct,
                )
                cash -= cost

            elif position is not None and signal == SignalType.EXIT:
                cash = self._close_position(position, close, timestamp, cash, trades)
                position = None

            # Flip: close and open opposite
            elif position is not None and (
                (position.side == Side.LONG and signal == SignalType.SELL)
                or (position.side == Side.SHORT and signal == SignalType.BUY)
            ):
                cash = self._close_position(position, close, timestamp, cash, trades)
                position = None
                side = Side.LONG if signal == SignalType.BUY else Side.SHORT
                position, cost = self._open_position(
                    symbol, side, close, timestamp, cash,
                    sl_pct, tp_pct,
                )
                cash -= cost

            # Mark to market
            if position is not None:
                mtm = self._mark_to_market(position, close)
                equity = cash + mtm
            else:
                equity = cash

            equity_values.append(equity)

        # Close any remaining position at last close
        if position is not None:
            last_close = float(data.iloc[-1]["close"])
            last_ts = data.iloc[-1]["timestamp"]
            cash = self._close_position(position, last_close, last_ts, cash, trades)
            equity_values[-1] = cash

        equity_curve = pd.Series(equity_values, name="equity")
        trade_returns = pd.Series([t.return_pct for t in trades], dtype=float)
        signals_df = pd.DataFrame(signal_records)

        metrics = compute_metrics(
            equity_curve=equity_curve,
            trade_returns=trade_returns,
            periods_per_year=self.periods_per_year,
        )

        logger.info(
            "Backtest complete: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
            len(trades), metrics.sharpe_ratio, metrics.total_return, metrics.max_drawdown,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            signals=signals_df,
        )

    def _open_position(
        self,
        symbol: str,
        side: Side,
        price: float,
        timestamp: pd.Timestamp,
        cash: float,
        stop_loss_pct: float | None,
        take_profit_pct: float | None,
    ) -> tuple[Position, float]:
        """Open a new position. Returns (Position, capital_used)."""
        notional = cash * self.position_size_pct
        slippage = price * self.slippage_rate
        fill_price = price + slippage if side == Side.LONG else price - slippage
        size = notional / fill_price
        commission = notional * self.commission_rate

        sl = None
        tp = None
        if stop_loss_pct is not None:
            sl = fill_price * (1 - stop_loss_pct) if side == Side.LONG else fill_price * (1 + stop_loss_pct)
        if take_profit_pct is not None:
            tp = fill_price * (1 + take_profit_pct) if side == Side.LONG else fill_price * (1 - take_profit_pct)

        pos = Position(
            symbol=symbol,
            side=side,
            entry_time=timestamp,
            entry_price=fill_price,
            size=size,
            stop_loss=sl,
            take_profit=tp,
        )
        return pos, notional + commission

    def _close_position(
        self,
        position: Position,
        price: float,
        timestamp: pd.Timestamp,
        cash: float,
        trades: list[Trade],
    ) -> float:
        """Close position at price. Returns updated cash."""
        slippage = price * self.slippage_rate
        if position.side == Side.LONG:
            fill_price = price - slippage
        else:
            fill_price = price + slippage

        notional_exit = position.size * fill_price
        notional_entry = position.size * position.entry_price
        commission = (notional_entry + notional_exit) * self.commission_rate
        slippage_cost = position.size * slippage * 2  # entry + exit

        if position.side == Side.LONG:
            pnl = (fill_price - position.entry_price) * position.size - commission
        else:
            pnl = (position.entry_price - fill_price) * position.size - commission

        return_pct = pnl / notional_entry if notional_entry > 0 else 0.0

        trades.append(Trade(
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=fill_price,
            size=position.size,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=pnl,
            return_pct=return_pct,
        ))

        return cash + notional_exit - commission

    def _check_stops(
        self,
        position: Position,
        high: float,
        low: float,
        close: float,
        timestamp: pd.Timestamp,
        cash: float,
        trades: list[Trade],
    ) -> float | None:
        """Check if stop loss or take profit is hit. Returns updated cash or None."""
        if position.side == Side.LONG:
            if position.stop_loss is not None and low <= position.stop_loss:
                return self._close_position(position, position.stop_loss, timestamp, cash, trades)
            if position.take_profit is not None and high >= position.take_profit:
                return self._close_position(position, position.take_profit, timestamp, cash, trades)
        else:
            if position.stop_loss is not None and high >= position.stop_loss:
                return self._close_position(position, position.stop_loss, timestamp, cash, trades)
            if position.take_profit is not None and low <= position.take_profit:
                return self._close_position(position, position.take_profit, timestamp, cash, trades)
        return None

    def _mark_to_market(self, position: Position, price: float) -> float:
        """Unrealized value of open position."""
        if position.side == Side.LONG:
            return position.size * price
        else:
            return position.size * (2 * position.entry_price - price)
