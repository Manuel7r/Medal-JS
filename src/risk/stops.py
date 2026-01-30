"""Dynamic stop loss and take profit management based on ATR.

Crypto defaults:
    atr_period: 14
    stop_loss_mult: 2.5x ATR
    take_profit_mult: 4.0x ATR
    trailing_activation: 2.0x initial risk
    trailing_atr_mult: 1.0x ATR (tighter trailing distance)
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.features.technical import atr as compute_atr


@dataclass
class StopParams:
    """Parameters for ATR-based stop management."""

    atr_period: int = 14
    stop_loss_mult: float = 2.5    # SL distance = 2.5x ATR
    take_profit_mult: float = 4.0  # TP distance = 4.0x ATR
    trailing_activation: float = 2.0  # activate trailing at 2x initial risk
    trailing_atr_mult: float = 1.0    # trailing distance = 1x ATR


@dataclass
class StopLevels:
    """Computed stop loss and take profit prices."""

    stop_loss: float
    take_profit: float
    atr_value: float


class DynamicStopManager:
    """Manages ATR-based stop loss, take profit, and trailing stops.

    Stop loss and take profit are set at entry based on ATR.
    After the position gains more than trailing_activation × initial_risk,
    the stop tightens to trailing_atr_mult × current ATR from the current price.
    The trailing stop only moves in the profitable direction.
    """

    def __init__(self, params: StopParams | None = None) -> None:
        self.params = params or StopParams()

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Compute ATR series."""
        return compute_atr(high, low, close, self.params.atr_period)

    def calculate_stops(
        self,
        entry_price: float,
        atr_value: float,
        direction: str = "LONG",
    ) -> StopLevels:
        """Calculate initial stop loss and take profit from entry.

        Args:
            entry_price: Entry fill price.
            atr_value: Current ATR value.
            direction: "LONG" or "SHORT".

        Returns:
            StopLevels with stop_loss, take_profit, and atr_value.
        """
        p = self.params

        if direction == "LONG":
            stop_loss = entry_price - p.stop_loss_mult * atr_value
            take_profit = entry_price + p.take_profit_mult * atr_value
        else:
            stop_loss = entry_price + p.stop_loss_mult * atr_value
            take_profit = entry_price - p.take_profit_mult * atr_value

        return StopLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=atr_value,
        )

    def update_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        current_atr: float,
        direction: str = "LONG",
    ) -> float:
        """Update trailing stop if activation threshold is met.

        The trailing stop only moves in the profitable direction:
            - LONG: stop can only go up
            - SHORT: stop can only go down

        Args:
            entry_price: Original entry price.
            current_price: Current market price.
            current_stop: Current stop loss level.
            current_atr: Current ATR value.
            direction: "LONG" or "SHORT".

        Returns:
            Updated stop loss price.
        """
        p = self.params
        initial_risk = abs(entry_price - current_stop)

        if direction == "LONG":
            profit = current_price - entry_price
            if profit > p.trailing_activation * initial_risk:
                new_stop = current_price - p.trailing_atr_mult * current_atr
                return max(new_stop, current_stop)  # only move up
        else:
            profit = entry_price - current_price
            if profit > p.trailing_activation * initial_risk:
                new_stop = current_price + p.trailing_atr_mult * current_atr
                return min(new_stop, current_stop)  # only move down

        return current_stop

    def is_stop_hit(
        self,
        stop_loss: float,
        take_profit: float,
        high: float,
        low: float,
        direction: str = "LONG",
    ) -> str | None:
        """Check if stop loss or take profit was hit on this bar.

        Args:
            stop_loss: Current stop loss level.
            take_profit: Current take profit level.
            high: Bar high price.
            low: Bar low price.
            direction: "LONG" or "SHORT".

        Returns:
            "stop_loss", "take_profit", or None.
        """
        if direction == "LONG":
            if low <= stop_loss:
                return "stop_loss"
            if high >= take_profit:
                return "take_profit"
        else:
            if high >= stop_loss:
                return "stop_loss"
            if low <= take_profit:
                return "take_profit"
        return None

    def manage_position(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        take_profit: float,
        current_atr: float,
        high: float,
        low: float,
        direction: str = "LONG",
    ) -> tuple[float, float, str | None]:
        """Full position management: update trailing + check stops.

        Args:
            entry_price: Original entry price.
            current_price: Current close price.
            current_stop: Current stop loss.
            take_profit: Take profit level.
            current_atr: Current ATR.
            high: Bar high.
            low: Bar low.
            direction: "LONG" or "SHORT".

        Returns:
            (updated_stop, take_profit, exit_reason)
            exit_reason is None if position stays open.
        """
        # Update trailing stop
        updated_stop = self.update_trailing_stop(
            entry_price, current_price, current_stop, current_atr, direction,
        )

        # Check if stopped out
        exit_reason = self.is_stop_hit(updated_stop, take_profit, high, low, direction)

        return updated_stop, take_profit, exit_reason
