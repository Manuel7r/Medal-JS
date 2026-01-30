"""Portfolio-level risk management.

Enforces hard limits on drawdown, daily loss, position count,
position size, and correlation between positions.

Crypto defaults:
    max_drawdown: 0.20 (20%)
    daily_loss_limit: 0.03 (3%)
    max_position_pct: 0.03 (3%)
    max_positions: 20
    max_correlation: 0.70
    max_leverage: 3.0
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskLimits:
    """Hard risk limits for the portfolio."""

    max_drawdown: float = 0.20
    daily_loss_limit: float = 0.03
    max_position_pct: float = 0.03
    max_positions: int = 20
    max_correlation: float = 0.70
    max_leverage: float = 3.0


@dataclass
class PortfolioPosition:
    """Tracks an open position in the portfolio."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    size: float
    entry_time: datetime | None = None


@dataclass
class RejectionReason:
    """Describes why a trade was rejected."""

    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class PortfolioRiskManager:
    """Validates trades against portfolio-level risk limits.

    Tracks equity, peak equity, daily P&L, and open positions to
    enforce all hard limits before allowing a new trade.
    """

    def __init__(self, limits: RiskLimits | None = None, initial_equity: float = 100_000.0) -> None:
        self.limits = limits or RiskLimits()
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.positions: dict[str, PortfolioPosition] = {}
        self._price_history: dict[str, list[float]] = {}

    # --- State updates ---

    def update_equity(self, equity: float) -> None:
        """Update current equity and peak."""
        self.equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def new_day(self) -> None:
        """Call at start of each trading day to reset daily P&L."""
        self.daily_start_equity = self.equity

    def add_position(self, position: PortfolioPosition) -> None:
        """Register an open position."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self.positions.pop(symbol, None)

    def record_price(self, symbol: str, price: float) -> None:
        """Record a price for correlation tracking."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)

    # --- Computed metrics ---

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak (0 to 1)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_loss(self) -> float:
        """Current daily loss fraction (0 to 1). Positive means loss."""
        if self.daily_start_equity <= 0:
            return 0.0
        loss = (self.daily_start_equity - self.equity) / self.daily_start_equity
        return max(loss, 0.0)

    @property
    def total_exposure(self) -> float:
        """Total notional exposure of all positions."""
        return sum(p.size * p.entry_price for p in self.positions.values())

    @property
    def leverage(self) -> float:
        """Current leverage = total_exposure / equity."""
        if self.equity <= 0:
            return 0.0
        return self.total_exposure / self.equity

    # --- Validation ---

    def can_open_position(
        self,
        symbol: str,
        price: float,
        size: float,
    ) -> RejectionReason:
        """Full pre-trade validation.

        Args:
            symbol: Asset to trade.
            price: Current price.
            size: Number of units.

        Returns:
            RejectionReason(allowed=True/False, reason=str).
        """
        limits = self.limits

        # 1. Drawdown check
        if self.drawdown >= limits.max_drawdown:
            return RejectionReason(False, f"Max drawdown exceeded: {self.drawdown:.2%} >= {limits.max_drawdown:.2%}")

        # 2. Daily loss check
        if self.daily_loss >= limits.daily_loss_limit:
            return RejectionReason(False, f"Daily loss limit reached: {self.daily_loss:.2%} >= {limits.daily_loss_limit:.2%}")

        # 3. Max positions check
        if len(self.positions) >= limits.max_positions:
            return RejectionReason(False, f"Max positions reached: {len(self.positions)} >= {limits.max_positions}")

        # 4. Position size check
        position_value = size * price
        if self.equity > 0:
            position_pct = position_value / self.equity
            if position_pct > limits.max_position_pct:
                return RejectionReason(False, f"Position size exceeds limit: {position_pct:.2%} > {limits.max_position_pct:.2%}")

        # 5. Leverage check
        new_exposure = self.total_exposure + position_value
        if self.equity > 0:
            new_leverage = new_exposure / self.equity
            if new_leverage > limits.max_leverage:
                return RejectionReason(False, f"Leverage limit exceeded: {new_leverage:.1f}x > {limits.max_leverage:.1f}x")

        # 6. Correlation check
        if symbol not in self.positions:
            corr_issue = self._check_correlation(symbol)
            if corr_issue:
                return RejectionReason(False, corr_issue)

        return RejectionReason(True, "OK")

    def _check_correlation(self, symbol: str) -> str | None:
        """Check if new symbol is too correlated with existing positions.

        Returns:
            Error message string or None if OK.
        """
        if symbol not in self._price_history or len(self._price_history[symbol]) < 30:
            return None

        new_prices = pd.Series(self._price_history[symbol])

        for existing_symbol in self.positions:
            if existing_symbol not in self._price_history:
                continue
            existing_prices = pd.Series(self._price_history[existing_symbol])

            min_len = min(len(new_prices), len(existing_prices))
            if min_len < 30:
                continue

            corr = new_prices.iloc[-min_len:].corr(existing_prices.iloc[-min_len:])
            if abs(corr) > self.limits.max_correlation:
                return (
                    f"High correlation with {existing_symbol}: "
                    f"{corr:.2f} > {self.limits.max_correlation}"
                )

        return None

    def is_suspended(self) -> RejectionReason:
        """Check if trading should be completely suspended.

        Returns:
            RejectionReason(False, reason) if suspended, (True, "OK") if active.
        """
        if self.drawdown >= self.limits.max_drawdown:
            return RejectionReason(False, f"SUSPENDED: Drawdown {self.drawdown:.2%} >= {self.limits.max_drawdown:.2%}")
        return RejectionReason(True, "OK")

    def is_paused(self) -> RejectionReason:
        """Check if trading is paused for the day.

        Returns:
            RejectionReason(False, reason) if paused, (True, "OK") if active.
        """
        if self.daily_loss >= self.limits.daily_loss_limit:
            return RejectionReason(False, f"PAUSED: Daily loss {self.daily_loss:.2%} >= {self.limits.daily_loss_limit:.2%}")
        return RejectionReason(True, "OK")

    def risk_summary(self) -> dict:
        """Current risk state summary."""
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "daily_loss": self.daily_loss,
            "positions": len(self.positions),
            "exposure": self.total_exposure,
            "leverage": self.leverage,
            "is_suspended": not self.is_suspended().allowed,
            "is_paused": not self.is_paused().allowed,
        }
