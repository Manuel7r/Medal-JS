"""Position sizing using Kelly Criterion.

Uses a conservative fraction (25%) of the optimal Kelly to determine
position size as a percentage of equity, with hard caps.

Crypto defaults:
    kelly_fraction: 0.25
    max_position_pct: 0.03 (3%)
    max_leverage: 3.0
"""

from dataclasses import dataclass

from loguru import logger


@dataclass
class KellyParams:
    """Parameters for Kelly position sizing."""

    kelly_fraction: float = 0.25   # fraction of optimal Kelly
    max_position_pct: float = 0.03 # hard cap per position (3% for crypto)
    min_position_pct: float = 0.005 # minimum to bother trading
    max_leverage: float = 3.0


class KellyPositionSizer:
    """Conservative Kelly Criterion position sizing.

    Formula:
        f* = (p * b - q) / b

    where:
        p = win probability
        q = 1 - p
        b = avg_win / avg_loss (reward/risk ratio)

    We use kelly_fraction * f* as the actual position size,
    clamped to [min_position_pct, max_position_pct].
    """

    def __init__(self, params: KellyParams | None = None) -> None:
        self.params = params or KellyParams()

    def optimal_kelly(self, win_prob: float, risk_reward: float) -> float:
        """Compute raw optimal Kelly fraction.

        Args:
            win_prob: Historical win probability (0-1).
            risk_reward: avg_win / avg_loss ratio.

        Returns:
            Optimal fraction of capital to risk (can be negative if edge is negative).
        """
        if risk_reward <= 0:
            return 0.0
        q = 1.0 - win_prob
        return (win_prob * risk_reward - q) / risk_reward

    def position_size_pct(self, win_prob: float, risk_reward: float) -> float:
        """Compute safe position size as fraction of equity.

        Applies kelly_fraction and clamps to [min, max].

        Args:
            win_prob: Win probability (0-1).
            risk_reward: avg_win / avg_loss ratio.

        Returns:
            Position size as fraction of equity (e.g. 0.02 = 2%).
        """
        p = self.params
        kelly = self.optimal_kelly(win_prob, risk_reward)

        if kelly <= 0:
            return 0.0

        safe = kelly * p.kelly_fraction
        clamped = max(min(safe, p.max_position_pct), 0.0)

        if clamped < p.min_position_pct:
            return 0.0

        return clamped

    def position_value(
        self,
        equity: float,
        win_prob: float,
        risk_reward: float,
    ) -> float:
        """Compute dollar value to allocate.

        Args:
            equity: Current portfolio equity.
            win_prob: Win probability.
            risk_reward: avg_win / avg_loss ratio.

        Returns:
            Dollar amount for the position.
        """
        pct = self.position_size_pct(win_prob, risk_reward)
        value = equity * pct
        max_leveraged = equity * self.params.max_leverage
        return min(value, max_leveraged)

    def position_size_units(
        self,
        equity: float,
        price: float,
        win_prob: float,
        risk_reward: float,
    ) -> float:
        """Compute number of units/shares/coins to buy.

        Args:
            equity: Current portfolio equity.
            price: Current asset price.
            win_prob: Win probability.
            risk_reward: avg_win / avg_loss ratio.

        Returns:
            Number of units.
        """
        value = self.position_value(equity, win_prob, risk_reward)
        if price <= 0:
            return 0.0
        return value / price
