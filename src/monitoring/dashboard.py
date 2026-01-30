"""Streamlit dashboard for real-time trading monitoring.

Displays:
    - Portfolio equity curve and drawdown
    - Open positions and P&L
    - Order history and fill log
    - Risk metrics (drawdown, daily loss, leverage)
    - Alert history
    - Scheduler job status

Run with: streamlit run src/monitoring/dashboard.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class EquityPoint:
    """Single equity curve data point."""

    timestamp: datetime
    equity: float
    drawdown_pct: float = 0.0
    daily_pnl: float = 0.0


@dataclass
class PositionSnapshot:
    """Current position state."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float


@dataclass
class DashboardState:
    """Aggregated state for dashboard rendering.

    Collects data from OMS, risk manager, alert manager, and scheduler
    into a single snapshot that the Streamlit UI can render.
    """

    # Portfolio
    equity: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_commission: float = 0.0

    # Positions
    positions: list[PositionSnapshot] = field(default_factory=list)
    open_position_count: int = 0

    # Orders
    total_orders: int = 0
    total_fills: int = 0
    order_summary: dict[str, int] = field(default_factory=dict)

    # Risk
    leverage: float = 0.0
    max_drawdown_pct: float = 0.0
    is_suspended: bool = False
    is_paused: bool = False

    # Equity history
    equity_history: list[EquityPoint] = field(default_factory=list)

    # Alerts
    recent_alerts: list[dict[str, Any]] = field(default_factory=list)
    unacknowledged_alerts: int = 0

    # Scheduler
    scheduler_running: bool = False
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DashboardDataCollector:
    """Collects data from system components into DashboardState.

    Args:
        oms: OrderManagementSystem instance.
        risk_manager: PortfolioRiskManager instance (optional).
        alert_manager: AlertManager instance (optional).
        scheduler: TradingScheduler instance (optional).
    """

    def __init__(
        self,
        oms: Any = None,
        risk_manager: Any = None,
        alert_manager: Any = None,
        scheduler: Any = None,
    ) -> None:
        self.oms = oms
        self.risk_manager = risk_manager
        self.alert_manager = alert_manager
        self.scheduler = scheduler
        self._equity_history: list[EquityPoint] = []

    def record_equity(self, equity: float, drawdown_pct: float = 0.0, daily_pnl: float = 0.0) -> None:
        """Record an equity data point for the curve."""
        self._equity_history.append(EquityPoint(
            timestamp=datetime.now(timezone.utc),
            equity=equity,
            drawdown_pct=drawdown_pct,
            daily_pnl=daily_pnl,
        ))

    def collect(self) -> DashboardState:
        """Collect current state from all components.

        Returns:
            DashboardState snapshot.
        """
        state = DashboardState()
        state.equity_history = list(self._equity_history)

        # OMS data
        if self.oms is not None:
            state.total_orders = self.oms.total_orders
            state.total_fills = self.oms.total_fills
            state.order_summary = self.oms.order_summary()
            state.total_commission = self.oms.total_commission()

        # Risk manager data
        if self.risk_manager is not None:
            try:
                summary = self.risk_manager.risk_summary()
                state.equity = summary.get("equity", 0)
                state.peak_equity = summary.get("peak_equity", 0)
                state.drawdown_pct = summary.get("drawdown", 0)
                state.daily_pnl = summary.get("daily_loss", 0)
                state.daily_pnl_pct = summary.get("daily_loss", 0)
                state.leverage = summary.get("leverage", 0)
                state.open_position_count = summary.get("positions", 0)
                state.is_suspended = not self.risk_manager.is_suspended().allowed
                state.is_paused = not self.risk_manager.is_paused().allowed
            except Exception as e:
                logger.error("Dashboard: Error collecting risk data: {}", e)

        # Alert data
        if self.alert_manager is not None:
            state.unacknowledged_alerts = len(self.alert_manager.unacknowledged())
            recent = self.alert_manager.history[-10:] if self.alert_manager.history else []
            state.recent_alerts = [
                {
                    "type": a.alert_type.value,
                    "level": a.level.value,
                    "message": a.message,
                    "time": a.timestamp.isoformat(),
                    "acknowledged": a.acknowledged,
                }
                for a in recent
            ]

        # Scheduler data
        if self.scheduler is not None:
            try:
                sched_status = self.scheduler.status()
                state.scheduler_running = sched_status.get("running", False)
                state.jobs = sched_status.get("jobs", {})
            except Exception as e:
                logger.error("Dashboard: Error collecting scheduler data: {}", e)

        state.last_update = datetime.now(timezone.utc)
        return state

    def equity_dataframe(self) -> pd.DataFrame:
        """Convert equity history to DataFrame for plotting.

        Returns:
            DataFrame with columns: timestamp, equity, drawdown_pct, daily_pnl.
        """
        if not self._equity_history:
            return pd.DataFrame(columns=["timestamp", "equity", "drawdown_pct", "daily_pnl"])

        return pd.DataFrame([
            {
                "timestamp": p.timestamp,
                "equity": p.equity,
                "drawdown_pct": p.drawdown_pct,
                "daily_pnl": p.daily_pnl,
            }
            for p in self._equity_history
        ])


def render_dashboard(state: DashboardState) -> dict[str, Any]:
    """Prepare dashboard data for rendering.

    This function structures the DashboardState into sections
    that can be consumed by Streamlit or any other UI framework.

    Args:
        state: Current dashboard state.

    Returns:
        Dict organized by dashboard section.
    """
    return {
        "portfolio": {
            "equity": state.equity,
            "peak_equity": state.peak_equity,
            "drawdown_pct": state.drawdown_pct,
            "daily_pnl": state.daily_pnl,
            "daily_pnl_pct": state.daily_pnl_pct,
            "total_commission": state.total_commission,
        },
        "positions": {
            "count": state.open_position_count,
            "leverage": state.leverage,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "pnl_pct": p.pnl_pct,
                }
                for p in state.positions
            ],
        },
        "orders": {
            "total_orders": state.total_orders,
            "total_fills": state.total_fills,
            "summary": state.order_summary,
        },
        "risk": {
            "drawdown_pct": state.drawdown_pct,
            "leverage": state.leverage,
            "suspended": state.is_suspended,
            "paused": state.is_paused,
        },
        "alerts": {
            "unacknowledged": state.unacknowledged_alerts,
            "recent": state.recent_alerts,
        },
        "scheduler": {
            "running": state.scheduler_running,
            "jobs": state.jobs,
        },
        "last_update": state.last_update.isoformat(),
    }
