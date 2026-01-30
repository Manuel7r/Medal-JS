"""Alert system for trading monitoring.

Monitors portfolio state and triggers alerts when risk thresholds
are breached. Supports multiple notification channels (log, webhook, callback).

Alert types:
    - DRAWDOWN: Portfolio drawdown exceeds threshold
    - DAILY_LOSS: Daily P&L loss exceeds threshold
    - STOP_HIT: A position stop loss was triggered
    - POSITION_LIMIT: Max positions reached
    - ERROR: System error (job failure, connectivity)
    - CUSTOM: User-defined alert
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from loguru import logger


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    DRAWDOWN = "DRAWDOWN"
    DAILY_LOSS = "DAILY_LOSS"
    STOP_HIT = "STOP_HIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    ERROR = "ERROR"
    CUSTOM = "CUSTOM"


@dataclass
class Alert:
    """Represents a single alert event."""

    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def acknowledge(self) -> None:
        self.acknowledged = True


@dataclass
class AlertRule:
    """Defines a condition that triggers an alert.

    Args:
        name: Unique rule identifier.
        alert_type: Type of alert to fire.
        level: Severity level.
        condition: Callable that receives metrics dict and returns True to trigger.
        message_template: Format string with {key} placeholders from metrics.
        cooldown_seconds: Minimum seconds between repeated alerts from this rule.
    """

    name: str
    alert_type: AlertType
    level: AlertLevel
    condition: Callable[[dict[str, Any]], bool]
    message_template: str
    cooldown_seconds: int = 300  # 5 min default


class AlertManager:
    """Manages alert rules, evaluation, and notification dispatch.

    Usage:
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high_drawdown",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.CRITICAL,
            condition=lambda m: m.get("drawdown_pct", 0) > 15,
            message_template="Drawdown at {drawdown_pct:.1f}% (limit 15%)",
        ))
        manager.add_callback(my_notification_func)
        alerts = manager.evaluate({"drawdown_pct": 18.5})
    """

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._history: list[Alert] = []
        self._callbacks: list[Callable[[Alert], None]] = []
        self._last_fired: dict[str, datetime] = {}

    # --- Rule management ---

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule
        logger.info("AlertManager: Added rule '{}'", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule. Returns True if removed."""
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a notification callback invoked on every alert."""
        self._callbacks.append(callback)

    # --- Evaluation ---

    def evaluate(self, metrics: dict[str, Any]) -> list[Alert]:
        """Evaluate all rules against current metrics.

        Args:
            metrics: Dict of current portfolio/system metrics.

        Returns:
            List of alerts that fired.
        """
        fired: list[Alert] = []
        now = datetime.now(timezone.utc)

        for rule in self._rules.values():
            # Check cooldown
            last = self._last_fired.get(rule.name)
            if last is not None:
                elapsed = (now - last).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue

            # Evaluate condition
            try:
                if not rule.condition(metrics):
                    continue
            except Exception as e:
                logger.error("AlertManager: Rule '{}' evaluation error: {}", rule.name, e)
                continue

            # Format message
            try:
                message = rule.message_template.format(**metrics)
            except (KeyError, ValueError):
                message = rule.message_template

            alert = Alert(
                alert_type=rule.alert_type,
                level=rule.level,
                message=message,
                data=dict(metrics),
            )

            self._history.append(alert)
            self._last_fired[rule.name] = now
            fired.append(alert)

            # Log
            if rule.level == AlertLevel.CRITICAL:
                logger.critical("ALERT [{}]: {}", rule.alert_type.value, message)
            elif rule.level == AlertLevel.WARNING:
                logger.warning("ALERT [{}]: {}", rule.alert_type.value, message)
            else:
                logger.info("ALERT [{}]: {}", rule.alert_type.value, message)

            # Notify callbacks
            for cb in self._callbacks:
                try:
                    cb(alert)
                except Exception as e:
                    logger.error("AlertManager: Callback error: {}", e)

        return fired

    def fire_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        """Manually fire an alert (bypasses rules).

        Args:
            alert_type: Type of alert.
            level: Severity.
            message: Alert message.
            data: Optional data payload.

        Returns:
            The created Alert.
        """
        alert = Alert(
            alert_type=alert_type,
            level=level,
            message=message,
            data=data or {},
        )
        self._history.append(alert)

        logger.warning("ALERT [{}]: {}", alert_type.value, message)
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error("AlertManager: Callback error: {}", e)

        return alert

    # --- Queries ---

    @property
    def history(self) -> list[Alert]:
        """All historical alerts."""
        return list(self._history)

    def unacknowledged(self) -> list[Alert]:
        """Get alerts that haven't been acknowledged."""
        return [a for a in self._history if not a.acknowledged]

    def history_by_type(self, alert_type: AlertType) -> list[Alert]:
        """Filter history by alert type."""
        return [a for a in self._history if a.alert_type == alert_type]

    def history_by_level(self, level: AlertLevel) -> list[Alert]:
        """Filter history by severity level."""
        return [a for a in self._history if a.level == level]

    @property
    def total_alerts(self) -> int:
        return len(self._history)

    def clear_history(self) -> None:
        """Clear all alert history."""
        self._history.clear()
        self._last_fired.clear()

    def summary(self) -> dict[str, Any]:
        """Get alert system summary."""
        by_type: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for a in self._history:
            by_type[a.alert_type.value] = by_type.get(a.alert_type.value, 0) + 1
            by_level[a.level.value] = by_level.get(a.level.value, 0) + 1

        return {
            "total_rules": len(self._rules),
            "total_alerts": len(self._history),
            "unacknowledged": len(self.unacknowledged()),
            "by_type": by_type,
            "by_level": by_level,
        }


# --- Default rules factory ---

def create_default_rules(
    max_drawdown_pct: float = 15.0,
    max_daily_loss_pct: float = 3.0,
    max_positions: int = 20,
) -> list[AlertRule]:
    """Create standard trading alert rules.

    Args:
        max_drawdown_pct: Drawdown warning threshold (%).
        max_daily_loss_pct: Daily loss warning threshold (%).
        max_positions: Position count warning threshold.

    Returns:
        List of pre-configured AlertRule objects.
    """
    return [
        AlertRule(
            name="drawdown_warning",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            condition=lambda m, t=max_drawdown_pct * 0.7: m.get("drawdown_pct", 0) > t,
            message_template="Drawdown at {drawdown_pct:.1f}%",
            cooldown_seconds=600,
        ),
        AlertRule(
            name="drawdown_critical",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.CRITICAL,
            condition=lambda m, t=max_drawdown_pct: m.get("drawdown_pct", 0) > t,
            message_template="CRITICAL: Drawdown at {drawdown_pct:.1f}% exceeds limit",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="daily_loss_warning",
            alert_type=AlertType.DAILY_LOSS,
            level=AlertLevel.WARNING,
            condition=lambda m, t=max_daily_loss_pct * 0.7: m.get("daily_loss_pct", 0) > t,
            message_template="Daily loss at {daily_loss_pct:.1f}%",
            cooldown_seconds=600,
        ),
        AlertRule(
            name="daily_loss_critical",
            alert_type=AlertType.DAILY_LOSS,
            level=AlertLevel.CRITICAL,
            condition=lambda m, t=max_daily_loss_pct: m.get("daily_loss_pct", 0) > t,
            message_template="CRITICAL: Daily loss at {daily_loss_pct:.1f}% exceeds limit",
            cooldown_seconds=300,
        ),
        AlertRule(
            name="position_limit",
            alert_type=AlertType.POSITION_LIMIT,
            level=AlertLevel.WARNING,
            condition=lambda m, t=max_positions: m.get("open_positions", 0) >= t,
            message_template="Position limit reached: {open_positions} positions",
            cooldown_seconds=900,
        ),
    ]
