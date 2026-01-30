"""Trade audit trail for compliance logging.

Records all trading decisions, order events, and system actions
in a structured, queryable format for post-mortem analysis.

Each AuditEntry has:
    - timestamp
    - event_type (SIGNAL, ORDER, FILL, RISK_CHECK, SYSTEM)
    - details dict
    - source (strategy name, component name)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class AuditEventType(str, Enum):
    SIGNAL = "SIGNAL"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RISK_CHECK = "RISK_CHECK"
    RISK_BREACH = "RISK_BREACH"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    DATA_FETCH = "DATA_FETCH"
    MODEL_RETRAIN = "MODEL_RETRAIN"
    CONFIG_CHANGE = "CONFIG_CHANGE"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    event_type: AuditEventType
    source: str
    details: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "source": self.source,
            "details": self.details,
        }


class AuditTrail:
    """In-memory audit trail with optional file persistence.

    Usage:
        audit = AuditTrail(log_file="audit.jsonl")
        audit.log(AuditEventType.SIGNAL, "MeanReversion", {"symbol": "BTC/USDT", "signal": "BUY"})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {"order_id": "ORD-001", "qty": 0.1})
    """

    def __init__(self, log_file: str | None = None, max_memory: int = 10000) -> None:
        self._entries: list[AuditEntry] = []
        self._max_memory = max_memory
        self._log_file = Path(log_file) if log_file else None

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: AuditEventType,
        source: str,
        details: dict[str, Any],
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            event_type: Type of event.
            source: Component that generated the event.
            details: Event-specific data.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(event_type=event_type, source=source, details=details)
        self._entries.append(entry)

        # Trim if over memory limit
        if len(self._entries) > self._max_memory:
            self._entries = self._entries[-self._max_memory:]

        # Persist to file
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception as e:
                logger.error("AuditTrail: Failed to write to file: {}", e)

        return entry

    def log_signal(self, strategy: str, symbol: str, signal: str, **extra: Any) -> AuditEntry:
        """Convenience: log a trading signal."""
        return self.log(AuditEventType.SIGNAL, strategy, {"symbol": symbol, "signal": signal, **extra})

    def log_order(self, event: AuditEventType, order_id: str, **extra: Any) -> AuditEntry:
        """Convenience: log an order event."""
        return self.log(event, "OMS", {"order_id": order_id, **extra})

    def log_risk(self, check_type: str, passed: bool, **extra: Any) -> AuditEntry:
        """Convenience: log a risk check."""
        event = AuditEventType.RISK_CHECK if passed else AuditEventType.RISK_BREACH
        return self.log(event, "RiskManager", {"check_type": check_type, "passed": passed, **extra})

    # --- Queries ---

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def query(
        self,
        event_type: AuditEventType | None = None,
        source: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters.

        Args:
            event_type: Filter by event type.
            source: Filter by source component.
            since: Only entries after this time.
            limit: Max entries to return.

        Returns:
            Filtered list of AuditEntry (most recent first).
        """
        results = self._entries
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if source:
            results = [e for e in results if e.source == source]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return list(reversed(results[-limit:]))

    def to_dataframe(self) -> pd.DataFrame:
        """Export audit trail as DataFrame."""
        if not self._entries:
            return pd.DataFrame(columns=["timestamp", "event_type", "source", "details"])
        return pd.DataFrame([e.to_dict() for e in self._entries])

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        """Get audit trail summary."""
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for e in self._entries:
            by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1
            by_source[e.source] = by_source.get(e.source, 0) + 1
        return {
            "total_entries": len(self._entries),
            "by_type": by_type,
            "by_source": by_source,
        }
