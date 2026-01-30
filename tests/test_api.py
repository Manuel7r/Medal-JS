"""Tests for API routes and WebSocket."""

import pytest
from unittest.mock import MagicMock, patch

from src.monitoring.audit import AuditTrail, AuditEventType


class TestAuditAPI:
    """Test audit trail API integration."""

    def test_audit_trail_log_and_query(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SYSTEM_START, "API", {"version": "1.0"})
        audit.log(AuditEventType.SIGNAL, "MeanReversion", {"symbol": "BTC/USDT", "signal": "BUY"})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {"order_id": "ORD-001"})

        entries = audit.query(limit=10)
        assert len(entries) == 3
        assert entries[0].event_type == AuditEventType.ORDER_CREATED  # most recent first

    def test_audit_query_by_type(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "MR", {"signal": "BUY"})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {"id": "1"})
        audit.log(AuditEventType.SIGNAL, "ML", {"signal": "SELL"})

        signals = audit.query(event_type=AuditEventType.SIGNAL)
        assert len(signals) == 2
        assert all(e.event_type == AuditEventType.SIGNAL for e in signals)

    def test_audit_summary(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "MR", {})
        audit.log(AuditEventType.SIGNAL, "ML", {})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {})

        summary = audit.summary()
        assert summary["total_entries"] == 3
        assert summary["by_type"]["SIGNAL"] == 2
        assert summary["by_type"]["ORDER_CREATED"] == 1

    def test_audit_memory_limit(self):
        audit = AuditTrail(max_memory=5)
        for i in range(10):
            audit.log(AuditEventType.SIGNAL, "test", {"i": i})
        assert audit.total_entries == 5


class TestWebSocketModule:
    """Test WebSocket module structure."""

    def test_websocket_module_imports(self):
        from src.api.websocket import ws_router, broadcast, _clients
        assert ws_router is not None
        assert len(_clients) == 0

    def test_serialize_dict(self):
        from src.api.websocket import _serialize
        result = _serialize({"key": "value"})
        assert result == {"key": "value"}

    def test_serialize_string(self):
        from src.api.websocket import _serialize
        result = _serialize("hello")
        assert result == {"value": "hello"}
