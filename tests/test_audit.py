"""Tests for audit trail."""

from src.monitoring.audit import AuditEventType, AuditTrail


class TestAuditTrail:
    def test_log_entry(self):
        audit = AuditTrail()
        entry = audit.log(AuditEventType.SIGNAL, "MeanReversion", {"symbol": "BTC/USDT", "signal": "BUY"})

        assert entry.event_type == AuditEventType.SIGNAL
        assert entry.source == "MeanReversion"
        assert entry.details["symbol"] == "BTC/USDT"
        assert audit.total_entries == 1

    def test_convenience_methods(self):
        audit = AuditTrail()
        audit.log_signal("MR", "BTC/USDT", "BUY", confidence=0.8)
        audit.log_order(AuditEventType.ORDER_CREATED, "ORD-001", qty=0.1)
        audit.log_risk("drawdown", True, value=0.05)

        assert audit.total_entries == 3

    def test_query_by_type(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "S1", {})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {})
        audit.log(AuditEventType.SIGNAL, "S2", {})

        signals = audit.query(event_type=AuditEventType.SIGNAL)
        assert len(signals) == 2

    def test_query_by_source(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "MR", {})
        audit.log(AuditEventType.SIGNAL, "PT", {})
        audit.log(AuditEventType.SIGNAL, "MR", {})

        mr = audit.query(source="MR")
        assert len(mr) == 2

    def test_summary(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "MR", {})
        audit.log(AuditEventType.SIGNAL, "PT", {})
        audit.log(AuditEventType.ORDER_CREATED, "OMS", {})

        summary = audit.summary()
        assert summary["total_entries"] == 3
        assert summary["by_type"]["SIGNAL"] == 2
        assert summary["by_source"]["OMS"] == 1

    def test_to_dataframe(self):
        audit = AuditTrail()
        audit.log(AuditEventType.SIGNAL, "MR", {"x": 1})
        audit.log(AuditEventType.ORDER_FILLED, "OMS", {"y": 2})

        df = audit.to_dataframe()
        assert len(df) == 2
        assert "event_type" in df.columns

    def test_max_memory(self):
        audit = AuditTrail(max_memory=5)
        for i in range(10):
            audit.log(AuditEventType.SIGNAL, "S", {"i": i})
        assert audit.total_entries == 5
