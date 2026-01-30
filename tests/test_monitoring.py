"""Tests for AlertManager, DashboardDataCollector, and dashboard rendering."""

import time

import pytest

from src.monitoring.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertRule,
    AlertType,
    create_default_rules,
)
from src.monitoring.dashboard import (
    DashboardDataCollector,
    DashboardState,
    EquityPoint,
    PositionSnapshot,
    render_dashboard,
)


# =======================
# Alert Tests
# =======================

class TestAlert:
    def test_create_alert(self) -> None:
        alert = Alert(
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            message="Drawdown at 12%",
        )
        assert alert.alert_type == AlertType.DRAWDOWN
        assert alert.level == AlertLevel.WARNING
        assert not alert.acknowledged

    def test_acknowledge(self) -> None:
        alert = Alert(
            alert_type=AlertType.ERROR,
            level=AlertLevel.CRITICAL,
            message="Connection lost",
        )
        alert.acknowledge()
        assert alert.acknowledged


class TestAlertRule:
    def test_create_rule(self) -> None:
        rule = AlertRule(
            name="test",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            condition=lambda m: m.get("dd", 0) > 10,
            message_template="DD at {dd}%",
        )
        assert rule.name == "test"
        assert rule.cooldown_seconds == 300


class TestAlertManager:
    def test_add_and_remove_rule(self) -> None:
        mgr = AlertManager()
        rule = AlertRule(
            name="r1",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            condition=lambda m: True,
            message_template="test",
        )
        mgr.add_rule(rule)
        assert mgr.remove_rule("r1") is True
        assert mgr.remove_rule("r1") is False

    def test_evaluate_fires_alert(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="dd",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.CRITICAL,
            condition=lambda m: m.get("drawdown_pct", 0) > 10,
            message_template="DD at {drawdown_pct:.1f}%",
            cooldown_seconds=0,
        ))
        alerts = mgr.evaluate({"drawdown_pct": 15.0})
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.DRAWDOWN
        assert "15.0" in alerts[0].message

    def test_evaluate_no_fire(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="dd",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            condition=lambda m: m.get("drawdown_pct", 0) > 10,
            message_template="DD",
            cooldown_seconds=0,
        ))
        alerts = mgr.evaluate({"drawdown_pct": 5.0})
        assert len(alerts) == 0

    def test_cooldown_prevents_repeat(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="dd",
            alert_type=AlertType.DRAWDOWN,
            level=AlertLevel.WARNING,
            condition=lambda m: True,
            message_template="test",
            cooldown_seconds=9999,
        ))
        mgr.evaluate({"x": 1})
        # Second call within cooldown
        alerts = mgr.evaluate({"x": 1})
        assert len(alerts) == 0

    def test_callback_invoked(self) -> None:
        mgr = AlertManager()
        received: list[Alert] = []
        mgr.add_callback(lambda a: received.append(a))
        mgr.add_rule(AlertRule(
            name="t",
            alert_type=AlertType.ERROR,
            level=AlertLevel.CRITICAL,
            condition=lambda m: True,
            message_template="err",
            cooldown_seconds=0,
        ))
        mgr.evaluate({})
        assert len(received) == 1

    def test_fire_alert_manual(self) -> None:
        mgr = AlertManager()
        alert = mgr.fire_alert(AlertType.STOP_HIT, AlertLevel.WARNING, "Stop hit on BTC")
        assert alert.alert_type == AlertType.STOP_HIT
        assert mgr.total_alerts == 1

    def test_history_filters(self) -> None:
        mgr = AlertManager()
        mgr.fire_alert(AlertType.DRAWDOWN, AlertLevel.WARNING, "dd")
        mgr.fire_alert(AlertType.ERROR, AlertLevel.CRITICAL, "err")
        mgr.fire_alert(AlertType.DRAWDOWN, AlertLevel.CRITICAL, "dd2")

        assert len(mgr.history_by_type(AlertType.DRAWDOWN)) == 2
        assert len(mgr.history_by_level(AlertLevel.CRITICAL)) == 2

    def test_unacknowledged(self) -> None:
        mgr = AlertManager()
        a1 = mgr.fire_alert(AlertType.ERROR, AlertLevel.CRITICAL, "e1")
        mgr.fire_alert(AlertType.ERROR, AlertLevel.WARNING, "e2")
        a1.acknowledge()
        assert len(mgr.unacknowledged()) == 1

    def test_clear_history(self) -> None:
        mgr = AlertManager()
        mgr.fire_alert(AlertType.ERROR, AlertLevel.INFO, "x")
        mgr.clear_history()
        assert mgr.total_alerts == 0

    def test_summary(self) -> None:
        mgr = AlertManager()
        mgr.fire_alert(AlertType.DRAWDOWN, AlertLevel.WARNING, "a")
        mgr.fire_alert(AlertType.ERROR, AlertLevel.CRITICAL, "b")
        s = mgr.summary()
        assert s["total_alerts"] == 2
        assert s["by_type"]["DRAWDOWN"] == 1
        assert s["by_level"]["CRITICAL"] == 1

    def test_condition_error_does_not_crash(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="bad",
            alert_type=AlertType.CUSTOM,
            level=AlertLevel.INFO,
            condition=lambda m: 1 / 0,  # ZeroDivisionError
            message_template="x",
            cooldown_seconds=0,
        ))
        alerts = mgr.evaluate({})
        assert len(alerts) == 0  # No crash, no alert

    def test_bad_template_does_not_crash(self) -> None:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="tmpl",
            alert_type=AlertType.CUSTOM,
            level=AlertLevel.INFO,
            condition=lambda m: True,
            message_template="Value is {missing_key}",
            cooldown_seconds=0,
        ))
        alerts = mgr.evaluate({})
        assert len(alerts) == 1
        # Falls back to raw template
        assert "missing_key" in alerts[0].message


class TestDefaultRules:
    def test_creates_5_rules(self) -> None:
        rules = create_default_rules()
        assert len(rules) == 5

    def test_drawdown_critical_fires(self) -> None:
        rules = create_default_rules(max_drawdown_pct=15.0)
        mgr = AlertManager()
        for r in rules:
            mgr.add_rule(r)
        alerts = mgr.evaluate({"drawdown_pct": 16.0, "daily_loss_pct": 0, "open_positions": 0})
        types = [a.alert_type for a in alerts]
        assert AlertType.DRAWDOWN in types

    def test_daily_loss_fires(self) -> None:
        rules = create_default_rules(max_daily_loss_pct=3.0)
        mgr = AlertManager()
        for r in rules:
            mgr.add_rule(r)
        alerts = mgr.evaluate({"drawdown_pct": 0, "daily_loss_pct": 4.0, "open_positions": 0})
        types = [a.alert_type for a in alerts]
        assert AlertType.DAILY_LOSS in types

    def test_position_limit_fires(self) -> None:
        rules = create_default_rules(max_positions=20)
        mgr = AlertManager()
        for r in rules:
            mgr.add_rule(r)
        alerts = mgr.evaluate({"drawdown_pct": 0, "daily_loss_pct": 0, "open_positions": 20})
        types = [a.alert_type for a in alerts]
        assert AlertType.POSITION_LIMIT in types


# =======================
# Dashboard Tests
# =======================

class TestDashboardState:
    def test_defaults(self) -> None:
        state = DashboardState()
        assert state.equity == 0.0
        assert state.positions == []
        assert state.total_orders == 0

    def test_position_snapshot(self) -> None:
        pos = PositionSnapshot(
            symbol="BTC/USDT",
            quantity=0.5,
            entry_price=50000.0,
            current_price=52000.0,
            unrealized_pnl=1000.0,
            pnl_pct=4.0,
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.unrealized_pnl == 1000.0


class TestDashboardDataCollector:
    def test_collect_empty(self) -> None:
        collector = DashboardDataCollector()
        state = collector.collect()
        assert state.total_orders == 0
        assert state.equity == 0.0

    def test_record_equity(self) -> None:
        collector = DashboardDataCollector()
        collector.record_equity(10000.0, drawdown_pct=0.0)
        collector.record_equity(10500.0, drawdown_pct=0.0, daily_pnl=500.0)
        state = collector.collect()
        assert len(state.equity_history) == 2

    def test_equity_dataframe_empty(self) -> None:
        collector = DashboardDataCollector()
        df = collector.equity_dataframe()
        assert len(df) == 0
        assert "equity" in df.columns

    def test_equity_dataframe_with_data(self) -> None:
        collector = DashboardDataCollector()
        collector.record_equity(10000.0)
        collector.record_equity(10500.0)
        df = collector.equity_dataframe()
        assert len(df) == 2
        assert df["equity"].iloc[1] == 10500.0

    def test_collect_with_oms(self) -> None:
        """Test collection from a mock OMS."""
        class MockOMS:
            total_orders = 5
            total_fills = 3
            def order_summary(self):
                return {"FILLED": 3, "REJECTED": 2}
            def total_commission(self):
                return 12.5

        collector = DashboardDataCollector(oms=MockOMS())
        state = collector.collect()
        assert state.total_orders == 5
        assert state.total_fills == 3
        assert state.total_commission == 12.5

    def test_collect_with_alert_manager(self) -> None:
        mgr = AlertManager()
        mgr.fire_alert(AlertType.ERROR, AlertLevel.CRITICAL, "test error")
        collector = DashboardDataCollector(alert_manager=mgr)
        state = collector.collect()
        assert state.unacknowledged_alerts == 1
        assert len(state.recent_alerts) == 1

    def test_collect_with_scheduler(self) -> None:
        class MockScheduler:
            def status(self):
                return {"running": True, "jobs": {"data": {"run_count": 5}}}

        collector = DashboardDataCollector(scheduler=MockScheduler())
        state = collector.collect()
        assert state.scheduler_running is True
        assert "data" in state.jobs


class TestRenderDashboard:
    def test_render_empty_state(self) -> None:
        state = DashboardState()
        result = render_dashboard(state)
        assert "portfolio" in result
        assert "positions" in result
        assert "orders" in result
        assert "risk" in result
        assert "alerts" in result
        assert "scheduler" in result

    def test_render_with_positions(self) -> None:
        state = DashboardState()
        state.positions = [
            PositionSnapshot("BTC/USDT", 0.5, 50000, 52000, 1000, 4.0),
        ]
        result = render_dashboard(state)
        assert len(result["positions"]["positions"]) == 1
        assert result["positions"]["positions"][0]["symbol"] == "BTC/USDT"

    def test_render_with_alerts(self) -> None:
        state = DashboardState()
        state.recent_alerts = [{"type": "ERROR", "level": "CRITICAL", "message": "x"}]
        state.unacknowledged_alerts = 1
        result = render_dashboard(state)
        assert result["alerts"]["unacknowledged"] == 1
