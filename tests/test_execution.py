"""Tests for OMS, BinanceBroker (mocked), and Scheduler."""

import time

import pytest

from src.execution.oms import (
    Fill,
    Order,
    OrderManagementSystem,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.execution.scheduler import TradingJob, TradingScheduler
from src.risk.portfolio import PortfolioRiskManager, RiskLimits


# =======================
# OMS Tests
# =======================

class TestOMSCreation:
    def test_create_market_order(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order("BTC/USDT", OrderSide.BUY, 0.5)
        assert order.status == OrderStatus.PENDING
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.5
        assert order.order_type == OrderType.MARKET

    def test_create_limit_order(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_limit_order("ETH/USDT", OrderSide.SELL, 2.0, 3500.0)
        assert order.status == OrderStatus.PENDING
        assert order.price == 3500.0
        assert order.order_type == OrderType.LIMIT

    def test_ids_are_unique(self) -> None:
        oms = OrderManagementSystem()
        o1 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        o2 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        assert o1.order_id != o2.order_id

    def test_metadata_stored(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order(
            "BTC/USDT", OrderSide.BUY, 1.0, metadata={"strategy": "pairs"}
        )
        assert order.metadata["strategy"] == "pairs"


class TestOMSLifecycle:
    def test_full_lifecycle(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order("BTC/USDT", OrderSide.BUY, 0.1)
        assert order.status == OrderStatus.PENDING

        oms.mark_submitted(order.order_id, "EX-123")
        assert order.status == OrderStatus.SUBMITTED
        assert order.exchange_order_id == "EX-123"

        fill = oms.mark_filled(order.order_id, 50000.0, 0.1, 5.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 50000.0
        assert order.commission == 5.0
        assert fill.price == 50000.0

    def test_partial_fill(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_submitted(order.order_id)

        oms.mark_filled(order.order_id, 50000.0, 0.3, 1.5)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 0.3

        oms.mark_filled(order.order_id, 50100.0, 0.7, 3.5)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.commission == 5.0

    def test_cancel(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_limit_order("BTC/USDT", OrderSide.BUY, 0.5, 48000.0)
        oms.mark_submitted(order.order_id)
        oms.mark_cancelled(order.order_id)
        assert order.status == OrderStatus.CANCELLED

    def test_reject(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order("BTC/USDT", OrderSide.BUY, 0.1)
        oms.mark_rejected(order.order_id, "Insufficient balance")
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient balance"

    def test_unknown_order_raises(self) -> None:
        oms = OrderManagementSystem()
        with pytest.raises(ValueError):
            oms.mark_submitted("NONEXISTENT")


class TestOMSQueries:
    def test_get_order(self) -> None:
        oms = OrderManagementSystem()
        order = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        found = oms.get_order(order.order_id)
        assert found is not None
        assert found.order_id == order.order_id

    def test_get_order_not_found(self) -> None:
        oms = OrderManagementSystem()
        assert oms.get_order("NOPE") is None

    def test_open_orders(self) -> None:
        oms = OrderManagementSystem()
        o1 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        o2 = oms.create_market_order("ETH/USDT", OrderSide.SELL, 2.0)
        oms.mark_submitted(o1.order_id)
        oms.mark_filled(o2.order_id, 3000.0)

        open_orders = oms.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == o1.order_id

    def test_open_orders_filter_by_symbol(self) -> None:
        oms = OrderManagementSystem()
        oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.create_market_order("ETH/USDT", OrderSide.BUY, 1.0)
        assert len(oms.get_open_orders("BTC/USDT")) == 1

    def test_fills(self) -> None:
        oms = OrderManagementSystem()
        o = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o.order_id, 50000.0, commission=5.0)
        fills = oms.get_fills("BTC/USDT")
        assert len(fills) == 1
        assert fills[0].price == 50000.0

    def test_total_counts(self) -> None:
        oms = OrderManagementSystem()
        o = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o.order_id, 50000.0)
        assert oms.total_orders == 1
        assert oms.total_fills == 1


class TestOMSReconciliation:
    def test_net_position_long(self) -> None:
        oms = OrderManagementSystem()
        o = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o.order_id, 50000.0)
        assert oms.net_position("BTC/USDT") == 1.0

    def test_net_position_flat(self) -> None:
        oms = OrderManagementSystem()
        o1 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o1.order_id, 50000.0)
        o2 = oms.create_market_order("BTC/USDT", OrderSide.SELL, 1.0)
        oms.mark_filled(o2.order_id, 51000.0)
        assert abs(oms.net_position("BTC/USDT")) < 1e-8

    def test_total_commission(self) -> None:
        oms = OrderManagementSystem()
        o1 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o1.order_id, 50000.0, commission=5.0)
        o2 = oms.create_market_order("BTC/USDT", OrderSide.SELL, 1.0)
        oms.mark_filled(o2.order_id, 51000.0, commission=5.1)
        assert abs(oms.total_commission("BTC/USDT") - 10.1) < 0.01

    def test_order_summary(self) -> None:
        oms = OrderManagementSystem()
        o1 = oms.create_market_order("BTC/USDT", OrderSide.BUY, 1.0)
        o2 = oms.create_market_order("ETH/USDT", OrderSide.BUY, 1.0)
        oms.mark_filled(o1.order_id, 50000.0)
        oms.mark_rejected(o2.order_id, "test")
        summary = oms.order_summary()
        assert summary["FILLED"] == 1
        assert summary["REJECTED"] == 1


# =======================
# Scheduler Tests
# =======================

class TestTradingJob:
    def test_execute_success(self) -> None:
        called = []
        job = TradingJob("test", lambda: called.append(1))
        job.execute()
        assert len(called) == 1
        assert job.run_count == 1
        assert job.error_count == 0
        assert job.last_run is not None

    def test_execute_error(self) -> None:
        def failing():
            raise ValueError("boom")
        job = TradingJob("fail", failing)
        job.execute()  # Should not raise
        assert job.run_count == 1
        assert job.error_count == 1
        assert job.last_error == "boom"

    def test_status(self) -> None:
        job = TradingJob("test", lambda: None)
        job.execute()
        status = job.status()
        assert status["name"] == "test"
        assert status["run_count"] == 1


class TestTradingScheduler:
    def test_add_and_run_job(self) -> None:
        counter = {"n": 0}
        def increment():
            counter["n"] += 1

        sched = TradingScheduler()
        sched.add_job("test", increment, seconds=60, start_now=True)
        # start_now=True runs it immediately
        assert counter["n"] == 1

    def test_remove_job(self) -> None:
        sched = TradingScheduler()
        sched.add_job("test", lambda: None, seconds=60, start_now=False)
        assert sched.remove_job("test") is True
        assert sched.remove_job("nonexistent") is False

    def test_run_job_now(self) -> None:
        counter = {"n": 0}
        def increment():
            counter["n"] += 1

        sched = TradingScheduler()
        sched.add_job("test", increment, seconds=60, start_now=False)
        sched.run_job_now("test")
        assert counter["n"] == 1

    def test_run_nonexistent_job(self) -> None:
        sched = TradingScheduler()
        assert sched.run_job_now("nope") is False

    def test_start_stop(self) -> None:
        sched = TradingScheduler()
        sched.add_job("test", lambda: None, seconds=60, start_now=False)
        sched.start()
        assert sched.is_running
        sched.stop()
        assert not sched.is_running

    def test_status(self) -> None:
        sched = TradingScheduler()
        sched.add_job("a", lambda: None, seconds=60, start_now=False)
        sched.add_job("b", lambda: None, seconds=120, start_now=False)
        status = sched.status()
        assert status["total_jobs"] == 2
        assert "a" in status["jobs"]
        assert "b" in status["jobs"]

    def test_kwargs_passed(self) -> None:
        results = {}
        def capture(key, value):
            results[key] = value

        sched = TradingScheduler()
        sched.add_job("test", capture, seconds=60, start_now=True, key="x", value=42)
        assert results["x"] == 42
