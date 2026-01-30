"""Tests for risk management: Kelly sizing, portfolio limits, dynamic stops."""

import numpy as np
import pandas as pd
import pytest

from src.risk.position_sizing import KellyParams, KellyPositionSizer
from src.risk.portfolio import (
    PortfolioPosition,
    PortfolioRiskManager,
    RejectionReason,
    RiskLimits,
)
from src.risk.stops import DynamicStopManager, StopParams


# =======================
# Kelly Position Sizing
# =======================

class TestKellyOptimal:
    def test_positive_edge(self) -> None:
        sizer = KellyPositionSizer()
        # 55% win rate, 1.5 R:R -> positive Kelly
        kelly = sizer.optimal_kelly(0.55, 1.5)
        assert kelly > 0

    def test_no_edge(self) -> None:
        sizer = KellyPositionSizer()
        # 50% win, 1:1 R:R -> Kelly = 0
        kelly = sizer.optimal_kelly(0.50, 1.0)
        assert kelly == 0.0

    def test_negative_edge(self) -> None:
        sizer = KellyPositionSizer()
        # 40% win, 1:1 R:R -> negative Kelly
        kelly = sizer.optimal_kelly(0.40, 1.0)
        assert kelly < 0

    def test_zero_risk_reward(self) -> None:
        sizer = KellyPositionSizer()
        assert sizer.optimal_kelly(0.55, 0.0) == 0.0


class TestKellyPositionSize:
    def test_applies_fraction(self) -> None:
        sizer = KellyPositionSizer(KellyParams(kelly_fraction=0.25, max_position_pct=0.10))
        full_kelly = sizer.optimal_kelly(0.55, 1.5)
        safe = sizer.position_size_pct(0.55, 1.5)
        assert safe <= full_kelly
        assert safe > 0

    def test_respects_max_cap(self) -> None:
        sizer = KellyPositionSizer(KellyParams(max_position_pct=0.03))
        pct = sizer.position_size_pct(0.80, 3.0)  # Very high edge
        assert pct <= 0.03

    def test_returns_zero_for_negative_edge(self) -> None:
        sizer = KellyPositionSizer()
        assert sizer.position_size_pct(0.30, 1.0) == 0.0

    def test_below_minimum_returns_zero(self) -> None:
        sizer = KellyPositionSizer(KellyParams(
            kelly_fraction=0.01, min_position_pct=0.005,
        ))
        # Very small edge -> below minimum
        pct = sizer.position_size_pct(0.51, 1.01)
        assert pct == 0.0 or pct >= 0.005


class TestKellyPositionValue:
    def test_dollar_value(self) -> None:
        sizer = KellyPositionSizer(KellyParams(max_position_pct=0.03))
        value = sizer.position_value(100_000, 0.55, 1.5)
        assert 0 < value <= 3_000  # max 3% of 100K

    def test_leverage_cap(self) -> None:
        sizer = KellyPositionSizer(KellyParams(max_position_pct=5.0, max_leverage=3.0))
        value = sizer.position_value(100_000, 0.80, 5.0)
        assert value <= 300_000  # 3x leverage cap

    def test_units(self) -> None:
        sizer = KellyPositionSizer(KellyParams(max_position_pct=0.03))
        units = sizer.position_size_units(100_000, 50_000, 0.55, 1.5)
        assert units > 0
        assert units * 50_000 <= 3_000 + 1  # Rounding tolerance


# =======================
# Portfolio Risk Manager
# =======================

class TestPortfolioDrawdown:
    def test_no_drawdown_initially(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        assert rm.drawdown == 0.0

    def test_drawdown_after_loss(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        rm.update_equity(110_000)  # New peak
        rm.update_equity(99_000)   # Drop
        assert abs(rm.drawdown - (11_000 / 110_000)) < 0.001

    def test_peak_updates(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        rm.update_equity(120_000)
        assert rm.peak_equity == 120_000
        rm.update_equity(115_000)
        assert rm.peak_equity == 120_000  # Peak should not decrease


class TestPortfolioDailyLoss:
    def test_no_daily_loss_initially(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        assert rm.daily_loss == 0.0

    def test_daily_loss_tracks(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        rm.new_day()
        rm.update_equity(97_000)
        assert abs(rm.daily_loss - 0.03) < 0.001

    def test_new_day_resets(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        rm.update_equity(95_000)
        rm.new_day()  # Reset daily start
        assert rm.daily_loss == 0.0


class TestPortfolioValidation:
    def test_allows_normal_trade(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        result = rm.can_open_position("BTC/USDT", 50_000, 0.05)
        assert result.allowed

    def test_rejects_on_max_drawdown(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(max_drawdown=0.15),
            initial_equity=100_000,
        )
        rm.update_equity(110_000)
        rm.update_equity(90_000)  # >15% DD from peak
        result = rm.can_open_position("BTC/USDT", 50_000, 0.01)
        assert not result.allowed
        assert "drawdown" in result.reason.lower()

    def test_rejects_on_daily_loss(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(daily_loss_limit=0.02),
            initial_equity=100_000,
        )
        rm.new_day()
        rm.update_equity(97_000)  # 3% loss > 2% limit
        result = rm.can_open_position("BTC/USDT", 50_000, 0.01)
        assert not result.allowed
        assert "daily" in result.reason.lower()

    def test_rejects_on_max_positions(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(max_positions=2),
            initial_equity=100_000,
        )
        rm.add_position(PortfolioPosition("BTC/USDT", "LONG", 50_000, 0.1))
        rm.add_position(PortfolioPosition("ETH/USDT", "LONG", 3_000, 1.0))
        result = rm.can_open_position("SOL/USDT", 100, 10)
        assert not result.allowed
        assert "positions" in result.reason.lower()

    def test_rejects_on_position_size(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(max_position_pct=0.03),
            initial_equity=100_000,
        )
        # 5% position -> exceeds 3% limit
        result = rm.can_open_position("BTC/USDT", 50_000, 0.1)  # 5000 = 5%
        assert not result.allowed
        assert "size" in result.reason.lower()

    def test_rejects_on_leverage(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(max_leverage=2.0, max_position_pct=1.0),
            initial_equity=100_000,
        )
        rm.add_position(PortfolioPosition("BTC/USDT", "LONG", 50_000, 3.0))  # 150K
        # Adding 60K more -> total 210K / 100K = 2.1x > 2.0x
        result = rm.can_open_position("ETH/USDT", 3_000, 20)
        assert not result.allowed
        assert "leverage" in result.reason.lower()


class TestPortfolioSuspension:
    def test_suspended_on_max_dd(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(max_drawdown=0.15),
            initial_equity=100_000,
        )
        rm.update_equity(80_000)
        assert not rm.is_suspended().allowed

    def test_not_suspended_normally(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        assert rm.is_suspended().allowed

    def test_paused_on_daily_loss(self) -> None:
        rm = PortfolioRiskManager(
            limits=RiskLimits(daily_loss_limit=0.02),
            initial_equity=100_000,
        )
        rm.new_day()
        rm.update_equity(97_500)  # 2.5% > 2%
        assert not rm.is_paused().allowed

    def test_risk_summary(self) -> None:
        rm = PortfolioRiskManager(initial_equity=100_000)
        rm.update_equity(95_000)
        summary = rm.risk_summary()
        assert "equity" in summary
        assert "drawdown" in summary
        assert summary["equity"] == 95_000


# =======================
# Dynamic Stops
# =======================

class TestCalculateStops:
    def test_long_stop_below_entry(self) -> None:
        sm = DynamicStopManager()
        levels = sm.calculate_stops(100.0, 5.0, "LONG")
        assert levels.stop_loss < 100.0
        assert levels.take_profit > 100.0

    def test_short_stop_above_entry(self) -> None:
        sm = DynamicStopManager()
        levels = sm.calculate_stops(100.0, 5.0, "SHORT")
        assert levels.stop_loss > 100.0
        assert levels.take_profit < 100.0

    def test_atr_multiplier_applied(self) -> None:
        params = StopParams(stop_loss_mult=2.5, take_profit_mult=4.0)
        sm = DynamicStopManager(params)
        levels = sm.calculate_stops(1000.0, 10.0, "LONG")
        assert abs(levels.stop_loss - (1000 - 25)) < 0.01
        assert abs(levels.take_profit - (1000 + 40)) < 0.01

    def test_atr_value_stored(self) -> None:
        sm = DynamicStopManager()
        levels = sm.calculate_stops(100.0, 7.5, "LONG")
        assert levels.atr_value == 7.5


class TestTrailingStop:
    def test_no_trailing_before_activation(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0))
        # Entry=100, stop=90 (risk=10). Need profit > 20 to activate.
        updated = sm.update_trailing_stop(100.0, 115.0, 90.0, 5.0, "LONG")
        assert updated == 90.0  # Not activated (profit=15 < 2*10=20)

    def test_trailing_activates_on_profit(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0, trailing_atr_mult=1.0))
        # Entry=100, stop=90 (risk=10). Profit=25 > 2*10=20 -> activate
        updated = sm.update_trailing_stop(100.0, 125.0, 90.0, 5.0, "LONG")
        expected = 125.0 - 5.0  # 120.0
        assert abs(updated - expected) < 0.01

    def test_trailing_only_moves_up_long(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0, trailing_atr_mult=1.0))
        # Activated, stop at 120
        updated = sm.update_trailing_stop(100.0, 122.0, 120.0, 5.0, "LONG")
        # new_stop = 122 - 5 = 117 < 120 -> keep 120
        assert updated == 120.0

    def test_trailing_short(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0, trailing_atr_mult=1.0))
        # Short: entry=100, stop=110 (risk=10). Price dropped to 75 -> profit=25
        updated = sm.update_trailing_stop(100.0, 75.0, 110.0, 5.0, "SHORT")
        expected = 75.0 + 5.0  # 80.0
        assert abs(updated - expected) < 0.01

    def test_trailing_short_only_moves_down(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0, trailing_atr_mult=1.0))
        # Short stop already at 80, new would be 83 -> keep 80
        updated = sm.update_trailing_stop(100.0, 78.0, 80.0, 5.0, "SHORT")
        assert updated == 80.0


class TestStopHit:
    def test_long_sl_hit(self) -> None:
        sm = DynamicStopManager()
        result = sm.is_stop_hit(95.0, 110.0, high=100.0, low=94.0, direction="LONG")
        assert result == "stop_loss"

    def test_long_tp_hit(self) -> None:
        sm = DynamicStopManager()
        result = sm.is_stop_hit(95.0, 110.0, high=111.0, low=105.0, direction="LONG")
        assert result == "take_profit"

    def test_long_no_hit(self) -> None:
        sm = DynamicStopManager()
        result = sm.is_stop_hit(95.0, 110.0, high=108.0, low=96.0, direction="LONG")
        assert result is None

    def test_short_sl_hit(self) -> None:
        sm = DynamicStopManager()
        result = sm.is_stop_hit(105.0, 90.0, high=106.0, low=98.0, direction="SHORT")
        assert result == "stop_loss"

    def test_short_tp_hit(self) -> None:
        sm = DynamicStopManager()
        result = sm.is_stop_hit(105.0, 90.0, high=92.0, low=89.0, direction="SHORT")
        assert result == "take_profit"


class TestManagePosition:
    def test_returns_exit_on_stop(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0))
        stop, tp, reason = sm.manage_position(
            entry_price=100.0, current_price=94.0,
            current_stop=95.0, take_profit=115.0,
            current_atr=5.0, high=96.0, low=93.0, direction="LONG",
        )
        assert reason == "stop_loss"

    def test_returns_none_when_open(self) -> None:
        sm = DynamicStopManager()
        stop, tp, reason = sm.manage_position(
            entry_price=100.0, current_price=105.0,
            current_stop=90.0, take_profit=120.0,
            current_atr=5.0, high=106.0, low=104.0, direction="LONG",
        )
        assert reason is None

    def test_trailing_updates_stop(self) -> None:
        sm = DynamicStopManager(StopParams(trailing_activation=2.0, trailing_atr_mult=1.0))
        stop, tp, reason = sm.manage_position(
            entry_price=100.0, current_price=125.0,
            current_stop=90.0, take_profit=140.0,
            current_atr=5.0, high=126.0, low=124.0, direction="LONG",
        )
        assert reason is None
        assert stop == 120.0  # Trailing activated: 125 - 5 = 120
