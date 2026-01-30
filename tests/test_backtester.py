"""Tests for backtesting engine, metrics, and walk-forward."""

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestEngine, SignalType, Side
from src.backtester.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    expectancy,
    compute_metrics,
)
from src.backtester.walk_forward import WalkForwardValidator


# --- Fixtures ---

@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Synthetic uptrending OHLCV data."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Uptrend with noise
    close = 1000 + np.cumsum(np.random.randn(n) * 5 + 0.5)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_ = close + np.random.randn(n) * 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(100, 5000, n),
    })


@pytest.fixture
def flat_data() -> pd.DataFrame:
    """Synthetic flat/ranging OHLCV data."""
    np.random.seed(99)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1000 + np.random.randn(n) * 10  # Mean-reverting around 1000
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_ = close + np.random.randn(n) * 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(100, 5000, n),
    })


def always_buy_signal(df: pd.DataFrame, i: int) -> SignalType:
    """Signal that buys at bar 10 and exits at bar 100."""
    if i == 10:
        return SignalType.BUY
    if i == 100:
        return SignalType.EXIT
    return SignalType.HOLD


def alternating_signal(df: pd.DataFrame, i: int) -> SignalType:
    """Buy every 50 bars, exit 25 bars later."""
    cycle = i % 50
    if cycle == 0 and i > 0:
        return SignalType.BUY
    if cycle == 25:
        return SignalType.EXIT
    return SignalType.HOLD


def sma_crossover_signal(df: pd.DataFrame, i: int) -> SignalType:
    """Simple SMA crossover: buy when SMA5 > SMA20, sell when SMA5 < SMA20."""
    if i < 20:
        return SignalType.HOLD
    close = df["close"]
    sma5 = close.iloc[max(0, i - 4) : i + 1].mean()
    sma20 = close.iloc[max(0, i - 19) : i + 1].mean()
    sma5_prev = close.iloc[max(0, i - 5) : i].mean()
    sma20_prev = close.iloc[max(0, i - 20) : i].mean()

    if sma5 > sma20 and sma5_prev <= sma20_prev:
        return SignalType.BUY
    if sma5 < sma20 and sma5_prev >= sma20_prev:
        return SignalType.SELL
    return SignalType.HOLD


# --- Metrics unit tests ---

class TestMetrics:
    def test_sharpe_positive_returns(self) -> None:
        returns = pd.Series(np.random.randn(1000) * 0.01 + 0.001)
        result = sharpe_ratio(returns, periods_per_year=8760)
        assert result > 0

    def test_sharpe_zero_std(self) -> None:
        returns = pd.Series([0.01] * 100)
        result = sharpe_ratio(returns)
        # Constant returns -> std = 0 -> should return 0
        assert result == 0.0

    def test_sortino_ignores_upside(self) -> None:
        # All positive returns -> no downside -> 0
        returns = pd.Series([0.01] * 100)
        result = sortino_ratio(returns)
        assert result == 0.0

    def test_sortino_with_downside(self) -> None:
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01 + 0.0005)
        result = sortino_ratio(returns, periods_per_year=8760)
        assert isinstance(result, float)

    def test_max_drawdown_known(self) -> None:
        # Equity: 100, 110, 90, 95 -> DD from 110 to 90 = 18.18%
        equity = pd.Series([100.0, 110.0, 90.0, 95.0])
        dd, dur = max_drawdown(equity)
        assert abs(dd - 0.1818) < 0.01
        assert dur > 0

    def test_max_drawdown_no_dd(self) -> None:
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        dd, dur = max_drawdown(equity)
        assert dd == 0.0

    def test_win_rate_known(self) -> None:
        trades = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])
        assert win_rate(trades) == 0.6

    def test_profit_factor_known(self) -> None:
        trades = pd.Series([0.10, -0.05, 0.08, -0.03])
        pf = profit_factor(trades)
        assert abs(pf - (0.18 / 0.08)) < 0.01

    def test_profit_factor_no_losses(self) -> None:
        trades = pd.Series([0.05, 0.10])
        assert profit_factor(trades) == float("inf")

    def test_expectancy_known(self) -> None:
        trades = pd.Series([0.10, -0.05, 0.10, -0.05])
        exp = expectancy(trades)
        # wr=0.5, avg_w=0.10, avg_l=-0.05 -> 0.5*0.10 + 0.5*(-0.05) = 0.025
        assert abs(exp - 0.025) < 0.001

    def test_compute_metrics_returns_dataclass(self) -> None:
        equity = pd.Series([100.0, 101.0, 102.0, 101.5, 103.0])
        trades = pd.Series([0.02, -0.005, 0.015])
        m = compute_metrics(equity, trades)
        assert m.total_trades == 3
        assert m.total_return > 0
        assert isinstance(m.summary(), str)


# --- Engine tests ---

class TestBacktestEngine:
    def test_single_trade_produces_result(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(trending_up_data, always_buy_signal)
        assert len(result.trades) == 1
        assert result.metrics.total_trades == 1
        assert len(result.equity_curve) == len(trending_up_data)

    def test_multiple_trades(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(trending_up_data, alternating_signal)
        assert result.metrics.total_trades >= 2

    def test_commission_reduces_pnl(self, trending_up_data: pd.DataFrame) -> None:
        engine_no_cost = BacktestEngine(initial_capital=100_000, commission_rate=0.0, slippage_rate=0.0)
        engine_with_cost = BacktestEngine(initial_capital=100_000, commission_rate=0.005, slippage_rate=0.005)

        r1 = engine_no_cost.run(trending_up_data, alternating_signal)
        r2 = engine_with_cost.run(trending_up_data, alternating_signal)

        # Higher costs should result in lower final equity
        assert r2.equity_curve.iloc[-1] < r1.equity_curve.iloc[-1]

    def test_equity_curve_length(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(trending_up_data, alternating_signal)
        assert len(result.equity_curve) == len(trending_up_data)

    def test_sma_crossover_on_trend(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(trending_up_data, sma_crossover_signal)
        # On uptrend, SMA crossover should produce trades
        assert result.metrics.total_trades >= 1

    def test_stop_loss_triggers(self) -> None:
        """Verify stop loss closes position when price drops."""
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        # Price drops sharply after bar 10
        close = np.array([100.0] * 11 + [80.0] * (n - 11))
        high = close + 1
        low = close - 1
        data = pd.DataFrame({
            "timestamp": timestamps, "open": close, "high": high,
            "low": low, "close": close, "volume": [1000.0] * n,
        })

        def buy_at_5(df, i):
            return SignalType.BUY if i == 5 else SignalType.HOLD

        engine = BacktestEngine(initial_capital=100_000, commission_rate=0, slippage_rate=0)
        result = engine.run(data, buy_at_5, stop_loss_pct=0.10)
        # Should have been stopped out
        assert len(result.trades) >= 1
        assert result.trades[0].pnl < 0

    def test_signals_recorded(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(trending_up_data, always_buy_signal)
        assert len(result.signals) == len(trending_up_data)
        assert "signal" in result.signals.columns


# --- Walk-forward tests ---

class TestWalkForward:
    def test_generate_windows(self) -> None:
        engine = BacktestEngine()
        wf = WalkForwardValidator(engine, train_size=200, test_size=100, step_size=100)
        windows = wf.generate_windows(600)
        assert len(windows) == 4

    def test_not_enough_data_raises(self) -> None:
        engine = BacktestEngine()
        wf = WalkForwardValidator(engine, train_size=200, test_size=100)
        with pytest.raises(ValueError, match="Not enough data"):
            data = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
                "open": [100] * 50, "high": [101] * 50, "low": [99] * 50,
                "close": [100] * 50, "volume": [1000] * 50,
            })
            wf.run(data, always_buy_signal)

    def test_walk_forward_produces_result(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine(initial_capital=100_000)
        wf = WalkForwardValidator(engine, train_size=150, test_size=100, step_size=100)
        result = wf.run(trending_up_data, sma_crossover_signal)

        assert len(result.windows) >= 1
        assert len(result.oos_equity) > 0
        assert isinstance(result.oos_metrics.sharpe_ratio, float)
        assert isinstance(result.summary(), str)

    def test_each_window_has_results(self, trending_up_data: pd.DataFrame) -> None:
        engine = BacktestEngine(initial_capital=100_000)
        wf = WalkForwardValidator(engine, train_size=150, test_size=100, step_size=100)
        result = wf.run(trending_up_data, sma_crossover_signal)

        for w in result.windows:
            assert w.train_result is not None
            assert w.test_result is not None
