"""Tests for enhanced walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import BacktestEngine, SignalType
from src.backtester.walk_forward import (
    MonteCarloResult,
    RegimeStats,
    WalkForwardDiagnostics,
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)


def _make_ohlcv(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10)  # floor at 10
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.uniform(100, 10000, n),
    })


def _simple_signal(data: pd.DataFrame, index: int) -> SignalType:
    """Simple MA crossover signal for testing."""
    if index < 50:
        return SignalType.HOLD
    fast = data["close"].iloc[max(0, index - 10) : index + 1].mean()
    slow = data["close"].iloc[max(0, index - 50) : index + 1].mean()
    if fast > slow:
        return SignalType.BUY
    elif fast < slow:
        return SignalType.SELL
    return SignalType.HOLD


class TestWalkForwardValidator:
    def test_generate_windows(self):
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(engine, train_size=500, test_size=200)
        windows = wfv.generate_windows(1500)
        assert len(windows) >= 2
        for w in windows:
            assert w.train_end == w.test_start
            assert w.test_end - w.test_start <= 200

    def test_run_produces_result(self):
        data = _make_ohlcv(3000)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(
            engine, train_size=500, test_size=200,
            monte_carlo_sims=50,
        )
        result = wfv.run(data, _simple_signal, symbol="TEST")

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 2
        assert result.oos_metrics is not None
        assert result.diagnostics is not None

    def test_diagnostics_computed(self):
        data = _make_ohlcv(3000)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(
            engine, train_size=500, test_size=200,
            monte_carlo_sims=50,
        )
        result = wfv.run(data, _simple_signal)
        d = result.diagnostics

        assert d is not None
        assert isinstance(d.avg_is_sharpe, float)
        assert isinstance(d.avg_oos_sharpe, float)
        assert isinstance(d.sharpe_degradation, float)
        assert 0 <= d.pct_profitable_windows <= 1

    def test_regime_detection(self):
        data = _make_ohlcv(3000)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(
            engine, train_size=500, test_size=200,
            monte_carlo_sims=0,
        )
        result = wfv.run(data, _simple_signal)

        assert result.diagnostics is not None
        for rs in result.diagnostics.regime_stats:
            assert rs.regime in ("low", "normal", "high")
            assert rs.volatility > 0

    def test_monte_carlo(self):
        data = _make_ohlcv(3000)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(
            engine, train_size=500, test_size=200,
            monte_carlo_sims=100,
        )
        result = wfv.run(data, _simple_signal)

        mc = result.diagnostics.monte_carlo
        if mc is not None:
            assert mc.n_simulations == 100
            assert 0 <= mc.p_value <= 1
            assert mc.percentile_5 <= mc.percentile_95

    def test_summary_string(self):
        data = _make_ohlcv(3000)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(
            engine, train_size=500, test_size=200,
            monte_carlo_sims=20,
        )
        result = wfv.run(data, _simple_signal)
        summary = result.summary()

        assert "Walk-Forward" in summary
        assert "Window" in summary
        assert "Diagnostics" in summary

    def test_not_enough_data_raises(self):
        data = _make_ohlcv(100)
        engine = BacktestEngine(initial_capital=10000)
        wfv = WalkForwardValidator(engine, train_size=500, test_size=200)

        with pytest.raises(ValueError, match="Not enough data"):
            wfv.run(data, _simple_signal)
