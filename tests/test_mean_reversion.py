"""Tests for Mean Reversion strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Signal
from src.strategies.mean_reversion import (
    MeanReversionParams,
    MeanReversionStrategy,
    backtest_mean_reversion,
)


# --- Fixtures ---

def _make_ohlcv(close: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(close)
    timestamps = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_ = close + np.random.randn(n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(100, 5000, n),
    })


@pytest.fixture
def ranging_data() -> pd.DataFrame:
    """Oscillating price around 1000 — ideal for mean reversion."""
    np.random.seed(42)
    n = 1000
    # Sine wave + noise -> mean reverting
    close = 1000 + 30 * np.sin(np.linspace(0, 20 * np.pi, n)) + np.random.randn(n) * 3
    return _make_ohlcv(close)


@pytest.fixture
def trending_data() -> pd.DataFrame:
    """Strong uptrend — mean reversion should struggle."""
    np.random.seed(99)
    n = 1000
    close = 1000 + np.cumsum(np.random.randn(n) * 2 + 1.5)
    return _make_ohlcv(close)


@pytest.fixture
def volatile_data() -> pd.DataFrame:
    """High volatility regime data."""
    np.random.seed(77)
    n = 1000
    # First 500: low vol, last 500: high vol
    low_vol = np.random.randn(500) * 2
    high_vol = np.random.randn(500) * 20
    returns = np.concatenate([low_vol, high_vol])
    close = 1000 + np.cumsum(returns)
    return _make_ohlcv(close)


# --- Prepare tests ---

class TestPrepare:
    def test_adds_columns(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        prepared = strategy.prepare(ranging_data)
        expected = ["mr_mean", "mr_std", "mr_zscore", "mr_vol", "mr_vol_avg", "mr_vol_ratio"]
        for col in expected:
            assert col in prepared.columns

    def test_zscore_centered(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        prepared = strategy.prepare(ranging_data)
        valid_z = prepared["mr_zscore"].dropna()
        assert abs(valid_z.mean()) < 1.0


# --- Signal generation ---

class TestSignalGeneration:
    def test_hold_during_warmup(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        prepared = strategy.prepare(ranging_data)
        for i in range(10):
            assert strategy.generate_signal(prepared, i) == Signal.HOLD

    def test_generates_buy_signals(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy(MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        ))
        prepared = strategy.prepare(ranging_data)
        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.BUY in signals

    def test_generates_sell_signals(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy(MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        ))
        prepared = strategy.prepare(ranging_data)
        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.SELL in signals

    def test_generates_exit_signals(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy(MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        ))
        prepared = strategy.prepare(ranging_data)
        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.EXIT in signals

    def test_max_hold_forces_exit(self) -> None:
        np.random.seed(42)
        n = 500
        # Price drops and stays low (no reversion)
        close = np.concatenate([
            np.array([1000.0] * 120),
            np.array([900.0] * (n - 120)),
        ])
        data = _make_ohlcv(close)

        params = MeanReversionParams(
            lookback=20, entry_std=1.5, exit_std=0.3,
            max_hold=20, vol_lookback=50, use_vol_filter=False,
        )
        strategy = MeanReversionStrategy(params)
        prepared = strategy.prepare(data)
        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.EXIT in signals

    def test_vol_filter_blocks_entries(self, volatile_data: pd.DataFrame) -> None:
        """In high-vol regime, fewer entries should be generated."""
        params_no_filter = MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        )
        params_with_filter = MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=True, vol_threshold=1.3,
        )

        strat_no = MeanReversionStrategy(params_no_filter)
        strat_yes = MeanReversionStrategy(params_with_filter)

        prepared_no = strat_no.prepare(volatile_data)
        prepared_yes = strat_yes.prepare(volatile_data)

        signals_no = [strat_no.generate_signal(prepared_no, i) for i in range(len(prepared_no))]
        signals_yes = [strat_yes.generate_signal(prepared_yes, i) for i in range(len(prepared_yes))]

        entries_no = sum(1 for s in signals_no if s in (Signal.BUY, Signal.SELL))
        entries_yes = sum(1 for s in signals_yes if s in (Signal.BUY, Signal.SELL))

        # Vol filter should reduce entries
        assert entries_yes <= entries_no

    def test_reset_clears_state(self, ranging_data: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy(MeanReversionParams(use_vol_filter=False))
        prepared = strategy.prepare(ranging_data)
        for i in range(200):
            strategy.generate_signal(prepared, i)
        strategy.reset()
        assert strategy._position_bar is None
        assert strategy._position_side is None


# --- Full backtest ---

class TestBacktestMeanReversion:
    def test_backtest_runs(self, ranging_data: pd.DataFrame) -> None:
        result = backtest_mean_reversion(ranging_data)
        assert result.metrics.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_backtest_on_ranging_produces_trades(self, ranging_data: pd.DataFrame) -> None:
        params = MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        )
        result = backtest_mean_reversion(ranging_data, params=params)
        assert result.metrics.total_trades > 0

    def test_equity_starts_at_capital(self, ranging_data: pd.DataFrame) -> None:
        capital = 50_000.0
        result = backtest_mean_reversion(ranging_data, initial_capital=capital)
        assert abs(result.equity_curve.iloc[0] - capital) < capital * 0.01

    def test_engine_signal_mapping(self, ranging_data: pd.DataFrame) -> None:
        from src.backtester.engine import SignalType as ST
        strategy = MeanReversionStrategy(MeanReversionParams(use_vol_filter=False))
        prepared = strategy.prepare(ranging_data)
        for i in range(len(prepared)):
            sig = strategy.generate_engine_signal(prepared, i)
            assert sig in (ST.BUY, ST.SELL, ST.EXIT, ST.HOLD)
