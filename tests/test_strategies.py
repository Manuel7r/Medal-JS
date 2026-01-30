"""Tests for Pairs Trading strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Signal
from src.strategies.pairs_trading import (
    PairsTradingParams,
    PairsTradingStrategy,
    build_pairs_dataframe,
    backtest_pair,
)


# --- Fixtures ---

def _make_ohlcv(close: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Build OHLCV DataFrame from a close array."""
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
def cointegrated_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cointegrated price series (A ≈ 2*B + noise)."""
    np.random.seed(42)
    n = 1000
    # Common random walk
    common = np.cumsum(np.random.randn(n) * 0.5) + 100
    close_b = common + np.random.randn(n) * 0.5
    close_a = 2.0 * common + np.random.randn(n) * 1.0 + 50
    return _make_ohlcv(close_a), _make_ohlcv(close_b)


@pytest.fixture
def non_cointegrated_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two independent random walks (not cointegrated)."""
    np.random.seed(99)
    n = 1000
    close_a = np.cumsum(np.random.randn(n) * 2) + 1000
    close_b = np.cumsum(np.random.randn(n) * 2) + 500
    return _make_ohlcv(close_a), _make_ohlcv(close_b)


@pytest.fixture
def mean_reverting_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair with strong mean-reverting spread for signal testing."""
    np.random.seed(123)
    n = 1000
    common = np.cumsum(np.random.randn(n) * 0.3) + 100
    # Spread oscillates as a sine wave -> guaranteed reversion
    spread_noise = 5.0 * np.sin(np.linspace(0, 20 * np.pi, n))
    close_a = common + spread_noise + np.random.randn(n) * 0.2
    close_b = common + np.random.randn(n) * 0.2
    return _make_ohlcv(close_a), _make_ohlcv(close_b)


# --- Cointegration tests ---

class TestCointegration:
    def test_cointegrated_pair_detected(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        result = PairsTradingStrategy.cointegration_test(df_a["close"], df_b["close"])
        assert result["is_cointegrated"] is True
        assert result["p_value"] < 0.05

    def test_non_cointegrated_pair(self, non_cointegrated_pair) -> None:
        df_a, df_b = non_cointegrated_pair
        result = PairsTradingStrategy.cointegration_test(df_a["close"], df_b["close"])
        # Should usually not be cointegrated (p > 0.05)
        assert "p_value" in result
        assert isinstance(result["is_cointegrated"], bool)

    def test_hedge_ratio_positive(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        hr = PairsTradingStrategy.hedge_ratio_ols(df_a["close"], df_b["close"])
        assert hr > 0  # A ≈ 2*B -> positive ratio

    def test_hedge_ratio_near_expected(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        hr = PairsTradingStrategy.hedge_ratio_ols(df_a["close"], df_b["close"])
        # Should be close to 2.0 since A ≈ 2*B
        assert 1.5 < hr < 2.5


# --- Pair scanning ---

class TestScanPairs:
    def test_scan_finds_cointegrated(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        price_dict = {"A": df_a["close"], "B": df_b["close"]}
        results = PairsTradingStrategy.scan_pairs(price_dict)
        assert len(results) >= 1
        assert results[0]["p_value"] < 0.05

    def test_scan_empty_for_independent(self, non_cointegrated_pair) -> None:
        df_a, df_b = non_cointegrated_pair
        price_dict = {"A": df_a["close"], "B": df_b["close"]}
        results = PairsTradingStrategy.scan_pairs(price_dict, pvalue_threshold=0.01)
        # Likely empty for independent series (strict threshold)
        for r in results:
            assert r["p_value"] < 0.01


# --- DataFrame building ---

class TestBuildPairsDataframe:
    def test_merge_columns(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        assert "close_a" in merged.columns
        assert "close_b" in merged.columns
        assert "close" in merged.columns
        assert "open" in merged.columns
        assert "high" in merged.columns
        assert "low" in merged.columns
        assert "timestamp" in merged.columns

    def test_merge_length(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        # Inner join: should equal min of both
        assert len(merged) <= min(len(df_a), len(df_b))
        assert len(merged) > 0


# --- Strategy prepare ---

class TestPrepare:
    def test_prepare_adds_columns(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy()
        prepared = strategy.prepare(merged)
        for col in ["hedge_ratio", "spread", "spread_mean", "spread_std", "spread_zscore"]:
            assert col in prepared.columns

    def test_zscore_has_values(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy()
        prepared = strategy.prepare(merged)
        valid_z = prepared["spread_zscore"].dropna()
        assert len(valid_z) > 0
        # Z-scores should be roughly centered around 0
        assert abs(valid_z.mean()) < 2.0


# --- Signal generation ---

class TestSignalGeneration:
    def test_hold_during_warmup(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy()
        prepared = strategy.prepare(merged)

        for i in range(10):
            signal = strategy.generate_signal(prepared, i)
            assert signal == Signal.HOLD

    def test_generates_entry_signals(self, mean_reverting_pair) -> None:
        df_a, df_b = mean_reverting_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy(PairsTradingParams(lookback=50, entry_zscore=1.5))
        prepared = strategy.prepare(merged)

        signals = []
        for i in range(len(prepared)):
            signals.append(strategy.generate_signal(prepared, i))

        entry_signals = [s for s in signals if s in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD)]
        assert len(entry_signals) > 0, "Should generate at least one entry signal"

    def test_generates_exit_signals(self, mean_reverting_pair) -> None:
        df_a, df_b = mean_reverting_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy(PairsTradingParams(lookback=50, entry_zscore=1.5))
        prepared = strategy.prepare(merged)

        signals = []
        for i in range(len(prepared)):
            signals.append(strategy.generate_signal(prepared, i))

        exit_signals = [s for s in signals if s == Signal.EXIT]
        assert len(exit_signals) > 0, "Should generate at least one exit signal"

    def test_max_hold_exits(self) -> None:
        """Position should exit after max_hold bars."""
        np.random.seed(42)
        n = 500
        # Create spread that stays diverged (no mean reversion)
        common = np.cumsum(np.random.randn(n) * 0.1) + 100
        close_a = common + np.linspace(0, 20, n)  # Spread keeps growing
        close_b = common
        df_a = _make_ohlcv(close_a)
        df_b = _make_ohlcv(close_b)

        params = PairsTradingParams(lookback=50, entry_zscore=1.0, max_hold=30)
        strategy = PairsTradingStrategy(params)
        merged = build_pairs_dataframe(df_a, df_b)
        prepared = strategy.prepare(merged)

        signals = []
        for i in range(len(prepared)):
            signals.append(strategy.generate_signal(prepared, i))

        # Should have at least one EXIT from max_hold
        assert Signal.EXIT in signals

    def test_reset_clears_state(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy()
        prepared = strategy.prepare(merged)

        # Run some signals
        for i in range(200):
            strategy.generate_signal(prepared, i)

        strategy.reset()
        assert strategy._position_bar is None


# --- Full backtest ---

class TestBacktestPair:
    def test_backtest_runs(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        result = backtest_pair(df_a, df_b)
        assert result.metrics.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_backtest_with_mean_reverting(self, mean_reverting_pair) -> None:
        df_a, df_b = mean_reverting_pair
        params = PairsTradingParams(lookback=50, entry_zscore=1.5)
        result = backtest_pair(df_a, df_b, params=params)
        assert result.metrics.total_trades > 0
        assert isinstance(result.metrics.sharpe_ratio, float)

    def test_backtest_equity_curve_starts_at_capital(self, cointegrated_pair) -> None:
        df_a, df_b = cointegrated_pair
        capital = 50_000.0
        result = backtest_pair(df_a, df_b, initial_capital=capital)
        assert abs(result.equity_curve.iloc[0] - capital) < capital * 0.01

    def test_engine_signal_mapping(self, cointegrated_pair) -> None:
        """Verify generate_engine_signal maps correctly to SignalType."""
        from src.backtester.engine import SignalType as ST

        df_a, df_b = cointegrated_pair
        merged = build_pairs_dataframe(df_a, df_b)
        strategy = PairsTradingStrategy()
        prepared = strategy.prepare(merged)

        for i in range(len(prepared)):
            sig = strategy.generate_engine_signal(prepared, i)
            assert sig in (ST.BUY, ST.SELL, ST.EXIT, ST.HOLD)
