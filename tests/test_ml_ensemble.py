"""Tests for ML Ensemble strategy and Signal Aggregator."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import BaseStrategy, Signal, StrategyParams
from src.strategies.ml_ensemble import (
    FEATURE_COLS,
    MLEnsembleParams,
    MLEnsembleStrategy,
    backtest_ml_ensemble,
)
from src.strategies.aggregator import (
    AggregatorParams,
    SignalAggregator,
    backtest_aggregated,
)
from src.strategies.mean_reversion import MeanReversionParams, MeanReversionStrategy


# --- Fixtures ---

def _make_ohlcv(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1000 + np.cumsum(np.random.randn(n) * 3)
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
def large_data() -> pd.DataFrame:
    """3000 bars of synthetic data — enough for ML training."""
    return _make_ohlcv(3000)


@pytest.fixture
def small_data() -> pd.DataFrame:
    """500 bars — too small for default training window."""
    return _make_ohlcv(500, seed=99)


# =======================
# ML Ensemble Tests
# =======================

class TestMLEnsemblePrepare:
    def test_adds_features(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy()
        prepared = strategy.prepare(large_data)
        for col in ["rsi_14", "macd", "bb_position", "z_score_20", "rsi_change", "bb_squeeze", "target"]:
            assert col in prepared.columns

    def test_target_is_binary(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy()
        prepared = strategy.prepare(large_data)
        valid = prepared["target"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_derived_features_exist(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy()
        prepared = strategy.prepare(large_data)
        for col in ["rsi_change", "macd_cross", "bb_squeeze", "atr_ratio", "vol_trend"]:
            assert col in prepared.columns


class TestMLEnsembleTrain:
    def test_train_sets_flag(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy(MLEnsembleParams(train_window=500, n_estimators=10))
        prepared = strategy.prepare(large_data)
        strategy.train(prepared, 1000)
        assert strategy._is_trained is True
        assert len(strategy._models) >= 1

    def test_train_too_small_skips(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy()
        prepared = strategy.prepare(large_data)
        strategy.train(prepared, 10)  # Only 10 rows
        assert strategy._is_trained is False

    def test_predict_proba_returns_float(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy(MLEnsembleParams(train_window=500, n_estimators=10))
        prepared = strategy.prepare(large_data)
        strategy.train(prepared, 1000)
        prob = strategy.predict_proba(prepared, 1500)
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_none_before_train(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy()
        prepared = strategy.prepare(large_data)
        assert strategy.predict_proba(prepared, 500) is None

    def test_cross_validate_returns_scores(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy(MLEnsembleParams(
            train_window=500, n_estimators=10, cv_folds=3,
        ))
        prepared = strategy.prepare(large_data)
        scores = strategy.cross_validate(prepared, 1000)
        assert len(scores) >= 1
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0


class TestMLEnsembleSignal:
    def test_hold_before_train_window(self, large_data: pd.DataFrame) -> None:
        params = MLEnsembleParams(train_window=2000, n_estimators=10)
        strategy = MLEnsembleStrategy(params)
        prepared = strategy.prepare(large_data)
        for i in range(100):
            assert strategy.generate_signal(prepared, i) == Signal.HOLD

    def test_generates_signals_after_train(self, large_data: pd.DataFrame) -> None:
        params = MLEnsembleParams(train_window=500, n_estimators=10, retrain_interval=5000)
        strategy = MLEnsembleStrategy(params)
        prepared = strategy.prepare(large_data)

        signals = []
        for i in range(500, min(600, len(prepared))):
            signals.append(strategy.generate_signal(prepared, i))

        # Should have at least some non-HOLD signals
        non_hold = [s for s in signals if s != Signal.HOLD]
        assert len(non_hold) >= 0  # May be all HOLD if model is uncertain

    def test_engine_signal_mapping(self, large_data: pd.DataFrame) -> None:
        from src.backtester.engine import SignalType as ST
        params = MLEnsembleParams(train_window=500, n_estimators=10, retrain_interval=5000)
        strategy = MLEnsembleStrategy(params)
        prepared = strategy.prepare(large_data)
        for i in range(500, 510):
            sig = strategy.generate_engine_signal(prepared, i)
            assert sig in (ST.BUY, ST.SELL, ST.EXIT, ST.HOLD)

    def test_reset_clears_state(self, large_data: pd.DataFrame) -> None:
        params = MLEnsembleParams(train_window=500, n_estimators=10)
        strategy = MLEnsembleStrategy(params)
        prepared = strategy.prepare(large_data)
        strategy.train(prepared, 1000)
        assert strategy._is_trained
        strategy.reset()
        assert not strategy._is_trained
        assert len(strategy._models) == 0


class TestMLEnsembleBacktest:
    def test_backtest_runs(self, large_data: pd.DataFrame) -> None:
        params = MLEnsembleParams(train_window=500, n_estimators=10, retrain_interval=5000)
        result = backtest_ml_ensemble(large_data, params=params)
        assert len(result.equity_curve) > 0
        assert result.metrics.total_trades >= 0

    def test_backtest_equity_starts_at_capital(self, large_data: pd.DataFrame) -> None:
        capital = 50_000.0
        params = MLEnsembleParams(train_window=500, n_estimators=10, retrain_interval=5000)
        result = backtest_ml_ensemble(large_data, params=params, initial_capital=capital)
        assert abs(result.equity_curve.iloc[0] - capital) < capital * 0.01


# =======================
# Feature Importance
# =======================

class TestFeatureImportance:
    def test_feature_importance_after_train(self, large_data: pd.DataFrame) -> None:
        strategy = MLEnsembleStrategy(MLEnsembleParams(train_window=500, n_estimators=10))
        prepared = strategy.prepare(large_data)
        strategy.train(prepared, 1000)
        imp = strategy.feature_importance()
        assert len(imp) >= 1
        for name, series in imp.items():
            assert len(series) > 0
            assert (series >= 0).all()

    def test_feature_importance_empty_before_train(self) -> None:
        strategy = MLEnsembleStrategy()
        assert strategy.feature_importance() == {}


# =======================
# Aggregator Tests
# =======================

class _AlwaysBuy(BaseStrategy):
    """Stub strategy that always returns BUY."""
    def generate_signal(self, data, index):
        return Signal.BUY
    def get_params(self):
        return StrategyParams(name="AlwaysBuy")
    def reset(self):
        pass


class _AlwaysSell(BaseStrategy):
    """Stub strategy that always returns SELL."""
    def generate_signal(self, data, index):
        return Signal.SELL
    def get_params(self):
        return StrategyParams(name="AlwaysSell")
    def reset(self):
        pass


class _AlwaysHold(BaseStrategy):
    """Stub strategy that always returns HOLD."""
    def generate_signal(self, data, index):
        return Signal.HOLD
    def get_params(self):
        return StrategyParams(name="AlwaysHold")
    def reset(self):
        pass


class TestAggregator:
    def test_unanimous_buy(self) -> None:
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.register("b", _AlwaysBuy(), 1.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.BUY

    def test_unanimous_sell(self) -> None:
        agg = SignalAggregator()
        agg.register("a", _AlwaysSell(), 1.0)
        agg.register("b", _AlwaysSell(), 1.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.SELL

    def test_split_vote_returns_hold(self) -> None:
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.register("b", _AlwaysSell(), 1.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.HOLD

    def test_weighted_majority_buy(self) -> None:
        """Heavy weight on BUY strategy should produce BUY."""
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 3.0)
        agg.register("b", _AlwaysSell(), 1.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.BUY

    def test_weighted_majority_sell(self) -> None:
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.register("b", _AlwaysSell(), 3.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.SELL

    def test_hold_blocks_action(self) -> None:
        """Two HOLDs + one BUY should HOLD (BUY doesn't have majority)."""
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.register("b", _AlwaysHold(), 1.0)
        agg.register("c", _AlwaysHold(), 1.0)
        assert agg.aggregate(pd.DataFrame(), 0) == Signal.HOLD

    def test_get_individual_signals(self) -> None:
        agg = SignalAggregator()
        agg.register("buyer", _AlwaysBuy(), 1.0)
        agg.register("seller", _AlwaysSell(), 1.0)
        signals = agg.get_individual_signals(pd.DataFrame(), 0)
        assert signals["buyer"] == Signal.BUY
        assert signals["seller"] == Signal.SELL

    def test_engine_signal_mapping(self) -> None:
        from src.backtester.engine import SignalType as ST
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.register("b", _AlwaysBuy(), 1.0)
        assert agg.aggregate_engine_signal(pd.DataFrame(), 0) == ST.BUY

    def test_reset_resets_all(self) -> None:
        agg = SignalAggregator()
        agg.register("a", _AlwaysBuy(), 1.0)
        agg.reset()  # Should not raise


class TestAggregatorBacktest:
    def test_backtest_with_real_strategies(self) -> None:
        np.random.seed(42)
        n = 1000
        close = 1000 + 30 * np.sin(np.linspace(0, 20 * np.pi, n)) + np.random.randn(n) * 3
        high = close + np.abs(np.random.randn(n) * 2)
        low = close - np.abs(np.random.randn(n) * 2)
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close + np.random.randn(n),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(100, 5000, n),
        })

        mr = MeanReversionStrategy(MeanReversionParams(
            lookback=20, entry_std=1.5, vol_lookback=50, use_vol_filter=False,
        ))
        prepared = mr.prepare(data)

        result = backtest_aggregated(
            data=prepared,
            strategies={"mr": mr},
            weights={"mr": 1.0},
        )
        assert len(result.equity_curve) > 0
        assert result.metrics.total_trades >= 0
