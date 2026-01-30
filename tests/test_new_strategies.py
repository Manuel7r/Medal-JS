"""Tests for new strategies: Momentum, Microstructure, Triangular Arb."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Signal
from src.strategies.momentum import MomentumStrategy, MomentumParams, backtest_momentum
from src.strategies.microstructure import MicrostructureStrategy, MicrostructureParams, backtest_microstructure
from src.strategies.triangular_arb import TriangularArbStrategy, TriangularArbParams


def _make_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.uniform(100, 10000, n),
    })


class TestMomentum:
    def test_prepare_adds_features(self):
        data = _make_ohlcv()
        strategy = MomentumStrategy()
        prepared = strategy.prepare(data)

        assert "mom_fast_ema" in prepared.columns
        assert "mom_slow_ema" in prepared.columns
        assert "mom_adx" in prepared.columns
        assert "mom_rsi" in prepared.columns

    def test_generates_signals(self):
        data = _make_ohlcv()
        strategy = MomentumStrategy()
        prepared = strategy.prepare(data)

        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        signal_types = set(signals)
        assert Signal.HOLD in signal_types

    def test_backtest_runs(self):
        data = _make_ohlcv(2000)
        result = backtest_momentum(data, initial_capital=10000)
        assert result.metrics.total_trades >= 0
        assert result.metrics.sharpe_ratio is not None

    def test_reset(self):
        strategy = MomentumStrategy()
        data = _make_ohlcv()
        prepared = strategy.prepare(data)
        strategy.generate_signal(prepared, 100)
        strategy.reset()
        assert strategy._position_bar is None

    def test_params(self):
        params = MomentumParams(fast_period=8, slow_period=24)
        strategy = MomentumStrategy(params)
        p = strategy.get_params()
        assert p.name == "Momentum"


class TestMicrostructure:
    def test_prepare_adds_features(self):
        data = _make_ohlcv()
        strategy = MicrostructureStrategy()
        prepared = strategy.prepare(data)

        assert "micro_vwap" in prepared.columns
        assert "micro_vwap_zscore" in prepared.columns
        assert "micro_vol_ratio" in prepared.columns
        assert "micro_pressure_ma" in prepared.columns

    def test_generates_signals(self):
        data = _make_ohlcv()
        strategy = MicrostructureStrategy()
        prepared = strategy.prepare(data)

        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.HOLD in set(signals)

    def test_backtest_runs(self):
        data = _make_ohlcv(2000)
        result = backtest_microstructure(data, initial_capital=10000)
        assert result.metrics is not None

    def test_reset(self):
        strategy = MicrostructureStrategy()
        strategy._position_bar = 5
        strategy.reset()
        assert strategy._position_bar is None


class TestTriangularArb:
    def _make_triangular_data(self, n: int = 500):
        rng = np.random.default_rng(42)
        btc = 40000 + np.cumsum(rng.normal(0, 100, n))
        eth = 2000 + np.cumsum(rng.normal(0, 20, n))
        eth_btc = eth / btc + rng.normal(0, 0.0005, n)  # slight noise

        return pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
            "open": btc,
            "high": btc * 1.001,
            "low": btc * 0.999,
            "close": btc,
            "volume": rng.uniform(100, 10000, n),
            "close_a": btc,
            "close_b": eth,
            "close_c": eth_btc,
        })

    def test_prepare_adds_features(self):
        data = self._make_triangular_data()
        strategy = TriangularArbStrategy()
        prepared = strategy.prepare(data)

        assert "tri_implied" in prepared.columns
        assert "tri_spread" in prepared.columns
        assert "tri_zscore" in prepared.columns

    def test_generates_signals(self):
        data = self._make_triangular_data()
        strategy = TriangularArbStrategy()
        prepared = strategy.prepare(data)

        signals = [strategy.generate_signal(prepared, i) for i in range(len(prepared))]
        assert Signal.HOLD in set(signals)

    def test_reset(self):
        strategy = TriangularArbStrategy()
        strategy._position_bar = 10
        strategy.reset()
        assert strategy._position_bar is None
