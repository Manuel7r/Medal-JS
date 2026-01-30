"""Tests for data pipeline, validation, and features."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline
from src.features import technical, statistical


# --- Fixtures ---

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(100, 10000, n),
        "symbol": "BTC/USDT",
    })


@pytest.fixture
def dirty_ohlcv(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data with intentional quality issues."""
    df = sample_ohlcv.copy()
    # Add null row
    df.loc[5, "close"] = np.nan
    # Add high < low row
    df.loc[10, "high"] = df.loc[10, "low"] - 1
    # Add negative volume
    df.loc[15, "volume"] = -100
    # Add duplicate timestamp
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


# --- Pipeline validation tests ---

class TestValidation:
    def test_validate_removes_nulls(self, dirty_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        result = pipeline.validate(dirty_ohlcv)
        assert result["close"].isna().sum() == 0

    def test_validate_removes_bad_highs(self, dirty_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        result = pipeline.validate(dirty_ohlcv)
        assert (result["high"] >= result["low"]).all()

    def test_validate_removes_negative_volume(self, dirty_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        result = pipeline.validate(dirty_ohlcv)
        assert (result["volume"] >= 0).all()

    def test_validate_removes_duplicates(self, dirty_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        result = pipeline.validate(dirty_ohlcv)
        assert result.duplicated(subset=["timestamp", "symbol"]).sum() == 0

    def test_validate_empty_df(self) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
        result = pipeline.validate(empty)
        assert result.empty


class TestGapDetection:
    def test_no_gaps_in_clean_data(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        gaps = pipeline.detect_gaps(sample_ohlcv, "1h")
        assert len(gaps) == 0

    def test_detects_gap(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = DataPipeline(source=None, storage=None)  # type: ignore[arg-type]
        # Remove rows 50-55 to create a gap
        df = sample_ohlcv.drop(index=range(50, 56)).reset_index(drop=True)
        gaps = pipeline.detect_gaps(df, "1h")
        assert len(gaps) >= 1


# --- Technical indicator tests ---

class TestTechnicalIndicators:
    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        result = technical.rsi(sample_ohlcv["close"])
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = technical.macd(sample_ohlcv["close"])
        assert list(result.columns) == ["macd", "signal", "histogram"]
        assert len(result) == len(sample_ohlcv)

    def test_bollinger_bands_order(self, sample_ohlcv: pd.DataFrame) -> None:
        result = technical.bollinger_bands(sample_ohlcv["close"])
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        result = technical.atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        valid = result.dropna()
        assert (valid > 0).all()

    def test_compute_all_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = technical.compute_all(sample_ohlcv)
        expected = ["rsi_14", "macd", "bb_upper", "atr_14", "volume_ma_ratio"]
        for col in expected:
            assert col in result.columns


# --- Statistical feature tests ---

class TestStatisticalFeatures:
    def test_z_score_mean_near_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        result = statistical.z_score(sample_ohlcv["close"], 20)
        valid = result.dropna()
        assert abs(valid.mean()) < 1.0  # Should be centered roughly around 0

    def test_cointegration_returns_dict(self) -> None:
        np.random.seed(42)
        n = 500
        x = np.cumsum(np.random.randn(n)) + 100
        y = x * 0.5 + np.random.randn(n) * 2 + 50  # Cointegrated
        result = statistical.cointegration_test(pd.Series(x), pd.Series(y))
        assert "p_value" in result
        assert "is_cointegrated" in result
        assert isinstance(result["is_cointegrated"], bool)

    def test_hedge_ratio_positive_for_correlated(self) -> None:
        np.random.seed(42)
        n = 500
        x = np.cumsum(np.random.randn(n)) + 100
        y = x * 2 + np.random.randn(n) * 5
        beta = statistical.hedge_ratio(pd.Series(y), pd.Series(x))
        assert beta > 0

    def test_spread_z_score_shape(self) -> None:
        np.random.seed(42)
        n = 300
        a = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        b = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        result = statistical.spread_z_score(a, b, lookback=20)
        assert len(result) == n

    def test_compute_all_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = statistical.compute_all(sample_ohlcv)
        expected = ["z_score_20", "returns_skew", "returns_1d", "returns_lag_1"]
        for col in expected:
            assert col in result.columns
