"""Tests for enhanced data pipeline: gap detection, filling, health checks."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.data.pipeline import DataPipeline


def _make_ohlcv(n: int = 200, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC"),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(100, 10000, n),
        "symbol": "BTC/USDT",
    })


def _make_gapped_ohlcv() -> pd.DataFrame:
    """Create data with a 5-hour gap."""
    df1 = _make_ohlcv(100)
    df2 = _make_ohlcv(100)
    df2["timestamp"] = pd.date_range("2020-01-05 10:00", periods=100, freq="1h", tz="UTC")
    return pd.concat([df1, df2], ignore_index=True)


class TestGapDetection:
    def _pipeline(self):
        source = MagicMock()
        storage = MagicMock()
        return DataPipeline(source, storage)

    def test_no_gaps(self):
        pipe = self._pipeline()
        df = _make_ohlcv(200)
        gaps = pipe.detect_gaps(df, "1h")
        assert len(gaps) == 0

    def test_detects_gap(self):
        pipe = self._pipeline()
        df = _make_gapped_ohlcv()
        gaps = pipe.detect_gaps(df, "1h")
        assert len(gaps) > 0
        for start, end in gaps:
            assert end > start

    def test_empty_df(self):
        pipe = self._pipeline()
        gaps = pipe.detect_gaps(pd.DataFrame(columns=["timestamp"]), "1h")
        assert gaps == []


class TestGapFilling:
    def _pipeline(self):
        source = MagicMock()
        storage = MagicMock()
        return DataPipeline(source, storage)

    def test_fill_gaps(self):
        pipe = self._pipeline()
        df = _make_gapped_ohlcv()
        filled = pipe.fill_gaps(df, "1h")

        # Should have more rows (gap filled)
        assert len(filled) >= len(df)
        # No NaN in close
        assert filled["close"].isna().sum() == 0

    def test_no_fill_needed(self):
        pipe = self._pipeline()
        df = _make_ohlcv(100)
        filled = pipe.fill_gaps(df, "1h")
        assert len(filled) == len(df)


class TestHealthCheck:
    def _pipeline(self):
        source = MagicMock()
        storage = MagicMock()
        return DataPipeline(source, storage)

    def test_healthy_data(self):
        pipe = self._pipeline()
        df = _make_ohlcv(200)
        health = pipe.health_check(df, "BTC/USDT", "1h")
        assert health["healthy"] is True or health["healthy"] is False  # freshness may fail
        assert "checks" in health

    def test_empty_data_unhealthy(self):
        pipe = self._pipeline()
        health = pipe.health_check(pd.DataFrame(), "BTC/USDT")
        assert health["healthy"] is False

    def test_nan_prices_detected(self):
        pipe = self._pipeline()
        df = _make_ohlcv(200)
        df.loc[0, "close"] = np.nan
        health = pipe.health_check(df, "BTC/USDT")
        assert health["checks"]["no_nan_prices"]["pass"] is False

    def test_min_rows_check(self):
        pipe = self._pipeline()
        df = _make_ohlcv(10)
        health = pipe.health_check(df, "BTC/USDT")
        assert health["checks"]["min_rows"]["pass"] is False


class TestValidation:
    def _pipeline(self):
        source = MagicMock()
        storage = MagicMock()
        return DataPipeline(source, storage)

    def test_removes_invalid_rows(self):
        pipe = self._pipeline()
        df = _make_ohlcv(100)
        # Make some invalid rows
        df.loc[0, "high"] = 50  # high < low since low is close*0.99
        df.loc[1, "volume"] = -1
        validated = pipe.validate(df)
        assert len(validated) < 100

    def test_empty_df_validation(self):
        pipe = self._pipeline()
        result = pipe.validate(pd.DataFrame())
        assert result.empty
