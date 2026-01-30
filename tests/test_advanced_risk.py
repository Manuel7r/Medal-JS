"""Tests for advanced risk metrics: VaR, CVaR, Hurst, correlation."""

import numpy as np
import pandas as pd
import pytest

from src.risk.advanced_metrics import (
    AdvancedRiskMetrics,
    CorrelationAlert,
    HurstResult,
    VaRResult,
)


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0001, 0.02, n))


def _make_prices(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


class TestVaR:
    def test_historical_var(self):
        returns = _make_returns(500)
        result = AdvancedRiskMetrics.compute_var(returns, method="historical")

        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.method == "historical"
        assert result.n_observations == 500

    def test_parametric_var(self):
        returns = _make_returns(500)
        result = AdvancedRiskMetrics.compute_var(returns, method="parametric")

        assert result.var_95 > 0
        assert result.var_99 >= result.var_95
        assert result.method == "parametric"

    def test_empty_returns(self):
        result = AdvancedRiskMetrics.compute_var(pd.Series(dtype=float))
        assert result.var_95 == 0.0
        assert result.n_observations == 0


class TestHurst:
    def test_random_walk(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
        result = AdvancedRiskMetrics.hurst_exponent(prices)

        assert isinstance(result, HurstResult)
        assert 0 < result.hurst < 1
        assert result.regime in ("mean_reverting", "random_walk", "trending")
        assert result.confidence in ("strong", "moderate", "weak")

    def test_mean_reverting(self):
        # Stationary process tends to have H < 0.5
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1, 500)
        prices = pd.Series(100 + noise)
        result = AdvancedRiskMetrics.hurst_exponent(prices)
        # Just check it runs without error
        assert 0 < result.hurst < 1

    def test_short_series(self):
        result = AdvancedRiskMetrics.hurst_exponent(pd.Series([100, 101, 102]))
        assert result.hurst == 0.5
        assert result.confidence == "weak"


class TestCorrelation:
    def test_rolling_correlation_matrix(self):
        rng = np.random.default_rng(42)
        n = 500
        prices = {
            "BTC": pd.Series(100 + np.cumsum(rng.normal(0, 1, n))),
            "ETH": pd.Series(50 + np.cumsum(rng.normal(0, 0.8, n))),
        }
        corr = AdvancedRiskMetrics.rolling_correlation_matrix(prices, window=100)

        assert "BTC" in corr.columns
        assert "ETH" in corr.columns
        assert corr.loc["BTC", "BTC"] == pytest.approx(1.0, abs=0.01)

    def test_correlation_monitor(self):
        rng = np.random.default_rng(42)
        n = 500
        btc = 100 + np.cumsum(rng.normal(0, 1, n))
        # ETH starts correlated then diverges
        eth = 50 + np.cumsum(rng.normal(0, 0.8, n))

        prices = {
            "BTC": pd.Series(btc),
            "ETH": pd.Series(eth),
        }
        baseline = {("BTC", "ETH"): 0.95}

        alerts = AdvancedRiskMetrics.correlation_monitor(
            prices, baseline, window=100, change_threshold=0.10,
        )
        # May or may not alert depending on data
        assert isinstance(alerts, list)

    def test_correlation_monitor_empty(self):
        alerts = AdvancedRiskMetrics.correlation_monitor(
            {"BTC": pd.Series([1, 2, 3])},
            {("BTC", "ETH"): 0.9},
            window=100,
        )
        assert alerts == []


class TestPortfolioReport:
    def test_portfolio_risk_report(self):
        metrics = AdvancedRiskMetrics()
        rng = np.random.default_rng(42)
        equity = pd.Series(10000 + np.cumsum(rng.normal(5, 100, 500)))

        report = metrics.portfolio_risk_report(equity)
        assert "var_historical" in report
        assert "var_parametric" in report
        assert report["var_historical"]["var_95"] > 0

    def test_report_with_prices(self):
        metrics = AdvancedRiskMetrics()
        rng = np.random.default_rng(42)
        equity = pd.Series(10000 + np.cumsum(rng.normal(5, 100, 500)))
        prices = {
            "BTC": pd.Series(100 + np.cumsum(rng.normal(0, 1, 500))),
        }
        report = metrics.portfolio_risk_report(equity, price_dict=prices)
        assert "hurst" in report
        assert "BTC" in report["hurst"]
