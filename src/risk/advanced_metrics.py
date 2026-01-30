"""Advanced risk metrics: VaR, CVaR, Hurst Exponent, dynamic correlation.

Provides quantitative risk analysis beyond basic drawdown and Sharpe:
    - Value at Risk (VaR): Historical and parametric
    - Conditional VaR (CVaR / Expected Shortfall)
    - Hurst Exponent: mean-reversion vs momentum detection
    - Dynamic correlation monitoring between pairs
    - Cointegration stability tracking
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats


@dataclass
class VaRResult:
    """Value at Risk computation result."""

    var_95: float       # 95% VaR (loss threshold)
    var_99: float       # 99% VaR
    cvar_95: float      # 95% CVaR (Expected Shortfall)
    cvar_99: float      # 99% CVaR
    method: str         # "historical" or "parametric"
    n_observations: int


@dataclass
class HurstResult:
    """Hurst exponent computation result."""

    hurst: float
    regime: str     # "mean_reverting" (<0.5), "random_walk" (~0.5), "trending" (>0.5)
    confidence: str # "strong" if far from 0.5, "weak" if close


@dataclass
class CorrelationAlert:
    """Alert when correlation between a pair changes significantly."""

    pair: tuple[str, str]
    current_corr: float
    baseline_corr: float
    change: float
    is_breakdown: bool  # True if correlation dropped significantly


class AdvancedRiskMetrics:
    """Computes advanced risk metrics for the portfolio.

    Usage:
        metrics = AdvancedRiskMetrics()
        var = metrics.compute_var(returns)
        hurst = metrics.hurst_exponent(prices)
        alerts = metrics.correlation_monitor(price_dict, baseline_corrs)
    """

    # --- Value at Risk ---

    @staticmethod
    def compute_var(
        returns: pd.Series,
        confidence_levels: tuple[float, float] = (0.95, 0.99),
        method: str = "historical",
    ) -> VaRResult:
        """Compute Value at Risk and Conditional VaR.

        Args:
            returns: Series of period returns.
            confidence_levels: Confidence levels for VaR.
            method: "historical" or "parametric" (Gaussian).

        Returns:
            VaRResult with VaR and CVaR at specified levels.
        """
        returns = returns.dropna()
        n = len(returns)

        if n < 30:
            logger.warning("VaR: only {} observations, results may be unreliable", n)
            if n == 0:
                return VaRResult(0.0, 0.0, 0.0, 0.0, method, 0)

        if method == "historical":
            var_95 = float(np.percentile(returns, (1 - confidence_levels[0]) * 100))
            var_99 = float(np.percentile(returns, (1 - confidence_levels[1]) * 100))
            cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
            cvar_99 = float(returns[returns <= var_99].mean()) if (returns <= var_99).any() else var_99
        else:
            mu = float(returns.mean())
            sigma = float(returns.std())
            var_95 = mu + sigma * scipy_stats.norm.ppf(1 - confidence_levels[0])
            var_99 = mu + sigma * scipy_stats.norm.ppf(1 - confidence_levels[1])
            # Parametric CVaR for Gaussian
            cvar_95 = mu - sigma * scipy_stats.norm.pdf(scipy_stats.norm.ppf(1 - confidence_levels[0])) / (1 - confidence_levels[0])
            cvar_99 = mu - sigma * scipy_stats.norm.pdf(scipy_stats.norm.ppf(1 - confidence_levels[1])) / (1 - confidence_levels[1])

        return VaRResult(
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            cvar_99=abs(cvar_99),
            method=method,
            n_observations=n,
        )

    # --- Hurst Exponent ---

    @staticmethod
    def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> HurstResult:
        """Compute the Hurst exponent using R/S analysis.

        H < 0.5: Mean-reverting (good for mean reversion strategies)
        H = 0.5: Random walk (no predictable pattern)
        H > 0.5: Trending (good for momentum strategies)

        Args:
            prices: Price series (close prices).
            max_lag: Maximum lag for R/S analysis.

        Returns:
            HurstResult with exponent, regime classification, and confidence.
        """
        prices = prices.dropna()
        n = len(prices)

        if n < 50:
            return HurstResult(hurst=0.5, regime="random_walk", confidence="weak")

        returns = np.log(prices / prices.shift(1)).dropna().values
        lags = range(2, min(max_lag, n // 4))
        rs_values = []

        for lag in lags:
            rs_list = []
            for start in range(0, len(returns) - lag, lag):
                chunk = returns[start:start + lag]
                mean_chunk = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean_chunk)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 3:
            return HurstResult(hurst=0.5, regime="random_walk", confidence="weak")

        log_lags, log_rs = zip(*rs_values)
        slope, _, _, _, _ = scipy_stats.linregress(log_lags, log_rs)
        hurst = float(slope)

        # Classify regime
        if hurst < 0.4:
            regime = "mean_reverting"
        elif hurst > 0.6:
            regime = "trending"
        else:
            regime = "random_walk"

        # Confidence based on distance from 0.5
        distance = abs(hurst - 0.5)
        confidence = "strong" if distance > 0.15 else "moderate" if distance > 0.05 else "weak"

        return HurstResult(hurst=hurst, regime=regime, confidence=confidence)

    # --- Dynamic Correlation ---

    @staticmethod
    def rolling_correlation_matrix(
        price_dict: dict[str, pd.Series],
        window: int = 168,
    ) -> pd.DataFrame:
        """Compute current rolling correlation matrix.

        Args:
            price_dict: Dict mapping symbol -> close price Series.
            window: Rolling window for correlation.

        Returns:
            Correlation matrix DataFrame.
        """
        prices_df = pd.DataFrame(price_dict)
        returns_df = prices_df.pct_change().dropna()

        if len(returns_df) < window:
            return returns_df.corr()

        return returns_df.tail(window).corr()

    @staticmethod
    def correlation_monitor(
        price_dict: dict[str, pd.Series],
        baseline_corrs: dict[tuple[str, str], float],
        window: int = 168,
        change_threshold: float = 0.20,
    ) -> list[CorrelationAlert]:
        """Monitor correlation changes between tracked pairs.

        Args:
            price_dict: Dict mapping symbol -> close price Series.
            baseline_corrs: Expected correlations for each pair.
            window: Rolling window for current correlation.
            change_threshold: Alert if correlation changes by more than this.

        Returns:
            List of CorrelationAlert for pairs with significant changes.
        """
        prices_df = pd.DataFrame(price_dict)
        returns_df = prices_df.pct_change().dropna()

        if len(returns_df) < window:
            return []

        current = returns_df.tail(window)
        alerts: list[CorrelationAlert] = []

        for pair, baseline in baseline_corrs.items():
            sym_a, sym_b = pair
            if sym_a not in current.columns or sym_b not in current.columns:
                continue

            current_corr = float(current[sym_a].corr(current[sym_b]))
            change = current_corr - baseline
            is_breakdown = abs(change) > change_threshold

            if is_breakdown:
                alerts.append(CorrelationAlert(
                    pair=pair,
                    current_corr=current_corr,
                    baseline_corr=baseline,
                    change=change,
                    is_breakdown=True,
                ))
                logger.warning(
                    "Correlation alert: {} {} changed from {:.2f} to {:.2f} (delta={:.2f})",
                    sym_a, sym_b, baseline, current_corr, change,
                )

        return alerts

    # --- Cointegration Stability ---

    @staticmethod
    def cointegration_stability(
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 500,
        step: int = 100,
    ) -> pd.DataFrame:
        """Track cointegration p-value over rolling windows.

        Useful for detecting when a pairs trading relationship breaks down.

        Args:
            series_a: First price series.
            series_b: Second price series.
            window: Window size for cointegration test.
            step: Step size between windows.

        Returns:
            DataFrame with columns: window_start, window_end, p_value, is_cointegrated.
        """
        from statsmodels.tsa.stattools import coint

        n = min(len(series_a), len(series_b))
        results: list[dict] = []

        for start in range(0, n - window, step):
            end = start + window
            a_win = series_a.iloc[start:end].dropna()
            b_win = series_b.iloc[start:end].dropna()

            if len(a_win) < 100 or len(b_win) < 100:
                continue

            common = a_win.index.intersection(b_win.index)
            if len(common) < 100:
                continue

            try:
                t_stat, p_value, _ = coint(a_win.loc[common], b_win.loc[common])
                results.append({
                    "window_start": start,
                    "window_end": end,
                    "p_value": float(p_value),
                    "t_stat": float(t_stat),
                    "is_cointegrated": bool(p_value < 0.05),
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    # --- Portfolio-level summary ---

    def portfolio_risk_report(
        self,
        equity_curve: pd.Series,
        price_dict: dict[str, pd.Series] | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ) -> dict:
        """Generate comprehensive risk report.

        Args:
            equity_curve: Portfolio equity over time.
            price_dict: Optional dict of asset prices for correlation/Hurst.
            pairs: Optional list of pairs for cointegration tracking.

        Returns:
            Dict with all risk metrics.
        """
        report: dict = {}

        # VaR
        returns = equity_curve.pct_change().dropna()
        report["var_historical"] = {
            "var_95": self.compute_var(returns, method="historical").var_95,
            "var_99": self.compute_var(returns, method="historical").var_99,
            "cvar_95": self.compute_var(returns, method="historical").cvar_95,
            "cvar_99": self.compute_var(returns, method="historical").cvar_99,
        }
        report["var_parametric"] = {
            "var_95": self.compute_var(returns, method="parametric").var_95,
            "var_99": self.compute_var(returns, method="parametric").var_99,
            "cvar_95": self.compute_var(returns, method="parametric").cvar_95,
            "cvar_99": self.compute_var(returns, method="parametric").cvar_99,
        }

        # Hurst exponents per asset
        if price_dict:
            report["hurst"] = {}
            for sym, prices in price_dict.items():
                h = self.hurst_exponent(prices)
                report["hurst"][sym] = {
                    "hurst": h.hurst,
                    "regime": h.regime,
                    "confidence": h.confidence,
                }

            # Correlation matrix
            report["correlation_matrix"] = self.rolling_correlation_matrix(price_dict).to_dict()

        return report
