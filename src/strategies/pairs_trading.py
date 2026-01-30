"""Pairs Trading strategy for cointegrated crypto pairs.

Exploits mean-reversion of the spread between two cointegrated assets.
When the z-score of the spread diverges beyond a threshold, the strategy
enters a position expecting reversion to the mean.

Designed for crypto (Binance) with adjusted parameters:
    - lookback: 168 hours (7 days)
    - entry_zscore: 2.0
    - stop_zscore: 3.5
    - exit_zscore: 0.5
    - max_hold: 168 hours
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.tsa.stattools import coint

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.backtester.metrics import BacktestMetrics
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class PairsTradingParams(StrategyParams):
    """Parameters for Pairs Trading strategy."""

    name: str = "PairsTrading"
    lookback: int = 168          # rolling window for z-score (hours)
    entry_zscore: float = 1.5    # entry threshold (earlier entry for more opportunities)
    exit_zscore: float = 0.2     # exit threshold (wait for fuller reversion)
    stop_zscore: float = 3.5     # stop loss threshold
    max_hold: int = 96           # max bars to hold (4 days — crypto moves fast)
    coint_pvalue: float = 0.05   # cointegration significance level
    hedge_lookback: int = 120    # rolling window for hedge ratio (more responsive)


class PairsTradingStrategy(BaseStrategy):
    """Pairs Trading on two cointegrated assets.

    This strategy operates on a merged DataFrame containing columns:
        close_a, close_b (prices of asset A and B)

    It precomputes:
        hedge_ratio, spread, spread_zscore

    Signal logic:
        - SHORT_SPREAD (z > entry): sell A, buy B
        - LONG_SPREAD  (z < -entry): buy A, sell B
        - EXIT          (|z| < exit or max_hold reached)
        - HOLD          otherwise

    For integration with BacktestEngine (single-symbol), the strategy maps:
        LONG_SPREAD  -> BUY  (buy A relative to B)
        SHORT_SPREAD -> SELL (sell A relative to B)
    """

    def __init__(self, params: PairsTradingParams | None = None) -> None:
        self.params = params or PairsTradingParams()
        self._position_bar: int | None = None

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Precompute hedge ratio, spread, and z-score.

        Expects columns: close_a, close_b.
        Adds columns: hedge_ratio, spread, spread_mean, spread_std, spread_zscore.
        """
        data = data.copy()
        p = self.params

        # Rolling hedge ratio via OLS
        data["hedge_ratio"] = self._rolling_hedge_ratio(
            data["close_a"], data["close_b"], p.hedge_lookback
        )

        # Spread
        data["spread"] = data["close_a"] - data["hedge_ratio"] * data["close_b"]

        # Rolling z-score of spread
        data["spread_mean"] = data["spread"].rolling(p.lookback).mean()
        data["spread_std"] = data["spread"].rolling(p.lookback).std()
        data["spread_zscore"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

        return data

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate pairs trading signal based on spread z-score."""
        p = self.params

        if index < p.lookback:
            return Signal.HOLD

        z = data["spread_zscore"].iloc[index]
        if np.isnan(z):
            return Signal.HOLD

        # Check max hold
        if self._position_bar is not None:
            bars_held = index - self._position_bar
            if bars_held >= p.max_hold:
                self._position_bar = None
                return Signal.EXIT

        # Check stop
        if self._position_bar is not None and abs(z) > p.stop_zscore:
            self._position_bar = None
            return Signal.EXIT

        # Exit on mean reversion
        if self._position_bar is not None and abs(z) < p.exit_zscore:
            self._position_bar = None
            return Signal.EXIT

        # Entry signals
        if self._position_bar is None:
            if z > p.entry_zscore:
                self._position_bar = index
                return Signal.SHORT_SPREAD
            if z < -p.entry_zscore:
                self._position_bar = index
                return Signal.LONG_SPREAD

        return Signal.HOLD

    def generate_engine_signal(self, data: pd.DataFrame, index: int) -> SignalType:
        """Adapter for BacktestEngine: maps pairs signals to BUY/SELL/EXIT/HOLD."""
        signal = self.generate_signal(data, index)
        mapping = {
            Signal.LONG_SPREAD: SignalType.BUY,
            Signal.SHORT_SPREAD: SignalType.SELL,
            Signal.EXIT: SignalType.EXIT,
            Signal.HOLD: SignalType.HOLD,
            Signal.BUY: SignalType.BUY,
            Signal.SELL: SignalType.SELL,
        }
        return mapping[signal]

    def reset(self) -> None:
        """Reset internal state for a new backtest run."""
        self._position_bar = None

    # --- Static analysis methods ---

    @staticmethod
    def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict:
        """Engle-Granger cointegration test between two price series.

        Returns:
            Dict with t_stat, p_value, critical_values, is_cointegrated.
        """
        clean_a = series_a.dropna()
        clean_b = series_b.dropna()
        common = clean_a.index.intersection(clean_b.index)
        t_stat, p_value, crit = coint(clean_a.loc[common], clean_b.loc[common])
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "critical_values": {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
            "is_cointegrated": bool(p_value < 0.05),
        }

    @staticmethod
    def hedge_ratio_ols(series_a: pd.Series, series_b: pd.Series) -> float:
        """OLS hedge ratio: series_a = beta * series_b + alpha."""
        clean_a = series_a.dropna()
        clean_b = series_b.dropna()
        common = clean_a.index.intersection(clean_b.index)
        slope, _, _, _, _ = stats.linregress(clean_b.loc[common], clean_a.loc[common])
        return float(slope)

    @staticmethod
    def _rolling_hedge_ratio(
        series_a: pd.Series, series_b: pd.Series, window: int
    ) -> pd.Series:
        """Compute rolling OLS hedge ratio."""
        ratios = pd.Series(np.nan, index=series_a.index)
        for i in range(window, len(series_a)):
            a_win = series_a.iloc[i - window : i]
            b_win = series_b.iloc[i - window : i]
            if b_win.std() == 0:
                ratios.iloc[i] = ratios.iloc[i - 1] if i > window else 1.0
                continue
            slope, _, _, _, _ = stats.linregress(b_win, a_win)
            ratios.iloc[i] = slope
        return ratios

    @staticmethod
    def scan_pairs(
        price_dict: dict[str, pd.Series],
        pvalue_threshold: float = 0.05,
    ) -> list[dict]:
        """Scan all combinations of symbols for cointegrated pairs.

        Args:
            price_dict: Dict mapping symbol -> close price Series.
            pvalue_threshold: Max p-value for cointegration.

        Returns:
            List of dicts with keys: pair, p_value, t_stat, hedge_ratio.
        """
        symbols = list(price_dict.keys())
        results: list[dict] = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a, sym_b = symbols[i], symbols[j]
                a, b = price_dict[sym_a], price_dict[sym_b]

                common = a.dropna().index.intersection(b.dropna().index)
                if len(common) < 100:
                    continue

                try:
                    t_stat, p_value, _ = coint(a.loc[common], b.loc[common])
                    if p_value < pvalue_threshold:
                        hr = PairsTradingStrategy.hedge_ratio_ols(a, b)
                        results.append({
                            "pair": (sym_a, sym_b),
                            "p_value": float(p_value),
                            "t_stat": float(t_stat),
                            "hedge_ratio": hr,
                        })
                except Exception:
                    continue

        results.sort(key=lambda x: x["p_value"])
        return results


def build_pairs_dataframe(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    symbol_a: str = "A",
    symbol_b: str = "B",
) -> pd.DataFrame:
    """Merge two OHLCV DataFrames into a pairs DataFrame.

    Expects both DataFrames to have 'timestamp' and 'close' columns.
    Returns DataFrame with columns: timestamp, close_a, close_b, close (= spread proxy).
    Also includes open/high/low from asset A for BacktestEngine compatibility.
    """
    a = df_a[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    a = a.rename(columns={
        "open": "open_a", "high": "high_a", "low": "low_a",
        "close": "close_a", "volume": "volume_a",
    })

    b = df_b[["timestamp", "close"]].copy()
    b = b.rename(columns={"close": "close_b"})

    merged = pd.merge(a, b, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    # BacktestEngine needs standard OHLCV columns — use asset A's
    merged["open"] = merged["open_a"]
    merged["high"] = merged["high_a"]
    merged["low"] = merged["low_a"]
    merged["close"] = merged["close_a"]
    merged["volume"] = merged["volume_a"]

    return merged


def backtest_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    params: PairsTradingParams | None = None,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run a full pairs trading backtest on two asset DataFrames.

    Args:
        df_a: OHLCV DataFrame for asset A.
        df_b: OHLCV DataFrame for asset B.
        params: Strategy parameters.
        initial_capital: Starting equity.
        commission_rate: Commission per side.
        slippage_rate: Slippage per trade.
        position_size_pct: Fraction of equity per trade.

    Returns:
        BacktestResult with metrics, equity curve, trades.
    """
    params = params or PairsTradingParams()
    strategy = PairsTradingStrategy(params)

    merged = build_pairs_dataframe(df_a, df_b)
    prepared = strategy.prepare(merged)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        position_size_pct=position_size_pct,
    )

    # ATR-based stops using asset A's ATR as proxy
    if "atr_14" in prepared.columns:
        median_atr = prepared["atr_14"].dropna().median()
        median_close = prepared["close"].dropna().median()
        atr_pct = median_atr / median_close if median_close > 0 else 0.02
        stop_loss = 2.5 * atr_pct
        take_profit = 4.0 * atr_pct
    else:
        stop_loss = 0.05
        take_profit = 0.08

    result = engine.run(
        data=prepared,
        signal_fn=strategy.generate_engine_signal,
        symbol="PAIR",
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    logger.info(
        "Pairs backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )

    return result
