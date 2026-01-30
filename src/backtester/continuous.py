"""Continuous backtester: re-evaluates strategy performance on rolling windows.

Detects strategy degradation by comparing current metrics against
historical baselines. Triggers alerts when Sharpe ratio or win rate
drop below thresholds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult
from src.backtester.metrics import BacktestMetrics
from src.features import technical, statistical
from src.strategies.base import BaseStrategy


@dataclass
class DegradationAlert:
    """Alert when a strategy metric degrades significantly."""

    strategy: str
    symbol: str
    metric: str
    historical_value: float
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "metric": self.metric,
            "historical_value": self.historical_value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ContinuousBacktestResult:
    """Result of a continuous backtest run for one strategy+symbol."""

    strategy: str
    symbol: str
    metrics: BacktestMetrics
    window_start: int
    window_end: int
    degraded: bool
    alerts: list[DegradationAlert]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "total_return": self.metrics.total_return,
            "max_drawdown": self.metrics.max_drawdown,
            "win_rate": self.metrics.win_rate,
            "total_trades": self.metrics.total_trades,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "degraded": self.degraded,
            "alerts": [a.to_dict() for a in self.alerts],
        }


class ContinuousBacktester:
    """Re-runs backtests on rolling windows to detect strategy degradation.

    Usage:
        cb = ContinuousBacktester(strategies, symbols)
        # Set baseline from initial backtests:
        cb.set_baseline("MeanReversion", "BTC/USDT", initial_metrics)
        # Run periodically (every 4h):
        results = cb.run_cycle(ohlcv_data)
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        symbols: list[str],
        window_size: int = 500,
        min_sharpe_threshold: float = 0.3,
        sharpe_drop_pct: float = 0.30,
        initial_capital: float = 10000.0,
    ) -> None:
        self._strategies = strategies
        self._symbols = symbols
        self._window_size = window_size
        self._min_sharpe = min_sharpe_threshold
        self._sharpe_drop = sharpe_drop_pct
        self._initial_capital = initial_capital
        self._baselines: dict[str, BacktestMetrics] = {}  # "strategy:symbol" -> metrics
        self._latest_results: list[ContinuousBacktestResult] = []
        self._all_alerts: list[DegradationAlert] = []
        self._cycle_count = 0

    def set_baseline(
        self,
        strategy: str,
        symbol: str,
        metrics: BacktestMetrics,
    ) -> None:
        """Set historical baseline metrics for comparison."""
        key = f"{strategy}:{symbol}"
        self._baselines[key] = metrics
        logger.info(
            "Baseline set: {} {} Sharpe={:.2f} WR={:.1%}",
            strategy, symbol, metrics.sharpe_ratio, metrics.win_rate,
        )

    def run_cycle(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
    ) -> list[ContinuousBacktestResult]:
        """Re-run backtests on latest data window for all strategy+symbol combos."""
        self._cycle_count += 1
        results: list[ContinuousBacktestResult] = []

        for strat_name, strategy in self._strategies.items():
            for symbol in self._symbols:
                if symbol not in ohlcv_data:
                    continue

                df = ohlcv_data[symbol]
                if len(df) < self._window_size:
                    continue

                try:
                    result = self._run_single(strat_name, strategy, symbol, df)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        "Continuous backtest failed: {} {}: {}",
                        strat_name, symbol, e,
                    )

        self._latest_results = results

        degraded = [r for r in results if r.degraded]
        logger.info(
            "Continuous backtest cycle #{}: {} results, {} degraded",
            self._cycle_count, len(results), len(degraded),
        )

        return results

    def _run_single(
        self,
        strat_name: str,
        strategy: BaseStrategy,
        symbol: str,
        df: pd.DataFrame,
    ) -> ContinuousBacktestResult:
        """Run backtest for one strategy+symbol on latest window."""
        # Take last window_size bars
        window = df.iloc[-self._window_size:].copy()
        window_start = len(df) - self._window_size
        window_end = len(df)

        # Prepare features
        window = technical.compute_all(window)
        window = statistical.compute_all(window)
        prepared = strategy.prepare(window)

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self._initial_capital,
            commission_rate=0.001,
            slippage_rate=0.001,
            position_size_pct=0.03,
        )
        result = engine.run(
            data=prepared,
            signal_fn=strategy.generate_engine_signal,
            symbol=symbol,
        )

        # Check degradation
        alerts = self._check_degradation(strat_name, symbol, result.metrics)
        degraded = len(alerts) > 0
        self._all_alerts.extend(alerts)

        return ContinuousBacktestResult(
            strategy=strat_name,
            symbol=symbol,
            metrics=result.metrics,
            window_start=window_start,
            window_end=window_end,
            degraded=degraded,
            alerts=alerts,
        )

    def _check_degradation(
        self,
        strategy: str,
        symbol: str,
        current: BacktestMetrics,
    ) -> list[DegradationAlert]:
        """Compare current metrics against baseline."""
        alerts: list[DegradationAlert] = []
        key = f"{strategy}:{symbol}"
        baseline = self._baselines.get(key)

        # Absolute Sharpe threshold
        if current.sharpe_ratio < self._min_sharpe:
            alerts.append(DegradationAlert(
                strategy=strategy,
                symbol=symbol,
                metric="sharpe_absolute",
                historical_value=baseline.sharpe_ratio if baseline else 0.0,
                current_value=current.sharpe_ratio,
                threshold=self._min_sharpe,
            ))

        # Relative Sharpe drop
        if baseline and baseline.sharpe_ratio > 0:
            drop = 1.0 - (current.sharpe_ratio / baseline.sharpe_ratio)
            if drop > self._sharpe_drop:
                alerts.append(DegradationAlert(
                    strategy=strategy,
                    symbol=symbol,
                    metric="sharpe_drop",
                    historical_value=baseline.sharpe_ratio,
                    current_value=current.sharpe_ratio,
                    threshold=self._sharpe_drop,
                ))

        # Win rate drop
        if baseline and baseline.win_rate > 0:
            wr_drop = baseline.win_rate - current.win_rate
            if wr_drop > 0.15:  # >15% win rate drop
                alerts.append(DegradationAlert(
                    strategy=strategy,
                    symbol=symbol,
                    metric="win_rate_drop",
                    historical_value=baseline.win_rate,
                    current_value=current.win_rate,
                    threshold=0.15,
                ))

        for alert in alerts:
            logger.warning(
                "Degradation: {} {} {}: {:.2f} -> {:.2f} (threshold: {:.2f})",
                strategy, symbol, alert.metric,
                alert.historical_value, alert.current_value, alert.threshold,
            )

        return alerts

    def get_latest_results(self) -> list[dict]:
        """Return latest continuous backtest results."""
        return [r.to_dict() for r in self._latest_results]

    def get_degradation_alerts(self) -> list[dict]:
        """Return all degradation alerts (latest 50)."""
        return [a.to_dict() for a in self._all_alerts[-50:]]

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
