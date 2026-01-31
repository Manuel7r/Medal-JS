"""Signal Aggregator: combines signals from multiple strategies via weighted voting.

Requires a weighted majority (> 50% of total weight) to produce a BUY or SELL.
Otherwise returns HOLD.
"""

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.strategies.base import BaseStrategy, Signal, StrategyParams


@dataclass
class AggregatorParams(StrategyParams):
    """Parameters for the Signal Aggregator."""

    name: str = "Aggregator"
    weights: dict[str, float] = field(default_factory=lambda: {
        "pairs_trading": 0.40,
        "mean_reversion": 0.35,
        "ml_ensemble": 0.25,
    })
    majority_threshold: float = 0.50  # fraction of total weight needed for consensus


class SignalAggregator:
    """Combines signals from multiple strategies using weighted voting.

    Each strategy contributes a weighted vote. A BUY or SELL signal
    is emitted only when its weighted score exceeds the majority threshold
    of total weight. EXIT is emitted if a majority votes EXIT.

    Usage:
        aggregator = SignalAggregator(params)
        aggregator.register("pairs_trading", pairs_strategy, 0.40)
        aggregator.register("mean_reversion", mr_strategy, 0.35)

        # Per bar:
        signal = aggregator.aggregate(data, index)
    """

    def __init__(self, params: AggregatorParams | None = None) -> None:
        self.params = params or AggregatorParams()
        self._strategies: dict[str, BaseStrategy] = {}
        self._weights: dict[str, float] = {}

    def register(self, name: str, strategy: BaseStrategy, weight: float | None = None) -> None:
        """Register a strategy with an optional weight override.

        Args:
            name: Strategy identifier.
            strategy: Strategy instance.
            weight: Voting weight. If None, uses params.weights[name] or 1.0.
        """
        self._strategies[name] = strategy
        self._weights[name] = weight if weight is not None else self.params.weights.get(name, 1.0)

    def update_weights(self, weights: dict[str, float]) -> None:
        """Update voting weights for registered strategies.

        Args:
            weights: Dict mapping strategy name -> new weight.
        """
        for name, weight in weights.items():
            if name in self._strategies:
                self._weights[name] = weight

    def aggregate(self, data: pd.DataFrame, index: int) -> Signal:
        """Collect signals from all strategies and vote.

        Returns:
            Winning Signal based on weighted majority.
        """
        scores: dict[str, float] = {
            "BUY": 0.0,
            "SELL": 0.0,
            "EXIT": 0.0,
            "HOLD": 0.0,
        }

        for name, strategy in self._strategies.items():
            signal = strategy.generate_signal(data, index)
            weight = self._weights.get(name, 1.0)

            # Normalize pairs signals
            if signal in (Signal.LONG_SPREAD, Signal.BUY):
                scores["BUY"] += weight
            elif signal in (Signal.SHORT_SPREAD, Signal.SELL):
                scores["SELL"] += weight
            elif signal == Signal.EXIT:
                scores["EXIT"] += weight
            else:
                scores["HOLD"] += weight

        total = sum(self._weights.values())
        threshold = total * self.params.majority_threshold

        if scores["BUY"] > threshold:
            return Signal.BUY
        if scores["SELL"] > threshold:
            return Signal.SELL
        if scores["EXIT"] > threshold:
            return Signal.EXIT
        return Signal.HOLD

    def aggregate_engine_signal(self, data: pd.DataFrame, index: int) -> SignalType:
        """Adapter for BacktestEngine."""
        signal = self.aggregate(data, index)
        mapping = {
            Signal.BUY: SignalType.BUY,
            Signal.SELL: SignalType.SELL,
            Signal.EXIT: SignalType.EXIT,
            Signal.HOLD: SignalType.HOLD,
        }
        return mapping[signal]

    def get_individual_signals(self, data: pd.DataFrame, index: int) -> dict[str, Signal]:
        """Get signals from each registered strategy (for debugging)."""
        return {
            name: strategy.generate_signal(data, index)
            for name, strategy in self._strategies.items()
        }

    def reset(self) -> None:
        """Reset all registered strategies."""
        for strategy in self._strategies.values():
            strategy.reset()


def backtest_aggregated(
    data: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    weights: dict[str, float] | None = None,
    majority_threshold: float = 0.50,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run a backtest with aggregated signals from multiple strategies.

    Args:
        data: OHLCV DataFrame (must already contain all features needed
              by the registered strategies).
        strategies: Dict mapping name -> strategy instance.
        weights: Dict mapping name -> voting weight.
        majority_threshold: Fraction of total weight for a signal to win.
        initial_capital: Starting equity.
        commission_rate: Commission per side.
        slippage_rate: Slippage per trade.
        position_size_pct: Fraction of equity per trade.

    Returns:
        BacktestResult.
    """
    params = AggregatorParams(
        weights=weights or {name: 1.0 for name in strategies},
        majority_threshold=majority_threshold,
    )
    aggregator = SignalAggregator(params)
    for name, strat in strategies.items():
        aggregator.register(name, strat, weights.get(name, 1.0) if weights else None)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        position_size_pct=position_size_pct,
    )

    result = engine.run(
        data=data,
        signal_fn=aggregator.aggregate_engine_signal,
        symbol="AGG",
    )

    logger.info(
        "Aggregator backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )

    return result
