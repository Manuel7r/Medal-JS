"""Prediction engine: generates real-time price direction predictions.

Runs as a scheduled job, fetching latest data and generating predictions
from all registered strategies. Resolves previous predictions against
actual prices to track accuracy.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

from src.data.sources import BinanceSource
from src.features import technical, statistical
from src.prediction.tracker import PredictionTracker
from src.strategies.base import BaseStrategy, Signal


class PredictionEngine:
    """Orchestrates prediction cycles across strategies and symbols.

    Usage:
        engine = PredictionEngine(source, strategies, tracker, symbols)
        # Called by scheduler every 15 min:
        summary = engine.run_cycle()
    """

    def __init__(
        self,
        source: BinanceSource,
        strategies: dict[str, BaseStrategy],
        tracker: PredictionTracker,
        symbols: list[str],
        timeframe: str = "1h",
    ) -> None:
        self._source = source
        self._strategies = strategies
        self._tracker = tracker
        self._symbols = symbols
        self._timeframe = timeframe
        self._last_cycle: datetime | None = None
        self._cycle_count = 0
        self._data_cache: dict[str, pd.DataFrame] = {}  # cached OHLCV per symbol

    def run_cycle(self) -> dict:
        """Main prediction cycle.

        1. Fetch latest candle for each symbol
        2. Resolve pending predictions from previous cycle
        3. Generate new predictions from each strategy
        4. Store predictions in tracker
        5. Return summary for broadcasting

        Returns:
            Dict with predictions and resolved results.
        """
        self._cycle_count += 1
        self._last_cycle = datetime.now(timezone.utc)
        since = datetime.now(timezone.utc) - timedelta(days=90)

        all_predictions: list[dict] = []
        all_resolved: list[dict] = []

        for symbol in self._symbols:
            try:
                # Use cache: fetch full history first time, then incremental
                if symbol in self._data_cache and len(self._data_cache[symbol]) > 500:
                    recent_since = datetime.now(timezone.utc) - timedelta(hours=2)
                    new_bars = self._source.fetch_ohlcv(
                        symbol, timeframe=self._timeframe, since=recent_since, limit=10,
                    )
                    if new_bars is not None and len(new_bars) > 0:
                        cached = self._data_cache[symbol]
                        combined = pd.concat([cached, new_bars]).drop_duplicates(
                            subset=["timestamp"], keep="last"
                        ).tail(2500)
                        self._data_cache[symbol] = combined.reset_index(drop=True)
                    df = self._data_cache[symbol]
                else:
                    df = self._source.fetch_ohlcv(
                        symbol, timeframe=self._timeframe, since=since, limit=2000,
                    )
                    if df is not None:
                        self._data_cache[symbol] = df

                if df is None or len(df) < 120:
                    logger.warning("Prediction: {} has too few bars ({})", symbol, len(df) if df is not None else 0)
                    continue

                current_price = float(df["close"].iloc[-1])

                # 1. Resolve pending predictions
                resolved = self._tracker.resolve_predictions(symbol, current_price)
                all_resolved.extend([p.to_dict() for p in resolved])

                # 2. Generate new predictions
                predictions = self._generate_predictions(symbol, df)
                all_predictions.extend(predictions)

            except Exception as e:
                logger.error("Prediction cycle failed for {}: {}", symbol, e)

        # Accuracy summary
        accuracy = self.get_accuracy_summary()

        logger.info(
            "Prediction cycle #{}: {} new predictions, {} resolved",
            self._cycle_count, len(all_predictions), len(all_resolved),
        )

        return {
            "predictions": all_predictions,
            "resolved": all_resolved,
            "accuracy": accuracy,
            "cycle": self._cycle_count,
            "timestamp": self._last_cycle.isoformat(),
        }

    def _generate_predictions(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> list[dict]:
        """Run all strategies on symbol, return predictions."""
        predictions: list[dict] = []
        current_price = float(data["close"].iloc[-1])

        for name, strategy in self._strategies.items():
            try:
                prepared = strategy.prepare(data.copy())
                signal = strategy.generate_signal(prepared, len(prepared) - 1)
                direction = self._signal_to_direction(signal)

                if direction is None:
                    continue

                confidence = self._extract_confidence(name, strategy, prepared)

                pred = self._tracker.record_prediction(
                    symbol=symbol,
                    strategy=name,
                    direction=direction,
                    confidence=confidence,
                    price=current_price,
                )
                predictions.append(pred.to_dict())

                logger.debug(
                    "Prediction: {} {} {} conf={:.2f} @ ${:.2f}",
                    name, symbol, direction, confidence, current_price,
                )

            except Exception as e:
                logger.error("Strategy {} failed on {}: {}", name, symbol, e)

        return predictions

    def _signal_to_direction(self, signal: Signal) -> str | None:
        """Map Signal enum to UP/DOWN direction."""
        if signal in (Signal.BUY, Signal.LONG_SPREAD):
            return "UP"
        if signal in (Signal.SELL, Signal.SHORT_SPREAD):
            return "DOWN"
        return None

    def _extract_confidence(
        self,
        name: str,
        strategy: BaseStrategy,
        data: pd.DataFrame,
    ) -> float:
        """Extract confidence from strategy.

        ML Ensemble: uses actual probability.
        Mean Reversion: derives from z-score magnitude.
        Momentum: derives from ADX value.
        Others: default 0.6.
        """
        # ML Ensemble has get_confidence()
        if hasattr(strategy, "get_confidence"):
            return strategy.get_confidence()

        # Mean Reversion: confidence from z-score magnitude
        if "mr_zscore" in data.columns:
            zscore = abs(float(data["mr_zscore"].iloc[-1]))
            return min(0.95, 0.5 + zscore * 0.1)

        # Momentum: confidence from ADX
        if "mom_adx" in data.columns:
            adx = float(data["mom_adx"].iloc[-1])
            return min(0.95, 0.4 + adx / 100.0)

        # Microstructure: confidence from VWAP z-score
        if "micro_vwap_zscore" in data.columns:
            vwap_z = abs(float(data["micro_vwap_zscore"].iloc[-1]))
            return min(0.95, 0.5 + vwap_z * 0.1)

        return 0.6

    def get_live_predictions(self) -> dict[str, list[dict]]:
        """Return current pending predictions grouped by symbol."""
        result: dict[str, list[dict]] = {}
        for symbol in self._symbols:
            pending = self._tracker.get_pending(symbol)
            if pending:
                result[symbol] = [p.to_dict() for p in pending]
        return result

    def get_accuracy_summary(self) -> dict:
        """Return accuracy metrics per strategy and overall."""
        strategies = list(self._strategies.keys())
        per_strategy: dict[str, dict] = {}
        for name in strategies:
            acc = self._tracker.get_accuracy(strategy=name)
            per_strategy[name] = acc.to_dict()

        overall = self._tracker.get_accuracy()
        ranking = self._tracker.strategy_ranking()

        return {
            "overall": overall.to_dict(),
            "per_strategy": per_strategy,
            "ranking": ranking,
            "total_pending": self._tracker.total_pending,
            "total_resolved": self._tracker.total_predictions,
        }

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_cycle(self) -> datetime | None:
        return self._last_cycle
