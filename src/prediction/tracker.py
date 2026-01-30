"""Prediction tracking and accuracy measurement.

Stores predictions, resolves them against actual prices,
and computes rolling accuracy metrics per strategy/symbol.
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from loguru import logger


@dataclass
class Prediction:
    """A single prediction record."""

    prediction_id: str
    symbol: str
    strategy: str
    direction: Literal["UP", "DOWN"]
    confidence: float
    price_at_prediction: float
    timestamp: datetime
    actual_direction: Literal["UP", "DOWN"] | None = None
    actual_price: float | None = None
    resolved_at: datetime | None = None
    correct: bool | None = None

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction,
            "confidence": self.confidence,
            "price_at_prediction": self.price_at_prediction,
            "timestamp": self.timestamp.isoformat(),
            "actual_direction": self.actual_direction,
            "actual_price": self.actual_price,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "correct": self.correct,
        }


@dataclass
class AccuracyMetrics:
    """Rolling accuracy for a strategy+symbol combo."""

    total_predictions: int = 0
    correct_predictions: int = 0
    hit_rate: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    calibration_error: float = 0.0
    recent_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "hit_rate": self.hit_rate,
            "avg_confidence_when_correct": self.avg_confidence_when_correct,
            "avg_confidence_when_wrong": self.avg_confidence_when_wrong,
            "calibration_error": self.calibration_error,
            "recent_hit_rate": self.recent_hit_rate,
        }


class PredictionTracker:
    """Tracks predictions and computes accuracy metrics.

    Usage:
        tracker = PredictionTracker()
        tracker.record_prediction("BTC/USDT", "MeanReversion", "UP", 0.72, 50000.0)
        # ... next candle arrives ...
        resolved = tracker.resolve_predictions("BTC/USDT", 50500.0)
        accuracy = tracker.get_accuracy(strategy="MeanReversion")
    """

    def __init__(self, max_history: int = 5000) -> None:
        self._history: deque[Prediction] = deque(maxlen=max_history)
        self._pending: dict[str, list[Prediction]] = {}  # symbol -> pending predictions

    def record_prediction(
        self,
        symbol: str,
        strategy: str,
        direction: Literal["UP", "DOWN"],
        confidence: float,
        price: float,
    ) -> Prediction:
        """Record a new prediction."""
        pred = Prediction(
            prediction_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            confidence=max(0.0, min(1.0, confidence)),
            price_at_prediction=price,
            timestamp=datetime.now(timezone.utc),
        )
        if symbol not in self._pending:
            self._pending[symbol] = []
        self._pending[symbol].append(pred)
        return pred

    def resolve_predictions(
        self,
        symbol: str,
        current_price: float,
    ) -> list[Prediction]:
        """Resolve all pending predictions for a symbol.

        Compares price_at_prediction vs current_price to determine
        if the predicted direction was correct.
        """
        pending = self._pending.pop(symbol, [])
        resolved: list[Prediction] = []
        now = datetime.now(timezone.utc)

        for pred in pending:
            actual_dir: Literal["UP", "DOWN"] = (
                "UP" if current_price > pred.price_at_prediction else "DOWN"
            )
            pred.actual_direction = actual_dir
            pred.actual_price = current_price
            pred.resolved_at = now
            pred.correct = pred.direction == actual_dir
            self._history.append(pred)
            resolved.append(pred)

        if resolved:
            correct = sum(1 for p in resolved if p.correct)
            logger.info(
                "Resolved {} predictions for {}: {}/{} correct",
                len(resolved), symbol, correct, len(resolved),
            )

        return resolved

    def get_accuracy(
        self,
        strategy: str | None = None,
        symbol: str | None = None,
    ) -> AccuracyMetrics:
        """Compute accuracy metrics, optionally filtered."""
        resolved = [
            p for p in self._history
            if p.correct is not None
            and (strategy is None or p.strategy == strategy)
            and (symbol is None or p.symbol == symbol)
        ]

        if not resolved:
            return AccuracyMetrics()

        total = len(resolved)
        correct_preds = [p for p in resolved if p.correct]
        wrong_preds = [p for p in resolved if not p.correct]
        correct = len(correct_preds)
        hit_rate = correct / total if total > 0 else 0.0

        avg_conf_correct = (
            sum(p.confidence for p in correct_preds) / len(correct_preds)
            if correct_preds else 0.0
        )
        avg_conf_wrong = (
            sum(p.confidence for p in wrong_preds) / len(wrong_preds)
            if wrong_preds else 0.0
        )

        # Recent hit rate (last 20)
        recent = resolved[-20:]
        recent_correct = sum(1 for p in recent if p.correct)
        recent_hit = recent_correct / len(recent) if recent else 0.0

        # Calibration error: |avg_confidence - hit_rate|
        avg_conf = sum(p.confidence for p in resolved) / total
        calibration = abs(avg_conf - hit_rate)

        return AccuracyMetrics(
            total_predictions=total,
            correct_predictions=correct,
            hit_rate=hit_rate,
            avg_confidence_when_correct=avg_conf_correct,
            avg_confidence_when_wrong=avg_conf_wrong,
            calibration_error=calibration,
            recent_hit_rate=recent_hit,
        )

    def get_pending(self, symbol: str | None = None) -> list[Prediction]:
        """Get pending (unresolved) predictions."""
        if symbol:
            return list(self._pending.get(symbol, []))
        return [p for preds in self._pending.values() for p in preds]

    def get_history(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        """Get resolved predictions, most recent first."""
        results = [
            p for p in self._history
            if (symbol is None or p.symbol == symbol)
            and (strategy is None or p.strategy == strategy)
        ]
        return list(reversed(results[-limit:]))

    def strategy_ranking(self) -> list[dict]:
        """Return strategies ranked by hit_rate descending."""
        strategies = set(p.strategy for p in self._history if p.correct is not None)
        ranking = []
        for strat in strategies:
            acc = self.get_accuracy(strategy=strat)
            ranking.append({
                "strategy": strat,
                "hit_rate": acc.hit_rate,
                "total_predictions": acc.total_predictions,
                "recent_hit_rate": acc.recent_hit_rate,
            })
        ranking.sort(key=lambda x: x["hit_rate"], reverse=True)
        return ranking

    @property
    def total_predictions(self) -> int:
        return len(self._history)

    @property
    def total_pending(self) -> int:
        return sum(len(v) for v in self._pending.values())
