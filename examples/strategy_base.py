"""
Clase base para todas las estrategias de trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class Signal:
    """Representa una señal de trading."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'EXIT', 'HOLD'
    confidence: float  # 0.0 a 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class BaseStrategy(ABC):
    """
    Clase abstracta base para todas las estrategias.

    Todas las estrategias deben implementar:
    - generate_signal(): Genera señal para un símbolo
    - get_parameters(): Retorna parámetros actuales
    """

    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, float] = {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Genera señal de trading basada en datos.

        Args:
            data: DataFrame con datos OHLCV
            symbol: Símbolo del activo

        Returns:
            Signal con acción recomendada
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict:
        """Retorna diccionario con parámetros de la estrategia."""
        pass

    def has_position(self, symbol: str) -> bool:
        """Verifica si hay posición abierta en el símbolo."""
        return symbol in self.positions and self.positions[symbol] != 0

    def update_position(self, symbol: str, quantity: float):
        """Actualiza posición para un símbolo."""
        self.positions[symbol] = quantity

    def close_position(self, symbol: str):
        """Cierra posición de un símbolo."""
        if symbol in self.positions:
            del self.positions[symbol]


class SignalAggregator:
    """
    Agrega señales de múltiples estrategias.
    Usa votación ponderada para decisión final.
    """

    def __init__(self, strategies: Dict[str, BaseStrategy],
                 weights: Optional[Dict[str, float]] = None):
        self.strategies = strategies
        self.weights = weights or {name: 1.0 for name in strategies}

    def aggregate(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Agrega señales de todas las estrategias.

        Args:
            data: DataFrame con datos
            symbol: Símbolo a evaluar

        Returns:
            Signal agregada basada en votación ponderada
        """
        scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0, 'EXIT': 0.0}
        total_confidence = 0.0

        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(data, symbol)
            weight = self.weights.get(name, 1.0)

            scores[signal.action] += weight * signal.confidence
            total_confidence += signal.confidence * weight

        # Determinar acción ganadora
        total_weight = sum(self.weights.values())
        best_action = max(scores, key=scores.get)

        # Requiere mayoría para actuar
        if scores[best_action] < total_weight * 0.5:
            best_action = 'HOLD'

        avg_confidence = total_confidence / len(self.strategies) if self.strategies else 0

        return Signal(
            symbol=symbol,
            action=best_action,
            confidence=avg_confidence
        )
