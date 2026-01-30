"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    HOLD = "HOLD"
    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"


@dataclass
class StrategyParams:
    """Base container for strategy parameters."""

    name: str

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BaseStrategy(ABC):
    """Abstract base for all strategies.

    Subclasses must implement:
        - generate_signal(): produce a trading signal for the current bar.
        - get_params(): return current strategy parameters.
    """

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate a trading signal for the given bar index.

        Args:
            data: Full OHLCV DataFrame with any precomputed features.
            index: Current bar index (0-based).

        Returns:
            A Signal enum value.
        """

    @abstractmethod
    def get_params(self) -> StrategyParams:
        """Return current strategy parameters."""

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optional: precompute features on the full dataset before backtesting.

        Default implementation returns data unchanged.
        """
        return data
