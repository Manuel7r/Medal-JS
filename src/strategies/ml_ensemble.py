"""ML Ensemble strategy: XGBoost + RandomForest + LightGBM.

Combines three gradient-boosted / tree-based classifiers via weighted
probability averaging to predict next-bar direction.

Features (20+):
    Technical: rsi_14, macd, macd_histogram, bb_position, atr_14
    Price action: returns_1d, returns_5d, returns_20d
    Volume: volume_ma_ratio
    Statistical: z_score_20, returns_skew, returns_kurt
    Lagged: returns_lag_1, returns_lag_2, returns_lag_5
    Derived: rsi_change, macd_cross, bb_squeeze, atr_ratio, vol_trend

Target: binary classification (1 = price up next bar, 0 = price down).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.backtester.engine import BacktestEngine, BacktestResult, SignalType
from src.features import technical, statistical
from src.strategies.base import BaseStrategy, Signal, StrategyParams

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


FEATURE_COLS = [
    # Technical
    "rsi_14", "macd", "macd_histogram", "bb_position", "atr_14",
    # Price action
    "returns_1d", "returns_5d", "returns_20d",
    # Volume
    "volume_ma_ratio",
    # Statistical
    "z_score_20", "returns_skew", "returns_kurt",
    # Lagged
    "returns_lag_1", "returns_lag_2", "returns_lag_5",
    # Derived (added in prepare)
    "rsi_change", "macd_cross", "bb_squeeze", "atr_ratio", "vol_trend",
]


@dataclass
class MLEnsembleParams(StrategyParams):
    """Parameters for ML Ensemble strategy."""

    name: str = "MLEnsemble"
    weights: dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.4,
        "random_forest": 0.3,
        "lightgbm": 0.3,
    })
    threshold_buy: float = 0.60
    threshold_sell: float = 0.40
    min_confidence: float = 0.55   # minimum |prob - 0.5| * 2 to trade
    max_trades_per_day: int = 1    # max entries per 24h window
    train_window: int = 1500      # bars for training (shorter = adapt faster)
    retrain_interval: int = 250   # retrain every N bars
    n_estimators: int = 150
    max_depth: int = 3            # shallow trees to reduce overfitting
    cv_folds: int = 5


class MLEnsembleStrategy(BaseStrategy):
    """ML Ensemble using XGBoost, RandomForest, and LightGBM.

    The strategy trains on historical features and predicts the probability
    of the next bar closing higher. Signals:
        BUY:  weighted probability > threshold_buy (0.55)
        SELL: weighted probability < threshold_sell (0.45)
        HOLD: otherwise

    Models are trained on the first `train_window` bars and retrained
    every `retrain_interval` bars using an expanding window.
    """

    def __init__(self, params: MLEnsembleParams | None = None) -> None:
        self.params = params or MLEnsembleParams()
        self._models: dict[str, object] = {}
        self._is_trained = False
        self._last_train_bar = -1
        self._last_features: list[str] = []
        self._last_probability: float | None = None
        self._daily_trades: int = 0
        self._last_trade_bar: int = -9999

    def get_params(self) -> StrategyParams:
        return self.params

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all features needed for ML prediction.

        Expects standard OHLCV columns. Adds technical, statistical,
        and derived features.
        """
        data = technical.compute_all(data)
        data = statistical.compute_all(data)

        # Derived features
        data["rsi_change"] = data["rsi_14"].diff()
        data["macd_cross"] = (data["macd"] - data["macd_signal"]).apply(np.sign)
        bb_range = data["bb_upper"] - data["bb_lower"]
        data["bb_squeeze"] = bb_range / data["bb_middle"]
        data["atr_ratio"] = data["atr_14"] / data["close"]
        data["vol_trend"] = data["volume_ma_ratio"].diff(5)

        # Target: next bar return > 0
        data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

        return data

    def _build_models(self) -> dict[str, object]:
        """Instantiate fresh model objects."""
        p = self.params
        models: dict[str, object] = {}

        models["random_forest"] = RandomForestClassifier(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            random_state=42,
            n_jobs=-1,
        )

        if HAS_XGB:
            models["xgboost"] = XGBClassifier(
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
            )

        if HAS_LGB:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                random_state=42,
                verbose=-1,
            )

        return models

    def train(self, data: pd.DataFrame, end_index: int) -> None:
        """Train all models on a rolling window ending at end_index.

        Uses a rolling train_window to avoid data leakage from
        training on the full history. Only the most recent train_window
        bars are used.

        Args:
            data: Prepared DataFrame with features and target.
            end_index: Train on data[start:end_index] where start
                       is max(0, end_index - train_window).
        """
        start_index = max(0, end_index - self.params.train_window)
        train = data.iloc[start_index:end_index].dropna(subset=FEATURE_COLS + ["target"])
        if len(train) < 100:
            logger.warning("Not enough training data ({} rows), skipping train", len(train))
            return

        available_features = [c for c in FEATURE_COLS if c in train.columns]
        self._last_features = available_features
        X = train[available_features].values
        y = train["target"].values

        self._models = self._build_models()

        for name, model in self._models.items():
            try:
                model.fit(X, y)  # type: ignore[union-attr]
            except Exception as e:
                logger.error("Failed to train {}: {}", name, e)

        self._is_trained = True
        self._last_train_bar = end_index
        logger.info(
            "Trained {} models on {} samples (window [{}, {}])",
            len(self._models), len(train),
            max(0, end_index - self.params.train_window), end_index,
        )

    def cross_validate(self, data: pd.DataFrame, end_index: int) -> dict[str, float]:
        """Run cross-validation on training data.

        Returns:
            Dict mapping model_name -> mean CV accuracy.
        """
        train = data.iloc[:end_index].dropna(subset=FEATURE_COLS + ["target"])
        available_features = [c for c in FEATURE_COLS if c in train.columns]
        X = train[available_features].values
        y = train["target"].values

        results: dict[str, float] = {}
        models = self._build_models()
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=self.params.cv_folds, scoring="accuracy")
                results[name] = float(scores.mean())
            except Exception as e:
                logger.error("CV failed for {}: {}", name, e)
                results[name] = 0.0

        return results

    def predict_proba(self, data: pd.DataFrame, index: int) -> float | None:
        """Weighted ensemble probability of next bar being up.

        Returns:
            Float in [0, 1] or None if models not trained.
        """
        if not self._is_trained or not self._models:
            return None

        available_features = [c for c in FEATURE_COLS if c in data.columns]
        row = data.iloc[index][available_features]
        if row.isna().any():
            return None

        X = row.values.reshape(1, -1)
        p = self.params

        total_weight = 0.0
        weighted_prob = 0.0

        for name, model in self._models.items():
            weight = p.weights.get(name, 0.0)
            if weight <= 0:
                continue
            try:
                prob = model.predict_proba(X)[0, 1]  # type: ignore[union-attr]
                weighted_prob += prob * weight
                total_weight += weight
            except Exception:
                continue

        if total_weight == 0:
            return None

        return weighted_prob / total_weight

    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """Generate ML ensemble signal."""
        p = self.params

        # Need enough data for training
        if index < p.train_window:
            return Signal.HOLD

        # Train or retrain
        if not self._is_trained or (index - self._last_train_bar >= p.retrain_interval):
            self.train(data, index)

        prob = self.predict_proba(data, index)
        self._last_probability = prob
        if prob is None:
            return Signal.HOLD

        # Confidence filter: require strong conviction
        confidence = abs(prob - 0.5) * 2  # 0-1 scale
        if confidence < p.min_confidence:
            return Signal.HOLD

        # Daily trade limit (24 bars = 24h at 1h timeframe)
        if index - self._last_trade_bar < 24:
            return Signal.HOLD

        if prob > p.threshold_buy:
            self._last_trade_bar = index
            return Signal.BUY
        if prob < p.threshold_sell:
            self._last_trade_bar = index
            return Signal.SELL
        return Signal.HOLD

    def get_confidence(self) -> float:
        """Return the last prediction probability as confidence."""
        return self._last_probability if self._last_probability is not None else 0.5

    def generate_engine_signal(self, data: pd.DataFrame, index: int) -> SignalType:
        """Adapter for BacktestEngine."""
        signal = self.generate_signal(data, index)
        mapping = {
            Signal.BUY: SignalType.BUY,
            Signal.SELL: SignalType.SELL,
            Signal.EXIT: SignalType.EXIT,
            Signal.HOLD: SignalType.HOLD,
            Signal.LONG_SPREAD: SignalType.BUY,
            Signal.SHORT_SPREAD: SignalType.SELL,
        }
        return mapping[signal]

    def reset(self) -> None:
        """Reset internal state."""
        self._models = {}
        self._is_trained = False
        self._last_train_bar = -1
        self._last_features = []
        self._daily_trades = 0
        self._last_trade_bar = -9999

    def feature_importance(self) -> dict[str, pd.Series]:
        """Get feature importances from trained models.

        Returns:
            Dict mapping model_name -> Series of importances indexed by feature name.
        """
        if not self._is_trained:
            return {}

        result: dict[str, pd.Series] = {}

        for name, model in self._models.items():
            try:
                imp = model.feature_importances_  # type: ignore[union-attr]
                result[name] = pd.Series(
                    imp,
                    index=self._last_features[:len(imp)],
                ).sort_values(ascending=False)
            except AttributeError:
                continue

        return result


def backtest_ml_ensemble(
    data: pd.DataFrame,
    params: MLEnsembleParams | None = None,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    position_size_pct: float = 0.03,
) -> BacktestResult:
    """Run a full ML ensemble backtest.

    Args:
        data: OHLCV DataFrame.
        params: Strategy parameters.
        initial_capital: Starting equity.
        commission_rate: Commission per side.
        slippage_rate: Slippage per trade.
        position_size_pct: Fraction of equity per trade.

    Returns:
        BacktestResult with metrics, equity curve, trades.
    """
    params = params or MLEnsembleParams()
    strategy = MLEnsembleStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        position_size_pct=position_size_pct,
    )

    result = engine.run(
        data=prepared,
        signal_fn=strategy.generate_engine_signal,
        symbol="ML",
        atr_stop_multiplier=2.5,
        atr_tp_multiplier=4.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
    )

    logger.info(
        "MLEnsemble backtest: {} trades, Sharpe={:.2f}, Return={:.2%}, MaxDD={:.2%}",
        result.metrics.total_trades,
        result.metrics.sharpe_ratio,
        result.metrics.total_return,
        result.metrics.max_drawdown,
    )

    return result
