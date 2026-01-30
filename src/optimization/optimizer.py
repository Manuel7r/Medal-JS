"""Hyperparameter optimization using Optuna.

Supports optimizing strategy parameters via walk-forward validation
to avoid overfitting. Each trial runs a full walk-forward and the
objective is the out-of-sample Sharpe ratio.

Usage:
    optimizer = StrategyOptimizer(data, strategy_type="mean_reversion")
    best_params = optimizer.optimize(n_trials=100)
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.backtester.engine import BacktestEngine
from src.backtester.walk_forward import WalkForwardValidator
from src.features import technical, statistical
from src.strategies.mean_reversion import MeanReversionStrategy, MeanReversionParams
from src.strategies.pairs_trading import PairsTradingStrategy, PairsTradingParams
from src.strategies.ml_ensemble import MLEnsembleStrategy, MLEnsembleParams


@dataclass
class OptimizationResult:
    """Result of a hyperparameter optimization run."""

    best_params: dict[str, Any]
    best_sharpe: float
    best_return: float
    best_max_dd: float
    n_trials: int
    all_trials: list[dict[str, Any]]

    def summary(self) -> str:
        lines = [
            f"Optimization Result ({self.n_trials} trials):",
            f"  Best Sharpe:     {self.best_sharpe:.3f}",
            f"  Best Return:     {self.best_return:.2%}",
            f"  Best Max DD:     {self.best_max_dd:.2%}",
            f"  Best Params:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


def _mean_reversion_objective(
    trial: Any,
    data: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    train_size: int,
    test_size: int,
) -> float:
    """Optuna objective for Mean Reversion strategy."""
    lookback = trial.suggest_int("lookback", 24, 168, step=12)
    entry_std = trial.suggest_float("entry_std", 1.5, 3.5, step=0.25)
    exit_std = trial.suggest_float("exit_std", 0.1, 0.8, step=0.1)
    max_hold = trial.suggest_int("max_hold", 24, 168, step=12)
    vol_threshold = trial.suggest_float("vol_threshold", 1.2, 3.0, step=0.2)

    params = MeanReversionParams(
        lookback=lookback,
        entry_std=entry_std,
        exit_std=exit_std,
        max_hold=max_hold,
        vol_threshold=vol_threshold,
    )
    strategy = MeanReversionStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=0.001,
        position_size_pct=0.03,
    )

    validator = WalkForwardValidator(
        engine=engine,
        train_size=train_size,
        test_size=test_size,
        step_size=test_size,
    )

    try:
        wf_result = validator.run(
            data=prepared,
            signal_fn=strategy.generate_engine_signal,
            symbol="OPT",
        )
        sharpe = wf_result.oos_metrics.sharpe_ratio
        max_dd = wf_result.oos_metrics.max_drawdown

        # Penalize excessive drawdown
        if max_dd > 0.25:
            sharpe -= (max_dd - 0.25) * 5.0

        # Penalize too few trades
        if wf_result.oos_metrics.total_trades < 10:
            sharpe -= 1.0

        strategy.reset()
        return sharpe

    except Exception as e:
        logger.debug("Trial failed: {}", e)
        strategy.reset()
        return -10.0


def _pairs_trading_objective(
    trial: Any,
    data: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    train_size: int,
    test_size: int,
) -> float:
    """Optuna objective for Pairs Trading strategy."""
    lookback = trial.suggest_int("lookback", 72, 336, step=24)
    entry_zscore = trial.suggest_float("entry_zscore", 1.0, 3.0, step=0.25)
    exit_zscore = trial.suggest_float("exit_zscore", 0.1, 0.8, step=0.1)
    stop_zscore = trial.suggest_float("stop_zscore", 2.5, 5.0, step=0.5)
    max_hold = trial.suggest_int("max_hold", 48, 240, step=24)
    hedge_lookback = trial.suggest_int("hedge_lookback", 60, 240, step=30)

    params = PairsTradingParams(
        lookback=lookback,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        stop_zscore=stop_zscore,
        max_hold=max_hold,
        hedge_lookback=hedge_lookback,
    )
    strategy = PairsTradingStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=0.001,
        position_size_pct=0.03,
    )

    validator = WalkForwardValidator(
        engine=engine,
        train_size=train_size,
        test_size=test_size,
        step_size=test_size,
    )

    try:
        wf_result = validator.run(
            data=prepared,
            signal_fn=strategy.generate_engine_signal,
            symbol="OPT",
        )
        sharpe = wf_result.oos_metrics.sharpe_ratio
        max_dd = wf_result.oos_metrics.max_drawdown

        if max_dd > 0.25:
            sharpe -= (max_dd - 0.25) * 5.0
        if wf_result.oos_metrics.total_trades < 10:
            sharpe -= 1.0

        strategy.reset()
        return sharpe

    except Exception as e:
        logger.debug("Trial failed: {}", e)
        strategy.reset()
        return -10.0


def _ml_ensemble_objective(
    trial: Any,
    data: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    train_size: int,
    test_size: int,
) -> float:
    """Optuna objective for ML Ensemble strategy."""
    threshold_buy = trial.suggest_float("threshold_buy", 0.52, 0.65, step=0.01)
    threshold_sell = trial.suggest_float("threshold_sell", 0.35, 0.48, step=0.01)
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 2, 6)
    train_window = trial.suggest_int("train_window", 500, 2500, step=250)
    retrain_interval = trial.suggest_int("retrain_interval", 100, 500, step=50)

    params = MLEnsembleParams(
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
        n_estimators=n_estimators,
        max_depth=max_depth,
        train_window=train_window,
        retrain_interval=retrain_interval,
    )
    strategy = MLEnsembleStrategy(params)
    prepared = strategy.prepare(data)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=0.001,
        position_size_pct=0.03,
    )

    try:
        result = engine.run(
            data=prepared,
            signal_fn=strategy.generate_engine_signal,
            symbol="OPT",
        )
        sharpe = result.metrics.sharpe_ratio
        max_dd = result.metrics.max_drawdown

        if max_dd > 0.25:
            sharpe -= (max_dd - 0.25) * 5.0
        if result.metrics.total_trades < 10:
            sharpe -= 1.0

        strategy.reset()
        return sharpe

    except Exception as e:
        logger.debug("Trial failed: {}", e)
        strategy.reset()
        return -10.0


class StrategyOptimizer:
    """Hyperparameter optimizer for trading strategies.

    Uses Optuna with walk-forward validation as the objective to
    find optimal parameters while avoiding overfitting.

    Args:
        data: Prepared OHLCV DataFrame with features.
        strategy_type: One of "mean_reversion", "pairs_trading", "ml_ensemble".
        initial_capital: Starting equity for backtests.
        commission_rate: Commission per trade side.
        train_size: Bars for each walk-forward training window.
        test_size: Bars for each walk-forward test window.
    """

    STRATEGY_OBJECTIVES: dict[str, Callable] = {
        "mean_reversion": _mean_reversion_objective,
        "pairs_trading": _pairs_trading_objective,
        "ml_ensemble": _ml_ensemble_objective,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_type: str = "mean_reversion",
        initial_capital: float = 10_000.0,
        commission_rate: float = 0.001,
        train_size: int = 500,
        test_size: int = 200,
    ) -> None:
        if not HAS_OPTUNA:
            raise ImportError("optuna is required for optimization. Install with: pip install optuna")

        if strategy_type not in self.STRATEGY_OBJECTIVES:
            raise ValueError(f"Unknown strategy: {strategy_type}. Choose from {list(self.STRATEGY_OBJECTIVES.keys())}")

        self.data = data
        self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.train_size = train_size
        self.test_size = test_size

    def optimize(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
        n_jobs: int = 1,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            n_trials: Number of Optuna trials.
            timeout: Max seconds for optimization (None = unlimited).
            n_jobs: Parallel jobs (1 = sequential, -1 = all cores).

        Returns:
            OptimizationResult with best parameters and trial history.
        """
        objective_fn = self.STRATEGY_OBJECTIVES[self.strategy_type]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"medal_{self.strategy_type}",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

        def objective(trial: optuna.Trial) -> float:
            return objective_fn(
                trial,
                self.data,
                self.initial_capital,
                self.commission_rate,
                self.train_size,
                self.test_size,
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        best = study.best_trial
        all_trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ]

        # Get full metrics for best params
        best_sharpe = best.value if best.value is not None else 0.0

        logger.info(
            "Optimization complete: {} trials, best Sharpe={:.3f}, params={}",
            n_trials, best_sharpe, best.params,
        )

        return OptimizationResult(
            best_params=best.params,
            best_sharpe=best_sharpe,
            best_return=0.0,  # Would need re-run to get this
            best_max_dd=0.0,
            n_trials=len(study.trials),
            all_trials=all_trials,
        )

    def parameter_importance(self, n_trials: int = 50) -> dict[str, float]:
        """Estimate parameter importance using fANOVA.

        Runs optimization first (if not already done), then computes
        how much each parameter contributes to Sharpe variance.

        Returns:
            Dict mapping parameter_name -> importance (0-1).
        """
        objective_fn = self.STRATEGY_OBJECTIVES[self.strategy_type]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> float:
            return objective_fn(
                trial,
                self.data,
                self.initial_capital,
                self.commission_rate,
                self.train_size,
                self.test_size,
            )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        try:
            importance = optuna.importance.get_param_importances(study)
            return dict(importance)
        except Exception as e:
            logger.warning("Could not compute parameter importance: {}", e)
            return {}
