"""Walk-forward validation for strategy robustness testing.

Enhanced with:
    - Sharpe degradation diagnostics (IS vs OOS)
    - Monte Carlo trade shuffling for robustness
    - Regime detection (high/low volatility windows)
    - Parameter sensitivity summary
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.backtester.engine import BacktestEngine, BacktestResult
from src.backtester.metrics import BacktestMetrics, compute_metrics


@dataclass
class WalkForwardWindow:
    """A single train/test window."""

    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None


@dataclass
class RegimeStats:
    """Volatility regime statistics for a window."""

    window_id: int
    volatility: float
    regime: str  # "low", "normal", "high"
    sharpe: float
    n_trades: int


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""

    original_sharpe: float
    simulated_sharpes: list[float]
    p_value: float  # probability of original Sharpe under random
    percentile_5: float
    percentile_95: float
    n_simulations: int


@dataclass
class WalkForwardDiagnostics:
    """Diagnostic metrics comparing in-sample vs out-of-sample."""

    avg_is_sharpe: float
    avg_oos_sharpe: float
    sharpe_degradation: float  # (IS - OOS) / IS
    sharpe_stability: float  # std of OOS Sharpes across windows
    worst_window_sharpe: float
    best_window_sharpe: float
    pct_profitable_windows: float
    regime_stats: list[RegimeStats]
    monte_carlo: MonteCarloResult | None = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""

    windows: list[WalkForwardWindow]
    oos_equity: pd.Series
    oos_metrics: BacktestMetrics
    diagnostics: WalkForwardDiagnostics | None = None

    def summary(self) -> str:
        lines = [f"Walk-Forward: {len(self.windows)} windows"]
        for w in self.windows:
            if w.test_result:
                m = w.test_result.metrics
                lines.append(
                    f"  Window {w.window_id}: "
                    f"Sharpe={m.sharpe_ratio:.2f}, "
                    f"Return={m.total_return:.2%}, "
                    f"DD={m.max_drawdown:.2%}, "
                    f"Trades={m.total_trades}"
                )
        lines.append(f"\nOut-of-Sample Aggregate:")
        lines.append(self.oos_metrics.summary())

        if self.diagnostics:
            d = self.diagnostics
            lines.append(f"\nDiagnostics:")
            lines.append(f"  Avg IS Sharpe:       {d.avg_is_sharpe:.3f}")
            lines.append(f"  Avg OOS Sharpe:      {d.avg_oos_sharpe:.3f}")
            lines.append(f"  Sharpe Degradation:  {d.sharpe_degradation:.1%}")
            lines.append(f"  OOS Sharpe Std:      {d.sharpe_stability:.3f}")
            lines.append(f"  Worst Window:        {d.worst_window_sharpe:.3f}")
            lines.append(f"  Best Window:         {d.best_window_sharpe:.3f}")
            lines.append(f"  Profitable Windows:  {d.pct_profitable_windows:.0%}")

            if d.regime_stats:
                lines.append(f"\n  Regime Analysis:")
                for rs in d.regime_stats:
                    lines.append(
                        f"    W{rs.window_id}: {rs.regime:>6s} vol "
                        f"(σ={rs.volatility:.4f}), Sharpe={rs.sharpe:.2f}, Trades={rs.n_trades}"
                    )

            if d.monte_carlo:
                mc = d.monte_carlo
                lines.append(f"\n  Monte Carlo ({mc.n_simulations} sims):")
                lines.append(f"    Original Sharpe: {mc.original_sharpe:.3f}")
                lines.append(f"    p-value:         {mc.p_value:.3f}")
                lines.append(f"    95% CI:          [{mc.percentile_5:.3f}, {mc.percentile_95:.3f}]")

        return "\n".join(lines)


class WalkForwardValidator:
    """Rolling walk-forward validation with enhanced diagnostics.

    Args:
        engine: BacktestEngine instance.
        train_size: Number of bars for training window.
        test_size: Number of bars for test window.
        step_size: Number of bars to step forward between windows.
        monte_carlo_sims: Number of Monte Carlo simulations (0 to disable).
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_size: int = 2000,
        test_size: int = 500,
        step_size: int | None = None,
        monte_carlo_sims: int = 500,
    ) -> None:
        self.engine = engine
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.monte_carlo_sims = monte_carlo_sims

    def generate_windows(self, n_bars: int) -> list[WalkForwardWindow]:
        """Generate train/test index ranges."""
        windows: list[WalkForwardWindow] = []
        window_id = 0
        start = 0

        while start + self.train_size + self.test_size <= n_bars:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end
            test_end = min(train_end + self.test_size, n_bars)

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            window_id += 1
            start += self.step_size

        return windows

    def run(
        self,
        data: pd.DataFrame,
        signal_fn,
        train_fn=None,
        symbol: str = "SYM",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> WalkForwardResult:
        """Execute walk-forward validation with diagnostics.

        Args:
            data: Full OHLCV DataFrame with features.
            signal_fn: Callable(df, index) -> SignalType.
            train_fn: Optional callable(train_df) that returns a fitted signal_fn.
            symbol: Symbol name.
            stop_loss_pct: Fixed stop loss fraction.
            take_profit_pct: Fixed take profit fraction.

        Returns:
            WalkForwardResult with per-window metrics, aggregate, and diagnostics.
        """
        windows = self.generate_windows(len(data))
        if not windows:
            raise ValueError(
                f"Not enough data ({len(data)} bars) for walk-forward "
                f"(need {self.train_size + self.test_size})"
            )

        logger.info(
            "Walk-forward: {} windows, train={}, test={}, step={}",
            len(windows), self.train_size, self.test_size, self.step_size,
        )

        oos_equities: list[pd.Series] = []

        for w in windows:
            train_data = data.iloc[w.train_start : w.train_end].reset_index(drop=True)
            test_data = data.iloc[w.test_start : w.test_end].reset_index(drop=True)

            current_signal_fn = signal_fn
            if train_fn is not None:
                current_signal_fn = train_fn(train_data)

            w.train_result = self.engine.run(
                train_data, current_signal_fn, symbol,
                stop_loss_pct, take_profit_pct,
            )

            w.test_result = self.engine.run(
                test_data, current_signal_fn, symbol,
                stop_loss_pct, take_profit_pct,
            )

            oos_equities.append(w.test_result.equity_curve)

            logger.info(
                "Window {}: train Sharpe={:.2f}, test Sharpe={:.2f}",
                w.window_id,
                w.train_result.metrics.sharpe_ratio,
                w.test_result.metrics.sharpe_ratio,
            )

        oos_equity = self._chain_equity(oos_equities)

        all_oos_returns = pd.concat([
            pd.Series([t.return_pct for t in w.test_result.trades])
            for w in windows if w.test_result and w.test_result.trades
        ], ignore_index=True)
        if all_oos_returns.empty:
            all_oos_returns = pd.Series(dtype=float)

        oos_metrics = compute_metrics(
            equity_curve=oos_equity,
            trade_returns=all_oos_returns,
            periods_per_year=self.engine.periods_per_year,
        )

        # Compute diagnostics
        diagnostics = self._compute_diagnostics(windows, data, all_oos_returns)

        return WalkForwardResult(
            windows=windows,
            oos_equity=oos_equity,
            oos_metrics=oos_metrics,
            diagnostics=diagnostics,
        )

    def _compute_diagnostics(
        self,
        windows: list[WalkForwardWindow],
        data: pd.DataFrame,
        all_oos_returns: pd.Series,
    ) -> WalkForwardDiagnostics:
        """Compute enhanced diagnostics."""
        is_sharpes = []
        oos_sharpes = []

        for w in windows:
            if w.train_result:
                is_sharpes.append(w.train_result.metrics.sharpe_ratio)
            if w.test_result:
                oos_sharpes.append(w.test_result.metrics.sharpe_ratio)

        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

        degradation = (avg_is - avg_oos) / avg_is if avg_is != 0 else 0.0
        stability = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        worst = float(np.min(oos_sharpes)) if oos_sharpes else 0.0
        best = float(np.max(oos_sharpes)) if oos_sharpes else 0.0
        pct_profitable = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) if oos_sharpes else 0.0

        # Regime detection
        regime_stats = self._detect_regimes(windows, data)

        # Monte Carlo
        monte_carlo = None
        if self.monte_carlo_sims > 0 and len(all_oos_returns) > 10:
            monte_carlo = self._monte_carlo_test(all_oos_returns)

        return WalkForwardDiagnostics(
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            sharpe_degradation=degradation,
            sharpe_stability=stability,
            worst_window_sharpe=worst,
            best_window_sharpe=best,
            pct_profitable_windows=pct_profitable,
            regime_stats=regime_stats,
            monte_carlo=monte_carlo,
        )

    def _detect_regimes(
        self,
        windows: list[WalkForwardWindow],
        data: pd.DataFrame,
    ) -> list[RegimeStats]:
        """Classify each test window into volatility regime."""
        if "close" not in data.columns:
            return []

        returns = data["close"].pct_change().dropna()
        overall_vol = float(returns.std()) if len(returns) > 0 else 1.0

        stats: list[RegimeStats] = []
        for w in windows:
            if w.test_result is None:
                continue

            window_returns = returns.iloc[w.test_start:w.test_end]
            if len(window_returns) < 5:
                continue

            vol = float(window_returns.std())
            ratio = vol / overall_vol if overall_vol > 0 else 1.0

            if ratio < 0.7:
                regime = "low"
            elif ratio > 1.3:
                regime = "high"
            else:
                regime = "normal"

            stats.append(RegimeStats(
                window_id=w.window_id,
                volatility=vol,
                regime=regime,
                sharpe=w.test_result.metrics.sharpe_ratio,
                n_trades=w.test_result.metrics.total_trades,
            ))

        return stats

    def _monte_carlo_test(
        self,
        trade_returns: pd.Series,
    ) -> MonteCarloResult:
        """Monte Carlo permutation test for strategy significance.

        Shuffles the order of trade returns to test if the Sharpe
        ratio is statistically significant vs random ordering.
        """
        n = len(trade_returns)
        original_sharpe = self._trade_sharpe(trade_returns)

        rng = np.random.default_rng(42)
        sim_sharpes: list[float] = []

        for _ in range(self.monte_carlo_sims):
            shuffled = trade_returns.sample(n=n, replace=True, random_state=rng.integers(0, 2**31))
            sim_sharpes.append(self._trade_sharpe(shuffled))

        sim_arr = np.array(sim_sharpes)
        p_value = float(np.mean(sim_arr >= original_sharpe))

        return MonteCarloResult(
            original_sharpe=original_sharpe,
            simulated_sharpes=sim_sharpes,
            p_value=p_value,
            percentile_5=float(np.percentile(sim_arr, 5)),
            percentile_95=float(np.percentile(sim_arr, 95)),
            n_simulations=self.monte_carlo_sims,
        )

    @staticmethod
    def _trade_sharpe(returns: pd.Series) -> float:
        """Compute annualized Sharpe from trade returns."""
        if len(returns) < 2 or returns.std() < 1e-12:
            return 0.0
        # Approximate: assume ~3 trades per day for crypto
        return float(returns.mean() / returns.std() * np.sqrt(3 * 365))

    def _chain_equity(self, curves: list[pd.Series]) -> pd.Series:
        """Chain multiple equity curves by scaling each window relative to the previous."""
        if not curves:
            return pd.Series(dtype=float)

        chained: list[float] = []
        current_equity = self.engine.initial_capital

        for curve in curves:
            if len(curve) == 0:
                continue
            initial_curve = curve.iloc[0]
            if initial_curve == 0:
                continue
            for val in curve:
                scaled = current_equity * (val / initial_curve)
                chained.append(scaled)
            current_equity = chained[-1] if chained else current_equity

        return pd.Series(chained, name="oos_equity")
