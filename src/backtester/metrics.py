"""Performance metrics for backtesting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Container for all backtest performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # periods
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    expectancy: float

    def summary(self) -> str:
        return (
            f"Total Return:      {self.total_return:>10.2%}\n"
            f"Annual Return:     {self.annualized_return:>10.2%}\n"
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}\n"
            f"Sortino Ratio:     {self.sortino_ratio:>10.2f}\n"
            f"Calmar Ratio:      {self.calmar_ratio:>10.2f}\n"
            f"Max Drawdown:      {self.max_drawdown:>10.2%}\n"
            f"DD Duration:       {self.max_drawdown_duration:>10d} periods\n"
            f"Win Rate:          {self.win_rate:>10.2%}\n"
            f"Profit Factor:     {self.profit_factor:>10.2f}\n"
            f"Total Trades:      {self.total_trades:>10d}\n"
            f"Avg Trade:         {self.avg_trade_return:>10.2%}\n"
            f"Avg Win:           {self.avg_win:>10.2%}\n"
            f"Avg Loss:          {self.avg_loss:>10.2%}\n"
            f"Expectancy:        {self.expectancy:>10.4f}\n"
        )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 8760,
) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: Period returns series.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods in a year (8760 for hourly crypto 24/7).
    """
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if len(returns) < 2 or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 8760,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Maximum drawdown and its duration in periods.

    Args:
        equity_curve: Equity values over time.

    Returns:
        (max_drawdown_pct, max_duration_periods)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak

    max_dd = float(drawdown.min())

    # Duration: longest stretch below previous peak
    is_dd = equity_curve < peak
    if not is_dd.any():
        return 0.0, 0

    groups = (~is_dd).cumsum()
    dd_groups = is_dd.groupby(groups)
    max_duration = int(dd_groups.sum().max()) if len(dd_groups) > 0 else 0

    return abs(max_dd), max_duration


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: float = 8760,
) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_ret = float(returns.mean() * periods_per_year)
    dd, _ = max_drawdown(equity_curve)
    if dd == 0:
        return 0.0
    return ann_ret / dd


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of profitable trades."""
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def profit_factor(trade_returns: pd.Series) -> float:
    """Sum of wins / sum of losses."""
    wins = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def expectancy(trade_returns: pd.Series) -> float:
    """Average expected return per trade."""
    if len(trade_returns) == 0:
        return 0.0
    wr = win_rate(trade_returns)
    avg_w = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
    avg_l = float(trade_returns[trade_returns <= 0].mean()) if (trade_returns <= 0).any() else 0.0
    return wr * avg_w + (1 - wr) * avg_l


def compute_metrics(
    equity_curve: pd.Series,
    trade_returns: pd.Series,
    periods_per_year: float = 8760,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute all backtest metrics from equity curve and trade returns.

    Args:
        equity_curve: Equity values over time (index = timestamps or ints).
        trade_returns: Return per closed trade.
        periods_per_year: Annualization factor (8760 for hourly 24/7).
        risk_free_rate: Annual risk-free rate.
    """
    period_returns = equity_curve.pct_change().dropna()
    total_ret = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0
    n_periods = len(period_returns)
    ann_ret = float(period_returns.mean() * periods_per_year) if n_periods > 0 else 0.0

    dd, dd_dur = max_drawdown(equity_curve)

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]

    return BacktestMetrics(
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=sharpe_ratio(period_returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(period_returns, risk_free_rate, periods_per_year),
        calmar_ratio=calmar_ratio(period_returns, equity_curve, periods_per_year),
        max_drawdown=dd,
        max_drawdown_duration=dd_dur,
        win_rate=win_rate(trade_returns),
        profit_factor=profit_factor(trade_returns),
        total_trades=len(trade_returns),
        avg_trade_return=float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0,
        avg_win=float(wins.mean()) if len(wins) > 0 else 0.0,
        avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
        expectancy=expectancy(trade_returns),
    )
