"""Pydantic response schemas for the REST API."""

from pydantic import BaseModel


class PositionOut(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float


class PortfolioOut(BaseModel):
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    total_commission: float


class PositionsOut(BaseModel):
    count: int
    leverage: float
    positions: list[PositionOut]


class OrdersSummaryOut(BaseModel):
    total_orders: int
    total_fills: int
    summary: dict[str, int]


class RiskOut(BaseModel):
    drawdown_pct: float
    leverage: float
    suspended: bool
    paused: bool


class AlertOut(BaseModel):
    type: str
    level: str
    message: str
    time: str
    acknowledged: bool


class AlertsOut(BaseModel):
    unacknowledged: int
    recent: list[AlertOut]


class JobOut(BaseModel):
    name: str
    last_run: str | None
    run_count: int
    error_count: int
    last_error: str | None


class SchedulerOut(BaseModel):
    running: bool
    jobs: dict[str, JobOut]


class DashboardOut(BaseModel):
    portfolio: PortfolioOut
    positions: PositionsOut
    orders: OrdersSummaryOut
    risk: RiskOut
    alerts: AlertsOut
    scheduler: SchedulerOut
    last_update: str


class OrderOut(BaseModel):
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None
    status: str
    filled_quantity: float
    filled_price: float
    commission: float
    created_at: str
    updated_at: str


class FillOut(BaseModel):
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: str


class EquityPointOut(BaseModel):
    timestamp: str
    equity: float
    drawdown_pct: float
    daily_pnl: float


class BacktestOut(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    expectancy: float


class StrategyParamsOut(BaseModel):
    name: str
    params: dict


class RiskAdvancedOut(BaseModel):
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    method: str = "historical"


class HurstOut(BaseModel):
    hurst: float
    regime: str
    confidence: str


class RiskReportOut(BaseModel):
    var_historical: RiskAdvancedOut | None = None
    var_parametric: RiskAdvancedOut | None = None
    hurst: dict[str, HurstOut] | None = None


class OptimizationRequestIn(BaseModel):
    strategy: str = "mean_reversion"
    n_trials: int = 30
    train_size: int = 500
    test_size: int = 200


class OptimizationResultOut(BaseModel):
    best_params: dict
    best_sharpe: float
    n_trials: int


class HealthCheckOut(BaseModel):
    symbol: str
    healthy: bool
    total_rows: int
    checks: dict


class AuditEntryOut(BaseModel):
    timestamp: str
    event_type: str
    source: str
    details: dict


class DiagnosticsOut(BaseModel):
    avg_is_sharpe: float
    avg_oos_sharpe: float
    sharpe_degradation: float
    sharpe_stability: float
    worst_window_sharpe: float
    best_window_sharpe: float
    pct_profitable_windows: float
