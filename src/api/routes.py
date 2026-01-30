"""API route definitions."""

from fastapi import APIRouter

from src.api.schemas import (
    BacktestOut,
    DashboardOut,
    EquityPointOut,
    FillOut,
    OrderOut,
)

router = APIRouter(prefix="/api")


def _get_app_state():
    """Get shared app state — injected at startup."""
    from src.api.main import app_state
    return app_state


@router.get("/dashboard", response_model=DashboardOut)
def get_dashboard():
    """Full dashboard snapshot."""
    from src.monitoring.dashboard import render_dashboard
    state = _get_app_state()
    snapshot = state["collector"].collect()
    return render_dashboard(snapshot)


@router.get("/equity-curve", response_model=list[EquityPointOut])
def get_equity_curve():
    """Equity curve data points."""
    state = _get_app_state()
    df = state["collector"].equity_dataframe()
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    return [
        {
            "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"]),
            "equity": r["equity"],
            "drawdown_pct": r["drawdown_pct"],
            "daily_pnl": r["daily_pnl"],
        }
        for r in records
    ]


@router.get("/orders", response_model=list[OrderOut])
def get_orders():
    """All orders."""
    state = _get_app_state()
    oms = state.get("oms")
    if oms is None:
        return []
    return [
        {
            "order_id": o.order_id,
            "symbol": o.symbol,
            "side": o.side.value,
            "order_type": o.order_type.value,
            "quantity": o.quantity,
            "price": o.price,
            "status": o.status.value,
            "filled_quantity": o.filled_quantity,
            "filled_price": o.filled_price,
            "commission": o.commission,
            "created_at": o.created_at.isoformat(),
            "updated_at": o.updated_at.isoformat(),
        }
        for o in oms.get_all_orders()
    ]


@router.get("/orders/open", response_model=list[OrderOut])
def get_open_orders():
    """Open orders only."""
    state = _get_app_state()
    oms = state.get("oms")
    if oms is None:
        return []
    return [
        {
            "order_id": o.order_id,
            "symbol": o.symbol,
            "side": o.side.value,
            "order_type": o.order_type.value,
            "quantity": o.quantity,
            "price": o.price,
            "status": o.status.value,
            "filled_quantity": o.filled_quantity,
            "filled_price": o.filled_price,
            "commission": o.commission,
            "created_at": o.created_at.isoformat(),
            "updated_at": o.updated_at.isoformat(),
        }
        for o in oms.get_open_orders()
    ]


@router.get("/fills", response_model=list[FillOut])
def get_fills():
    """All fills."""
    state = _get_app_state()
    oms = state.get("oms")
    if oms is None:
        return []
    return [
        {
            "order_id": f.order_id,
            "symbol": f.symbol,
            "side": f.side.value,
            "quantity": f.quantity,
            "price": f.price,
            "commission": f.commission,
            "timestamp": f.timestamp.isoformat(),
        }
        for f in oms.get_fills()
    ]


@router.get("/backtest", response_model=BacktestOut | None)
def get_backtest():
    """Best backtest results."""
    state = _get_app_state()
    metrics = state.get("backtest_metrics")
    if metrics is None:
        return None
    return _metrics_to_dict(metrics)


@router.get("/backtest/all")
def get_all_backtests():
    """All backtest results by strategy."""
    state = _get_app_state()
    results = state.get("backtest_results", [])
    return [
        {
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "metrics": _metrics_to_dict(r["result"].metrics),
        }
        for r in results
    ]


@router.get("/backtest/walk-forward")
def get_walk_forward():
    """Walk-forward validation results."""
    state = _get_app_state()
    wf = state.get("walk_forward_result")
    if wf is None:
        return None
    return {
        "symbol": wf["symbol"],
        "strategy": wf["strategy"],
        "n_windows": wf["n_windows"],
        "oos_metrics": _metrics_to_dict(wf["oos_metrics"]),
        "windows": wf["windows"],
    }


@router.get("/strategy/params")
def get_strategy_params():
    """Get current strategy parameters for all strategies."""
    state = _get_app_state()
    strategies = state.get("strategies", {})
    return [
        {"name": name, "params": strat.get_params().to_dict()}
        for name, strat in strategies.items()
    ]


@router.get("/risk/advanced")
def get_risk_advanced():
    """Advanced risk metrics: VaR, CVaR, Hurst."""
    state = _get_app_state()
    risk_report = state.get("risk_report")
    if risk_report is None:
        return {"var_historical": None, "var_parametric": None, "hurst": None}
    return risk_report


@router.get("/risk/status")
def get_risk_status():
    """Full risk status including advanced metrics."""
    state = _get_app_state()
    risk_mgr = state.get("risk_manager")
    if risk_mgr is None:
        return {"basic": {}, "advanced": {}}

    basic = {}
    try:
        basic = risk_mgr.risk_summary()
    except Exception:
        pass

    advanced = state.get("risk_report", {})
    return {"basic": basic, "advanced": advanced}


@router.get("/health")
def get_health():
    """Data pipeline health check for all symbols."""
    state = _get_app_state()
    health = state.get("health_report")
    if health is None:
        return {"overall_healthy": True, "symbols": {}}
    return health


@router.get("/audit")
def get_audit(limit: int = 50, event_type: str | None = None):
    """Query audit trail entries."""
    state = _get_app_state()
    audit = state.get("audit")
    if audit is None:
        return []

    from src.monitoring.audit import AuditEventType
    et = None
    if event_type:
        try:
            et = AuditEventType(event_type)
        except ValueError:
            pass

    entries = audit.query(event_type=et, limit=limit)
    return [e.to_dict() for e in entries]


@router.get("/audit/summary")
def get_audit_summary():
    """Audit trail summary statistics."""
    state = _get_app_state()
    audit = state.get("audit")
    if audit is None:
        return {"total_entries": 0, "by_type": {}, "by_source": {}}
    return audit.summary()


@router.post("/optimization/run")
def run_optimization(request: dict):
    """Run hyperparameter optimization (background task).

    Body: {"strategy": "mean_reversion", "n_trials": 30}
    """
    state = _get_app_state()
    opt_result = state.get("optimization_result")
    if opt_result:
        return {
            "best_params": opt_result.best_params,
            "best_sharpe": opt_result.best_sharpe,
            "n_trials": opt_result.n_trials,
        }
    return {"status": "no optimization result available"}


@router.get("/optimization/result")
def get_optimization_result():
    """Get latest optimization result."""
    state = _get_app_state()
    opt_result = state.get("optimization_result")
    if opt_result is None:
        return None
    return {
        "best_params": opt_result.best_params,
        "best_sharpe": opt_result.best_sharpe,
        "n_trials": opt_result.n_trials,
        "all_trials": opt_result.all_trials[:20],  # Limit to avoid large responses
    }


@router.get("/backtest/diagnostics")
def get_diagnostics():
    """Walk-forward diagnostics (IS vs OOS analysis)."""
    state = _get_app_state()
    wf = state.get("walk_forward_result")
    if wf is None or wf.get("diagnostics") is None:
        return None
    d = wf["diagnostics"]
    return {
        "avg_is_sharpe": d.avg_is_sharpe,
        "avg_oos_sharpe": d.avg_oos_sharpe,
        "sharpe_degradation": d.sharpe_degradation,
        "sharpe_stability": d.sharpe_stability,
        "worst_window_sharpe": d.worst_window_sharpe,
        "best_window_sharpe": d.best_window_sharpe,
        "pct_profitable_windows": d.pct_profitable_windows,
        "regime_stats": [
            {
                "window_id": rs.window_id,
                "volatility": rs.volatility,
                "regime": rs.regime,
                "sharpe": rs.sharpe,
                "n_trades": rs.n_trades,
            }
            for rs in d.regime_stats
        ] if d.regime_stats else [],
        "monte_carlo": {
            "original_sharpe": d.monte_carlo.original_sharpe,
            "p_value": d.monte_carlo.p_value,
            "percentile_5": d.monte_carlo.percentile_5,
            "percentile_95": d.monte_carlo.percentile_95,
            "n_simulations": d.monte_carlo.n_simulations,
        } if d.monte_carlo else None,
    }


@router.get("/predictions/live")
def get_live_predictions():
    """Current pending predictions per symbol/strategy."""
    state = _get_app_state()
    engine = state.get("prediction_engine")
    if engine is None:
        return {}
    return engine.get_live_predictions()


@router.get("/predictions/accuracy")
def get_predictions_accuracy():
    """Rolling accuracy metrics per strategy."""
    state = _get_app_state()
    engine = state.get("prediction_engine")
    if engine is None:
        return {"overall": {}, "per_strategy": {}, "ranking": []}
    return engine.get_accuracy_summary()


@router.get("/predictions/history")
def get_predictions_history(
    symbol: str | None = None,
    strategy: str | None = None,
    limit: int = 100,
):
    """Prediction history with outcomes."""
    state = _get_app_state()
    tracker = state.get("prediction_tracker")
    if tracker is None:
        return []
    preds = tracker.get_history(symbol=symbol, strategy=strategy, limit=limit)
    return [p.to_dict() for p in preds]


@router.get("/backtest/continuous")
def get_continuous_backtest():
    """Latest continuous backtest results."""
    state = _get_app_state()
    cb = state.get("continuous_backtester")
    if cb is None:
        return []
    return cb.get_latest_results()


@router.get("/backtest/degradation")
def get_degradation_alerts():
    """Strategy health/degradation alerts."""
    state = _get_app_state()
    cb = state.get("continuous_backtester")
    if cb is None:
        return []
    return cb.get_degradation_alerts()


@router.get("/backtest/streaming/status")
def get_streaming_status():
    """Current state of the streaming backtest."""
    state = _get_app_state()
    manager = state.get("streaming_manager")
    if manager is None:
        return {"active": False, "current_strategy": "", "current_symbol": "", "progress_pct": 0}
    return manager.get_status()


@router.get("/backtest/streaming/results")
def get_streaming_results():
    """Results from the last completed streaming cycle."""
    state = _get_app_state()
    manager = state.get("streaming_manager")
    if manager is None:
        return []
    return manager.get_all_results()


def _metrics_to_dict(metrics) -> dict:
    return {
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_duration": metrics.max_drawdown_duration,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "avg_trade_return": metrics.avg_trade_return,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "expectancy": metrics.expectancy,
    }
