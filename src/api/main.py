"""FastAPI application for Medal Trading System."""

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.data.sources import BinanceSource
from src.data.pipeline import DataPipeline
from src.data.storage import Storage
from src.execution.oms import OrderManagementSystem, OrderSide
from src.execution.scheduler import TradingScheduler
from src.features import technical, statistical
from src.monitoring.alerts import AlertManager, create_default_rules
from src.monitoring.dashboard import DashboardDataCollector
from src.risk.portfolio import PortfolioRiskManager
from src.strategies.base import Signal
from src.strategies.mean_reversion import MeanReversionStrategy, MeanReversionParams
from src.strategies.pairs_trading import (
    PairsTradingStrategy,
    PairsTradingParams,
    build_pairs_dataframe,
    backtest_pair,
)
from src.strategies.mean_reversion import backtest_mean_reversion
from src.strategies.ml_ensemble import MLEnsembleStrategy, MLEnsembleParams, backtest_ml_ensemble
from src.strategies.aggregator import SignalAggregator, AggregatorParams, backtest_aggregated
from src.backtester.engine import BacktestEngine
from src.backtester.walk_forward import WalkForwardValidator
from src.monitoring.audit import AuditTrail, AuditEventType
from src.risk.advanced_metrics import AdvancedRiskMetrics
from src.prediction.tracker import PredictionTracker
from src.prediction.engine import PredictionEngine
from src.backtester.continuous import ContinuousBacktester
from src.backtester.stream_manager import StreamingBacktestManager

# Shared state accessible by routes
app_state: dict = {}


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_all_data(source: BinanceSource, config: dict) -> dict:
    """Fetch OHLCV data for all symbols using full history pagination."""
    symbols = config["symbols"]
    timeframe = config["data"]["timeframe"]
    lookback_days = config["data"]["lookback_days"]
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    data: dict = {}
    for sym in symbols:
        try:
            df = source.fetch_full_history(
                sym, timeframe=timeframe, since=since,
                limit_per_request=1000, max_candles=5000,
            )
            data[sym] = df
            logger.info("Fetched {}: {} candles", sym, len(df))
        except Exception as e:
            logger.error("Failed to fetch {}: {}", sym, e)
    return data


def run_backtests(ohlcv_data: dict, config: dict) -> list[dict]:
    """Run backtests on available data and return results."""
    results = []
    initial_capital = 10000.0

    # 1. Mean Reversion backtests on individual symbols
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        if symbol not in ohlcv_data or len(ohlcv_data[symbol]) < 120:
            continue
        try:
            df = ohlcv_data[symbol].copy()
            df = technical.compute_all(df)
            result = backtest_mean_reversion(
                data=df,
                initial_capital=initial_capital,
                commission_rate=0.001,
                slippage_rate=0.001,
                position_size_pct=0.03,
            )
            results.append({
                "strategy": "MeanReversion",
                "symbol": symbol,
                "result": result,
            })
            logger.info(
                "MR Backtest {}: Sharpe={:.2f}, Return={:.2%}, Trades={}",
                symbol, result.metrics.sharpe_ratio,
                result.metrics.total_return, result.metrics.total_trades,
            )
        except Exception as e:
            logger.error("MR Backtest {} failed: {}", symbol, e)

    # 2. Pairs Trading backtests
    pairs = config.get("pairs", [])
    for pair in pairs:
        sym_a, sym_b = pair[0], pair[1]
        if sym_a not in ohlcv_data or sym_b not in ohlcv_data:
            continue
        if len(ohlcv_data[sym_a]) < 200 or len(ohlcv_data[sym_b]) < 200:
            continue
        try:
            result = backtest_pair(
                df_a=ohlcv_data[sym_a],
                df_b=ohlcv_data[sym_b],
                initial_capital=initial_capital,
                commission_rate=0.001,
                slippage_rate=0.001,
                position_size_pct=0.03,
            )
            results.append({
                "strategy": "PairsTrading",
                "symbol": f"{sym_a}/{sym_b}",
                "result": result,
            })
            logger.info(
                "Pairs Backtest {}/{}: Sharpe={:.2f}, Return={:.2%}, Trades={}",
                sym_a, sym_b, result.metrics.sharpe_ratio,
                result.metrics.total_return, result.metrics.total_trades,
            )
        except Exception as e:
            logger.error("Pairs Backtest {}/{} failed: {}", sym_a, sym_b, e)

    # 3. ML Ensemble backtests on individual symbols
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        if symbol not in ohlcv_data or len(ohlcv_data[symbol]) < 2500:
            logger.info("ML Ensemble {}: skipped (need 2500 bars, have {})",
                        symbol, len(ohlcv_data.get(symbol, [])))
            continue
        try:
            result = backtest_ml_ensemble(
                data=ohlcv_data[symbol].copy(),
                initial_capital=initial_capital,
                commission_rate=0.001,
                slippage_rate=0.001,
                position_size_pct=0.03,
            )
            results.append({
                "strategy": "MLEnsemble",
                "symbol": symbol,
                "result": result,
            })
            logger.info(
                "ML Backtest {}: Sharpe={:.2f}, Return={:.2%}, Trades={}",
                symbol, result.metrics.sharpe_ratio,
                result.metrics.total_return, result.metrics.total_trades,
            )
        except Exception as e:
            logger.error("ML Backtest {} failed: {}", symbol, e)

    # 4. Aggregated strategy backtest (MeanReversion + ML on BTC)
    btc_symbol = "BTC/USDT"
    if btc_symbol in ohlcv_data and len(ohlcv_data[btc_symbol]) >= 2500:
        try:
            df = ohlcv_data[btc_symbol].copy()

            mr_strategy = MeanReversionStrategy()
            ml_strategy = MLEnsembleStrategy()

            # Prepare with both strategies so all features are present
            df = mr_strategy.prepare(df)
            df = ml_strategy.prepare(df)

            result = backtest_aggregated(
                data=df,
                strategies={
                    "mean_reversion": mr_strategy,
                    "ml_ensemble": ml_strategy,
                },
                weights={"mean_reversion": 0.5, "ml_ensemble": 0.5},
                initial_capital=initial_capital,
                commission_rate=0.001,
                slippage_rate=0.001,
                position_size_pct=0.03,
            )
            results.append({
                "strategy": "Aggregator",
                "symbol": btc_symbol,
                "result": result,
            })
            logger.info(
                "Aggregator Backtest {}: Sharpe={:.2f}, Return={:.2%}, Trades={}",
                btc_symbol, result.metrics.sharpe_ratio,
                result.metrics.total_return, result.metrics.total_trades,
            )
        except Exception as e:
            logger.error("Aggregator Backtest failed: {}", e)

    # 5. Walk-forward validation on MeanReversion (BTC)
    if btc_symbol in ohlcv_data and len(ohlcv_data[btc_symbol]) >= 1000:
        try:
            df = ohlcv_data[btc_symbol].copy()
            df = technical.compute_all(df)
            mr = MeanReversionStrategy()
            prepared = mr.prepare(df)

            engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_rate=0.001,
                slippage_rate=0.001,
                position_size_pct=0.03,
            )
            validator = WalkForwardValidator(
                engine=engine,
                train_size=500,
                test_size=200,
                step_size=200,
            )
            wf_result = validator.run(
                data=prepared,
                signal_fn=mr.generate_engine_signal,
                symbol=btc_symbol,
            )
            # Store walk-forward results separately
            app_state["walk_forward_result"] = {
                "symbol": btc_symbol,
                "strategy": "MeanReversion",
                "n_windows": len(wf_result.windows),
                "oos_metrics": wf_result.oos_metrics,
                "windows": [
                    {
                        "window_id": w.window_id,
                        "train_sharpe": w.train_result.metrics.sharpe_ratio if w.train_result else None,
                        "test_sharpe": w.test_result.metrics.sharpe_ratio if w.test_result else None,
                        "test_return": w.test_result.metrics.total_return if w.test_result else None,
                        "test_trades": w.test_result.metrics.total_trades if w.test_result else None,
                    }
                    for w in wf_result.windows
                ],
            }
            logger.info(
                "Walk-forward {}: {} windows, OOS Sharpe={:.2f}, OOS Return={:.2%}",
                btc_symbol, len(wf_result.windows),
                wf_result.oos_metrics.sharpe_ratio,
                wf_result.oos_metrics.total_return,
            )
        except Exception as e:
            logger.error("Walk-forward validation failed: {}", e)

    return results


def populate_dashboard_from_backtests(
    backtest_results: list[dict],
    collector: DashboardDataCollector,
    risk_manager: PortfolioRiskManager,
) -> None:
    """Feed backtest equity curves into the dashboard collector."""
    if not backtest_results:
        return

    # Use the best backtest (highest Sharpe) as the primary display
    best = max(backtest_results, key=lambda r: r["result"].metrics.sharpe_ratio)
    result = best["result"]

    # Store best backtest metrics in app_state
    app_state["backtest_metrics"] = result.metrics
    app_state["backtest_results"] = backtest_results

    # Feed equity curve to collector
    equity_curve = result.equity_curve
    initial = equity_curve.iloc[0] if len(equity_curve) > 0 else 10000.0
    peak = initial

    for i, eq in enumerate(equity_curve):
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        daily_pnl = eq - equity_curve.iloc[max(0, i - 24)] if i >= 24 else 0
        collector.record_equity(equity=eq, drawdown_pct=dd, daily_pnl=daily_pnl)

    # Update risk manager with final equity
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else initial
    risk_manager.update_equity(final_equity)

    logger.info(
        "Dashboard populated: best={} ({}), Sharpe={:.2f}, Final equity=${:.2f}",
        best["strategy"], best["symbol"],
        result.metrics.sharpe_ratio, final_equity,
    )


def create_strategy_job(
    source: BinanceSource,
    config: dict,
    oms: OrderManagementSystem,
    risk_manager: PortfolioRiskManager,
    alert_manager: AlertManager,
    collector: DashboardDataCollector,
) -> callable:
    """Create a strategy execution job for the scheduler."""

    def run_strategy_cycle():
        """Fetch latest data, generate signals, create paper orders."""
        try:
            symbols = config["symbols"][:4]  # Top 4 symbols
            since = datetime.now(timezone.utc) - timedelta(days=30)
            data = source.fetch_multiple_ohlcv(symbols, timeframe="1h", since=since, limit=500)

            for symbol, df in data.items():
                if len(df) < 120:
                    continue

                df = technical.compute_all(df)
                df = statistical.compute_all(df)

                # Use Aggregator if enough data for ML, otherwise MeanReversion only
                mr_strategy = MeanReversionStrategy()
                prepared = mr_strategy.prepare(df)

                if len(df) >= 2500:
                    ml_strategy = MLEnsembleStrategy()
                    prepared = ml_strategy.prepare(prepared)
                    aggregator = SignalAggregator()
                    aggregator.register("mean_reversion", mr_strategy, 0.5)
                    aggregator.register("ml_ensemble", ml_strategy, 0.5)
                    signal = aggregator.aggregate(prepared, len(prepared) - 1)
                else:
                    signal = mr_strategy.generate_signal(prepared, len(prepared) - 1)

                if signal.value in ("BUY", "SELL"):
                    side = OrderSide.BUY if signal.value == "BUY" else OrderSide.SELL
                    price = float(df["close"].iloc[-1])
                    qty = (risk_manager.equity * 0.03) / price

                    order = oms.create_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        metadata={"strategy": "MeanReversion", "signal": signal.value},
                    )
                    # Simulate instant fill for paper trading
                    oms.mark_submitted(order.order_id, f"PAPER-{order.order_id}")
                    commission = price * qty * 0.001
                    oms.mark_filled(order.order_id, price, qty, commission)

                    logger.info(
                        "Paper trade: {} {} {} qty={:.6f} @ ${:.2f}",
                        signal.value, symbol, order.order_id, qty, price,
                    )

            # Update equity from paper P&L
            total_commission = oms.total_commission()
            net_fills_value = sum(
                f.price * f.quantity * (1 if f.side == OrderSide.BUY else -1)
                for f in oms.get_fills()
            )
            equity = risk_manager.equity - total_commission
            risk_manager.update_equity(equity)

            # Record equity point
            dd = (risk_manager.peak_equity - equity) / risk_manager.peak_equity if risk_manager.peak_equity > 0 else 0
            collector.record_equity(equity=equity, drawdown_pct=dd)

            # Evaluate alert rules
            alert_manager.evaluate({
                "drawdown_pct": dd * 100,
                "daily_loss_pct": risk_manager.daily_loss * 100 if hasattr(risk_manager, 'daily_loss') else 0,
                "open_positions": len(risk_manager.positions),
            })

            logger.info(
                "Strategy cycle complete: equity=${:.2f}, orders={}, fills={}",
                equity, oms.total_orders, oms.total_fills,
            )

        except Exception as e:
            logger.error("Strategy cycle failed: {}", e)

    return run_strategy_cycle


def create_data_update_job(source: BinanceSource, storage, config: dict) -> callable:
    """Create a data refresh job."""

    def update_data():
        try:
            symbols = config["symbols"]
            since = datetime.now(timezone.utc) - timedelta(hours=2)
            data = source.fetch_multiple_ohlcv(symbols, timeframe="1h", since=since, limit=10)
            for sym, df in data.items():
                logger.info("Data update {}: {} new candles", sym, len(df))
            if storage:
                pipeline = DataPipeline(source=source, storage=storage)
                pipeline.ingest_multiple(symbols, timeframe="1h", since=since)
        except Exception as e:
            logger.error("Data update failed: {}", e)

    return update_data


def create_risk_check_job(
    risk_manager: PortfolioRiskManager,
    alert_manager: AlertManager,
    collector: DashboardDataCollector,
) -> callable:
    """Create a periodic risk monitoring job."""

    def check_risk():
        try:
            summary = risk_manager.risk_summary()
            dd = summary.get("drawdown", 0)
            collector.record_equity(
                equity=risk_manager.equity,
                drawdown_pct=dd,
            )
            alert_manager.evaluate({
                "drawdown_pct": dd * 100,
                "daily_loss_pct": summary.get("daily_loss", 0) * 100,
                "open_positions": summary.get("positions", 0),
            })
            logger.info("Risk check: equity=${:.2f}, DD={:.2%}", risk_manager.equity, dd)
        except Exception as e:
            logger.error("Risk check failed: {}", e)

    return check_risk


def startup_sequence(
    source: BinanceSource,
    storage,
    config: dict,
    oms: OrderManagementSystem,
    risk_manager: PortfolioRiskManager,
    alert_manager: AlertManager,
    scheduler: TradingScheduler,
    collector: DashboardDataCollector,
) -> None:
    """Full startup: fetch data, run backtests, start scheduler."""
    logger.info("=== Startup sequence: fetching data ===")

    # 1. Fetch all OHLCV data
    ohlcv_data = fetch_all_data(source, config)
    app_state["ohlcv_data"] = ohlcv_data

    # 2. Also persist to DB if available
    if storage:
        symbols = config["symbols"]
        since = datetime.now(timezone.utc) - timedelta(days=config["data"]["lookback_days"])
        pipeline = DataPipeline(source=source, storage=storage)
        pipeline.ingest_multiple(symbols, timeframe=config["data"]["timeframe"], since=since)

    # 3. Run backtests
    logger.info("=== Running backtests ===")
    backtest_results = run_backtests(ohlcv_data, config)
    populate_dashboard_from_backtests(backtest_results, collector, risk_manager)

    # 3b. Compute advanced risk metrics
    try:
        arm = AdvancedRiskMetrics()
        price_dict = {sym: df["close"] for sym, df in ohlcv_data.items() if len(df) > 50}
        best = max(backtest_results, key=lambda r: r["result"].metrics.sharpe_ratio) if backtest_results else None
        equity_series = pd.Series(best["result"].equity_curve) if best else pd.Series(dtype=float)
        if len(equity_series) > 30:
            report = arm.portfolio_risk_report(equity_series, price_dict=price_dict)
            app_state["risk_report"] = report
            logger.info("Advanced risk report computed")
    except Exception as e:
        logger.error("Advanced risk computation failed: {}", e)

    # 3c. Store walk-forward diagnostics
    wf = app_state.get("walk_forward_result")
    if wf and hasattr(wf.get("oos_metrics", None), "sharpe_ratio"):
        audit = app_state.get("audit")
        if audit:
            audit.log(AuditEventType.SYSTEM_START, "WalkForward", {
                "oos_sharpe": wf["oos_metrics"].sharpe_ratio,
                "n_windows": wf["n_windows"],
            })

    # 4. Start scheduler with periodic jobs
    logger.info("=== Starting scheduler ===")

    strategy_job = create_strategy_job(
        source, config, oms, risk_manager, alert_manager, collector,
    )
    data_job = create_data_update_job(source, storage, config)
    risk_job = create_risk_check_job(risk_manager, alert_manager, collector)

    scheduler.add_job("strategy_cycle", strategy_job, hours=1, start_now=True)
    scheduler.add_job("data_update", data_job, hours=1, start_now=False)
    scheduler.add_job("risk_check", risk_job, minutes=15, start_now=True)

    # 4b. Set baselines for continuous backtester and add prediction jobs
    cb = app_state.get("continuous_backtester")
    if cb and backtest_results:
        for r in backtest_results:
            cb.set_baseline(r["strategy"], r["symbol"], r["result"].metrics)

    pred_engine = app_state.get("prediction_engine")

    def prediction_cycle():
        if pred_engine:
            pred_engine.run_cycle()

    def continuous_backtest_cycle():
        ohlcv = app_state.get("ohlcv_data", {})
        if cb and ohlcv:
            cb.run_cycle(ohlcv)

    scheduler.add_job("prediction_cycle", prediction_cycle, minutes=15, start_now=False)
    scheduler.add_job("continuous_backtest", continuous_backtest_cycle, hours=4, start_now=False)

    scheduler.start()
    logger.info("=== Startup sequence complete ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize trading system components on startup."""
    load_dotenv()
    config = load_config()

    log_level = os.getenv("LOG_LEVEL", config["app"]["log_level"])
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info("=== Medal Trading System — API Starting ===")

    # Data source
    # Use real Binance data (public OHLCV doesn't require API keys)
    source = BinanceSource(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        secret=os.getenv("BINANCE_SECRET", ""),
        testnet=os.getenv("BINANCE_TESTNET", "false").lower() == "true",
    )

    # Storage
    database_url = os.getenv("DATABASE_URL", "")
    storage = None
    if database_url:
        db_config = config.get("database", {})
        storage = Storage(
            database_url=database_url,
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10),
        )
        storage.init_db()
        logger.info("Database initialized")

    # Core components
    oms = OrderManagementSystem()
    alert_manager = AlertManager()
    for rule in create_default_rules():
        alert_manager.add_rule(rule)

    risk_manager = PortfolioRiskManager(initial_equity=10000.0)
    scheduler = TradingScheduler()

    collector = DashboardDataCollector(
        oms=oms,
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        scheduler=scheduler,
    )

    # Audit trail
    audit = AuditTrail(log_file="logs/audit.jsonl")
    audit.log(AuditEventType.SYSTEM_START, "API", {"version": "1.0"})

    # Populate state
    app_state["oms"] = oms
    app_state["alert_manager"] = alert_manager
    app_state["risk_manager"] = risk_manager
    app_state["scheduler"] = scheduler
    app_state["collector"] = collector
    app_state["source"] = source
    app_state["storage"] = storage
    app_state["config"] = config
    app_state["audit"] = audit

    # Prediction system
    prediction_tracker = PredictionTracker(max_history=5000)
    prediction_strategies = {
        "mean_reversion": MeanReversionStrategy(),
        "ml_ensemble": MLEnsembleStrategy(),
    }
    prediction_engine = PredictionEngine(
        source=source,
        strategies=prediction_strategies,
        tracker=prediction_tracker,
        symbols=config["symbols"][:4],
        timeframe=config["data"]["timeframe"],
    )
    continuous_backtester = ContinuousBacktester(
        strategies={"mean_reversion": MeanReversionStrategy()},
        symbols=config["symbols"][:4],
    )
    app_state["prediction_tracker"] = prediction_tracker
    app_state["prediction_engine"] = prediction_engine
    app_state["continuous_backtester"] = continuous_backtester

    # Streaming backtest manager (runs continuously via WebSocket)
    from src.api.websocket import broadcast as ws_broadcast
    streaming_manager = StreamingBacktestManager(
        strategies={"mean_reversion": MeanReversionStrategy()},
        symbols=config["symbols"][:4],
        source=source,
        broadcast_fn=ws_broadcast,
        cycle_delay_minutes=5,
        initial_capital=10_000.0,
    )
    app_state["streaming_manager"] = streaming_manager

    # Run startup in background thread (backtests + scheduler)
    thread = threading.Thread(
        target=startup_sequence,
        args=(source, storage, config, oms, risk_manager, alert_manager, scheduler, collector),
        daemon=True,
    )
    thread.start()

    # Start streaming backtest loop (async background task)
    await streaming_manager.start()

    logger.info("=== API Ready (startup running in background) ===")
    yield

    # Shutdown
    await streaming_manager.stop()
    scheduler.stop()
    logger.info("=== API Shutdown ===")


app = FastAPI(title="Medal Trading System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes import router  # noqa: E402
from src.api.websocket import ws_router  # noqa: E402
app.include_router(router)
app.include_router(ws_router)
