"""Manager for continuous streaming backtests."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable, Awaitable

from loguru import logger

from src.backtester.streaming import StreamingBacktestRunner
from src.features import technical


class StreamingBacktestManager:
    """Runs streaming backtests in a continuous loop.

    For each strategy+symbol combination, runs a bar-by-bar backtest
    broadcasting progress via WebSocket, then waits before repeating.
    """

    def __init__(
        self,
        strategies: dict,
        symbols: list[str],
        source,
        broadcast_fn: Callable[[str, dict], Awaitable[None]],
        cycle_delay_minutes: int = 30,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.strategies = strategies
        self.symbols = symbols
        self.source = source
        self.broadcast = broadcast_fn
        self.cycle_delay = cycle_delay_minutes * 60
        self.initial_capital = initial_capital

        self.runner = StreamingBacktestRunner(
            broadcast_fn=broadcast_fn,
            initial_capital=initial_capital,
        )

        self._task: asyncio.Task | None = None
        self._current_strategy: str = ""
        self._current_symbol: str = ""
        self._cycle_count: int = 0
        self._last_results: list[dict] = []

    async def start(self) -> None:
        """Start the continuous loop as a background task."""
        self._task = asyncio.create_task(self._run_loop())
        logger.info("StreamingBacktestManager started")

    async def stop(self) -> None:
        """Cancel the background task."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("StreamingBacktestManager stopped")

    async def _run_loop(self) -> None:
        """Main loop: fetch data → stream backtests → wait → repeat."""
        # Wait for startup data to be available
        await asyncio.sleep(10)

        while True:
            try:
                self._cycle_count += 1
                cycle_results: list[dict] = []

                logger.info("=== Streaming backtest cycle #{} starting ===", self._cycle_count)

                await self.broadcast("backtest_cycle_start", {
                    "cycle": self._cycle_count,
                    "strategies": list(self.strategies.keys()),
                    "symbols": self.symbols,
                })

                # Fetch latest data
                ohlcv_data = await self._fetch_data()

                for strategy_name, strategy in self.strategies.items():
                    for symbol in self.symbols:
                        if symbol not in ohlcv_data or len(ohlcv_data[symbol]) < 120:
                            continue

                        self._current_strategy = strategy_name
                        self._current_symbol = symbol

                        try:
                            df = ohlcv_data[symbol].copy()
                            df = technical.compute_all(df)
                            df = strategy.prepare(df)

                            metrics = await self.runner.run_streaming(
                                data=df,
                                signal_fn=strategy.generate_engine_signal,
                                symbol=symbol,
                                strategy=strategy_name,
                                atr_stop_multiplier=2.5,
                                atr_tp_multiplier=4.0,
                            )

                            cycle_results.append({
                                "strategy": strategy_name,
                                "symbol": symbol,
                                "metrics": metrics,
                            })
                        except Exception as e:
                            logger.error("Streaming backtest {}/{} failed: {}", strategy_name, symbol, e)

                self._last_results = cycle_results
                self._current_strategy = ""
                self._current_symbol = ""

                await self.broadcast("backtest_cycle_complete", {
                    "cycle": self._cycle_count,
                    "results": cycle_results,
                })

                logger.info(
                    "=== Streaming cycle #{} complete: {} results. Next in {} min ===",
                    self._cycle_count, len(cycle_results), self.cycle_delay // 60,
                )

                # Wait before next cycle
                await asyncio.sleep(self.cycle_delay)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Streaming loop error: {}", e)
                await asyncio.sleep(60)

    async def _fetch_data(self) -> dict:
        """Fetch OHLCV data for all symbols."""
        since = datetime.now(timezone.utc) - timedelta(days=60)
        data: dict = {}

        for symbol in self.symbols:
            try:
                df = self.source.fetch_full_history(
                    symbol, timeframe="1h", since=since,
                    limit_per_request=1000, max_candles=2000,
                )
                data[symbol] = df
                logger.debug("Streaming fetch {}: {} candles", symbol, len(df))
            except Exception as e:
                logger.error("Streaming fetch {} failed: {}", symbol, e)

        return data

    def get_status(self) -> dict:
        """Current manager status."""
        runner_status = self.runner.get_status()
        return {
            "active": runner_status["running"],
            "current_strategy": self._current_strategy,
            "current_symbol": self._current_symbol,
            "cycle_count": self._cycle_count,
            "progress_pct": runner_status["progress_pct"],
            "bar_index": runner_status["bar_index"],
            "total_bars": runner_status["total_bars"],
            "equity": runner_status["equity"],
            "total_trades": runner_status["total_trades"],
            "metrics": runner_status["metrics"],
        }

    def get_all_results(self) -> list[dict]:
        """Results from the last completed cycle."""
        return self._last_results
