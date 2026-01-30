"""Trading scheduler for automated 24/7 crypto operation.

Uses APScheduler to run periodic jobs:
    - data_update: fetch latest OHLCV data
    - strategy_run: generate signals and execute trades
    - reconciliation: verify positions match expectations
    - risk_check: monitor portfolio risk state

Crypto runs 24/7 with configurable intervals (default: 1h strategy, 4h recon).
"""

from datetime import datetime, timezone
from typing import Any, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


class TradingJob:
    """Wraps a callable as a scheduler job with error handling."""

    def __init__(self, name: str, func: Callable[..., Any], **kwargs: Any) -> None:
        self.name = name
        self.func = func
        self.kwargs = kwargs
        self.last_run: datetime | None = None
        self.run_count: int = 0
        self.error_count: int = 0
        self.last_error: str | None = None

    def execute(self) -> None:
        """Run the job with error handling and logging."""
        self.last_run = datetime.now(timezone.utc)
        self.run_count += 1
        logger.info("Scheduler: Running job '{}' (run #{})", self.name, self.run_count)
        try:
            self.func(**self.kwargs)
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error("Scheduler: Job '{}' failed: {}", self.name, e)

    def status(self) -> dict:
        return {
            "name": self.name,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


class TradingScheduler:
    """Manages periodic trading jobs for 24/7 crypto operation.

    Usage:
        scheduler = TradingScheduler()
        scheduler.add_job("data_update", fetch_data, hours=1)
        scheduler.add_job("strategy", run_strategy, hours=1)
        scheduler.add_job("reconciliation", reconcile, hours=4)
        scheduler.add_job("risk_check", check_risk, minutes=15)
        scheduler.start()

    Args:
        timezone_str: Timezone for job scheduling (default UTC).
    """

    def __init__(self, timezone_str: str = "UTC") -> None:
        self._scheduler = BackgroundScheduler(timezone=timezone_str)
        self._jobs: dict[str, TradingJob] = {}
        self._running = False

    def add_job(
        self,
        name: str,
        func: Callable[..., Any],
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        start_now: bool = True,
        **kwargs: Any,
    ) -> None:
        """Register a periodic job.

        Args:
            name: Unique job identifier.
            func: Callable to run.
            hours: Interval hours.
            minutes: Interval minutes.
            seconds: Interval seconds.
            start_now: Run immediately on start.
            **kwargs: Arguments passed to func.
        """
        job = TradingJob(name, func, **kwargs)
        self._jobs[name] = job

        trigger = IntervalTrigger(hours=hours, minutes=minutes, seconds=seconds)
        self._scheduler.add_job(
            job.execute,
            trigger=trigger,
            id=name,
            name=name,
            replace_existing=True,
        )
        logger.info(
            "Scheduler: Added job '{}' (every {}h {}m {}s)",
            name, hours, minutes, seconds,
        )

        if start_now:
            job.execute()

    def remove_job(self, name: str) -> bool:
        """Remove a registered job.

        Returns:
            True if removed.
        """
        if name in self._jobs:
            try:
                self._scheduler.remove_job(name)
            except Exception:
                pass
            del self._jobs[name]
            logger.info("Scheduler: Removed job '{}'", name)
            return True
        return False

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        self._scheduler.start()
        self._running = True
        logger.info("Scheduler: Started with {} jobs", len(self._jobs))

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._running:
            return
        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Scheduler: Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def run_job_now(self, name: str) -> bool:
        """Manually trigger a job immediately.

        Returns:
            True if executed.
        """
        job = self._jobs.get(name)
        if job is None:
            logger.warning("Scheduler: Job '{}' not found", name)
            return False
        job.execute()
        return True

    def status(self) -> dict:
        """Get status of all jobs.

        Returns:
            Dict with scheduler state and per-job status.
        """
        return {
            "running": self._running,
            "total_jobs": len(self._jobs),
            "jobs": {name: job.status() for name, job in self._jobs.items()},
        }
