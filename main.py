"""Entry point: fetch OHLCV data from Binance and store in database."""

import os
import sys
from datetime import datetime, timedelta, timezone

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.data.sources import BinanceSource
from src.data.pipeline import DataPipeline
from src.data.storage import Storage
from src.features import technical, statistical


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    load_dotenv()

    config = load_config()
    log_level = os.getenv("LOG_LEVEL", config["app"]["log_level"])
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info("=== Medal Trading System — Phase 1: Data Pipeline ===")

    # Init source
    source = BinanceSource(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        secret=os.getenv("BINANCE_SECRET", ""),
        testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
    )

    # Init storage
    database_url = os.getenv("DATABASE_URL", "")
    use_db = bool(database_url)

    storage = None
    if use_db:
        db_config = config.get("database", {})
        storage = Storage(
            database_url=database_url,
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10),
        )
        storage.init_db()
        logger.info("Database initialized")

    # Fetch data
    symbols = config["symbols"]
    timeframe = config["data"]["timeframe"]
    lookback_days = config["data"]["lookback_days"]
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    if storage:
        pipeline = DataPipeline(source=source, storage=storage)
        results = pipeline.ingest_multiple(symbols, timeframe=timeframe, since=since)
        for sym, count in results.items():
            logger.info("{}: {} rows stored", sym, count)
    else:
        logger.info("No DATABASE_URL set — fetching to memory only")
        data = source.fetch_multiple_ohlcv(symbols, timeframe=timeframe, since=since, limit=500)
        for sym, df in data.items():
            logger.info("{}: {} candles fetched", sym, len(df))

            # Compute features as demo
            df = technical.compute_all(df)
            df = statistical.compute_all(df)
            logger.info("{}: {} columns after features", sym, len(df.columns))
            logger.info("{} latest:\n{}", sym, df[["timestamp", "close", "rsi_14", "z_score_20"]].tail(5).to_string(index=False))

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
