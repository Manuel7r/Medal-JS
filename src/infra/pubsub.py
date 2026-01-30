"""Redis Pub/Sub for inter-component communication.

Channels:
    - medal:signals    -> Strategy signals for execution
    - medal:orders     -> Order events (created, filled, cancelled)
    - medal:alerts     -> Risk/system alerts
    - medal:equity     -> Equity updates for real-time dashboard

Usage:
    bus = EventBus(redis_url="redis://localhost:6379")
    bus.publish("medal:signals", {"symbol": "BTC/USDT", "signal": "BUY"})

    # In another component:
    bus.subscribe("medal:signals", handler_fn)
    bus.listen()  # blocking
"""

import json
import os
import threading
from typing import Any, Callable

from loguru import logger

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class EventBus:
    """Redis-backed event bus for system components.

    Provides publish/subscribe communication between:
        - Data pipeline -> Strategies
        - Strategies -> Execution
        - Execution -> Dashboard
        - Risk -> Alerts
    """

    CHANNELS = [
        "medal:signals",
        "medal:orders",
        "medal:alerts",
        "medal:equity",
        "medal:data",
        "medal:risk",
    ]

    def __init__(self, redis_url: str | None = None) -> None:
        if not HAS_REDIS:
            logger.warning("EventBus: redis package not installed")
            self._client = None
            return

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info("EventBus: Connected to Redis at {}", url)
        except Exception as e:
            logger.warning("EventBus: Redis connection failed: {}. Events will be logged only.", e)
            self._client = None

        self._pubsub = None
        self._handlers: dict[str, list[Callable]] = {}
        self._thread: threading.Thread | None = None

    def publish(self, channel: str, data: dict[str, Any]) -> bool:
        """Publish an event to a channel.

        Args:
            channel: Redis channel name.
            data: Event payload (dict, will be JSON-serialized).

        Returns:
            True if published.
        """
        if self._client is None:
            logger.debug("EventBus: No Redis, logging event: {} -> {}", channel, data)
            return False

        try:
            self._client.publish(channel, json.dumps(data))
            return True
        except Exception as e:
            logger.error("EventBus: Publish failed on {}: {}", channel, e)
            return False

    def subscribe(self, channel: str, handler: Callable[[dict], None]) -> None:
        """Register a handler for a channel.

        Args:
            channel: Channel to subscribe to.
            handler: Callable(data_dict) to invoke on each message.
        """
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

    def listen(self, daemon: bool = True) -> None:
        """Start listening for messages on subscribed channels.

        Args:
            daemon: Run as daemon thread (dies with main process).
        """
        if self._client is None or not self._handlers:
            return

        self._pubsub = self._client.pubsub()
        self._pubsub.subscribe(**{
            ch: self._dispatch for ch in self._handlers
        })

        self._thread = threading.Thread(target=self._listen_loop, daemon=daemon)
        self._thread.start()
        logger.info("EventBus: Listening on {} channels", len(self._handlers))

    def _listen_loop(self) -> None:
        """Internal message loop."""
        if self._pubsub is None:
            return
        for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                data = {"raw": message["data"]}
            self._dispatch_to_handlers(channel, data)

    def _dispatch(self, message: Any) -> None:
        """PubSub callback."""
        pass

    def _dispatch_to_handlers(self, channel: str, data: dict) -> None:
        """Route message to registered handlers."""
        handlers = self._handlers.get(channel, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error("EventBus: Handler error on {}: {}", channel, e)

    def stop(self) -> None:
        """Stop listening."""
        if self._pubsub:
            self._pubsub.unsubscribe()
            self._pubsub.close()
        logger.info("EventBus: Stopped")

    # --- Cache helpers ---

    def cache_set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set a cached value with TTL.

        Args:
            key: Cache key.
            value: Value (will be JSON-serialized).
            ttl: Time-to-live in seconds.
        """
        if self._client is None:
            return False
        try:
            self._client.setex(key, ttl, json.dumps(value))
            return True
        except Exception:
            return False

    def cache_get(self, key: str) -> Any | None:
        """Get a cached value."""
        if self._client is None:
            return None
        try:
            val = self._client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None
