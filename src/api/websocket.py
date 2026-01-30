"""WebSocket endpoint for real-time dashboard updates."""

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

ws_router = APIRouter()

# Connected clients
_clients: set[WebSocket] = set()


@ws_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Stream dashboard updates to connected clients."""
    await ws.accept()
    _clients.add(ws)
    logger.info("WebSocket client connected ({} total)", len(_clients))

    try:
        while True:
            # Send periodic updates
            from src.api.main import app_state

            collector = app_state.get("collector")
            if collector:
                try:
                    from src.monitoring.dashboard import render_dashboard
                    snapshot = collector.collect()
                    data = render_dashboard(snapshot)
                    await ws.send_json({
                        "type": "dashboard",
                        "data": _serialize(data),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    pass

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)
        logger.info("WebSocket client disconnected ({} remain)", len(_clients))


async def broadcast(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    if not _clients:
        return

    message = json.dumps({
        "type": event_type,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    disconnected: set[WebSocket] = set()
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    _clients -= disconnected


def _serialize(obj: object) -> dict:
    """Convert Pydantic model or dict to JSON-safe dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
