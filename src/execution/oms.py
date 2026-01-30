"""Order Management System (OMS).

Manages the full order lifecycle: creation, submission, fill tracking,
cancellation, and position reconciliation. All events are logged.

Order states:
    PENDING   -> SUBMITTED -> FILLED | PARTIALLY_FILLED | CANCELLED | REJECTED
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a single order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None  # Required for LIMIT, None for MARKET
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    exchange_order_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Represents a trade fill."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderManagementSystem:
    """Manages order lifecycle and tracking.

    Responsibilities:
        - Create and validate orders
        - Track order status transitions
        - Record fills
        - Maintain order history
        - Provide position reconciliation
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._next_id: int = 1

    def _generate_id(self) -> str:
        oid = f"ORD-{self._next_id:06d}"
        self._next_id += 1
        return oid

    # --- Order creation ---

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Create a market order.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            side: BUY or SELL.
            quantity: Number of units.
            metadata: Optional extra data (strategy name, signal, etc.).

        Returns:
            Created Order in PENDING status.
        """
        order = Order(
            order_id=self._generate_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata=metadata or {},
        )
        self._orders[order.order_id] = order
        logger.info(
            "OMS: Created {} {} {} qty={:.6f} [{}]",
            order.order_type.value, order.side.value, symbol,
            quantity, order.order_id,
        )
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Create a limit order."""
        order = Order(
            order_id=self._generate_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            metadata=metadata or {},
        )
        self._orders[order.order_id] = order
        logger.info(
            "OMS: Created {} {} {} qty={:.6f} price={:.2f} [{}]",
            order.order_type.value, order.side.value, symbol,
            quantity, price, order.order_id,
        )
        return order

    # --- Status transitions ---

    def mark_submitted(self, order_id: str, exchange_order_id: str | None = None) -> None:
        """Mark order as submitted to exchange."""
        order = self._get_order(order_id)
        order.status = OrderStatus.SUBMITTED
        order.exchange_order_id = exchange_order_id
        order.updated_at = datetime.now(timezone.utc)
        logger.info("OMS: Order {} -> SUBMITTED (exchange={})", order_id, exchange_order_id)

    def mark_filled(
        self,
        order_id: str,
        filled_price: float,
        filled_quantity: float | None = None,
        commission: float = 0.0,
    ) -> Fill:
        """Mark order as filled and record the fill.

        Args:
            order_id: Internal order ID.
            filled_price: Execution price.
            filled_quantity: Filled quantity (defaults to full order quantity).
            commission: Commission paid.

        Returns:
            Fill record.
        """
        order = self._get_order(order_id)
        qty = filled_quantity or order.quantity

        order.filled_quantity += qty
        order.filled_price = filled_price
        order.commission += commission
        order.updated_at = datetime.now(timezone.utc)

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=filled_price,
            commission=commission,
        )
        self._fills.append(fill)

        logger.info(
            "OMS: Order {} -> {} price={:.2f} qty={:.6f} comm={:.4f}",
            order_id, order.status.value, filled_price, qty, commission,
        )
        return fill

    def mark_cancelled(self, order_id: str) -> None:
        """Mark order as cancelled."""
        order = self._get_order(order_id)
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc)
        logger.info("OMS: Order {} -> CANCELLED", order_id)

    def mark_rejected(self, order_id: str, reason: str) -> None:
        """Mark order as rejected."""
        order = self._get_order(order_id)
        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason
        order.updated_at = datetime.now(timezone.utc)
        logger.warning("OMS: Order {} -> REJECTED: {}", order_id, reason)

    # --- Queries ---

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID, or None if not found."""
        return self._orders.get(order_id)

    def _get_order(self, order_id: str) -> Order:
        """Get order or raise."""
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")
        return order

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open (PENDING or SUBMITTED) orders."""
        open_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        orders = [o for o in self._orders.values() if o.status in open_statuses]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fills(self, symbol: str | None = None) -> list[Fill]:
        """Get all fills, optionally filtered by symbol."""
        if symbol:
            return [f for f in self._fills if f.symbol == symbol]
        return list(self._fills)

    def get_all_orders(self) -> list[Order]:
        """Get all orders (any status)."""
        return list(self._orders.values())

    @property
    def total_orders(self) -> int:
        return len(self._orders)

    @property
    def total_fills(self) -> int:
        return len(self._fills)

    # --- Reconciliation ---

    def net_position(self, symbol: str) -> float:
        """Calculate net filled position for a symbol.

        Positive = net long, negative = net short.
        """
        net = 0.0
        for fill in self._fills:
            if fill.symbol != symbol:
                continue
            if fill.side == OrderSide.BUY:
                net += fill.quantity
            else:
                net -= fill.quantity
        return net

    def total_commission(self, symbol: str | None = None) -> float:
        """Total commission paid, optionally filtered by symbol."""
        fills = self.get_fills(symbol)
        return sum(f.commission for f in fills)

    def order_summary(self) -> dict[str, int]:
        """Count of orders by status."""
        summary: dict[str, int] = {}
        for order in self._orders.values():
            key = order.status.value
            summary[key] = summary.get(key, 0) + 1
        return summary
