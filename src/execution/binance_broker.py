"""Binance broker: executes real orders via CCXT and integrates with OMS.

Supports market and limit orders, cancel, balance queries.
Uses testnet by default for paper trading.
"""

from datetime import datetime, timezone

import ccxt
from loguru import logger

from src.execution.oms import (
    Fill,
    Order,
    OrderManagementSystem,
    OrderSide,
    OrderType,
)
from src.risk.portfolio import PortfolioRiskManager, RejectionReason


class BinanceBroker:
    """Executes orders on Binance via CCXT with OMS integration.

    All orders pass through the OMS for tracking and through the
    PortfolioRiskManager for pre-trade validation.

    Args:
        api_key: Binance API key.
        secret: Binance API secret.
        testnet: Use testnet (default True for safety).
        oms: Order Management System instance.
        risk_manager: Portfolio risk manager (optional).
    """

    def __init__(
        self,
        api_key: str = "",
        secret: str = "",
        testnet: bool = True,
        oms: OrderManagementSystem | None = None,
        risk_manager: PortfolioRiskManager | None = None,
    ) -> None:
        config: dict = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if testnet:
            config["sandbox"] = True

        self.exchange = ccxt.binance(config)
        self.oms = oms or OrderManagementSystem()
        self.risk_manager = risk_manager
        self._testnet = testnet

        logger.info("BinanceBroker initialized (testnet={})", testnet)

    def _validate_risk(self, symbol: str, price: float, quantity: float) -> RejectionReason:
        """Run pre-trade risk validation if risk manager is set."""
        if self.risk_manager is None:
            return RejectionReason(True, "OK")
        return self.risk_manager.can_open_position(symbol, price, quantity)

    # --- Order execution ---

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        metadata: dict | None = None,
    ) -> Order:
        """Place a market order on Binance.

        Flow: risk check -> OMS create -> exchange submit -> OMS fill/reject.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".
            side: BUY or SELL.
            quantity: Number of units.
            metadata: Optional extra data.

        Returns:
            Order with final status (FILLED or REJECTED).
        """
        # Create in OMS
        order = self.oms.create_market_order(symbol, side, quantity, metadata)

        # Risk check
        ticker = self._fetch_ticker_safe(symbol)
        price_estimate = ticker.get("last", 0) if ticker else 0
        risk_result = self._validate_risk(symbol, price_estimate, quantity)
        if not risk_result.allowed:
            self.oms.mark_rejected(order.order_id, risk_result.reason)
            return order

        # Submit to exchange
        try:
            ccxt_side = "buy" if side == OrderSide.BUY else "sell"
            result = self.exchange.create_order(symbol, "market", ccxt_side, quantity)

            self.oms.mark_submitted(order.order_id, result.get("id"))

            # Record fill
            filled_price = float(result.get("average", result.get("price", 0)))
            filled_qty = float(result.get("filled", quantity))
            fee = result.get("fee", {})
            commission = float(fee.get("cost", 0)) if fee else 0.0

            self.oms.mark_filled(order.order_id, filled_price, filled_qty, commission)

            logger.info(
                "Broker: MARKET {} {} qty={:.6f} filled@{:.2f}",
                side.value, symbol, filled_qty, filled_price,
            )

        except ccxt.BaseError as e:
            self.oms.mark_rejected(order.order_id, str(e))
            logger.error("Broker: Market order failed for {}: {}", symbol, e)

        return order

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        metadata: dict | None = None,
    ) -> Order:
        """Place a limit order on Binance.

        Args:
            symbol: Trading pair.
            side: BUY or SELL.
            quantity: Number of units.
            price: Limit price.
            metadata: Optional extra data.

        Returns:
            Order in SUBMITTED or REJECTED status.
        """
        order = self.oms.create_limit_order(symbol, side, quantity, price, metadata)

        risk_result = self._validate_risk(symbol, price, quantity)
        if not risk_result.allowed:
            self.oms.mark_rejected(order.order_id, risk_result.reason)
            return order

        try:
            ccxt_side = "buy" if side == OrderSide.BUY else "sell"
            result = self.exchange.create_order(symbol, "limit", ccxt_side, quantity, price)

            self.oms.mark_submitted(order.order_id, result.get("id"))

            logger.info(
                "Broker: LIMIT {} {} qty={:.6f} price={:.2f} [{}]",
                side.value, symbol, quantity, price, result.get("id"),
            )

        except ccxt.BaseError as e:
            self.oms.mark_rejected(order.order_id, str(e))
            logger.error("Broker: Limit order failed for {}: {}", symbol, e)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Internal OMS order ID.

        Returns:
            True if cancelled successfully.
        """
        order = self.oms.get_order(order_id)
        if order is None:
            logger.warning("Broker: Order {} not found for cancel", order_id)
            return False

        if order.exchange_order_id is None:
            self.oms.mark_cancelled(order_id)
            return True

        try:
            self.exchange.cancel_order(order.exchange_order_id, order.symbol)
            self.oms.mark_cancelled(order_id)
            logger.info("Broker: Cancelled order {} (exchange={})", order_id, order.exchange_order_id)
            return True
        except ccxt.BaseError as e:
            logger.error("Broker: Cancel failed for {}: {}", order_id, e)
            return False

    def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol.

        Returns:
            Number of orders cancelled.
        """
        open_orders = self.oms.get_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            if self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    # --- Queries ---

    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance for a currency."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("free", {}).get(currency, 0))
        except ccxt.BaseError as e:
            logger.error("Broker: Failed to fetch balance: {}", e)
            return 0.0

    def get_total_equity(self, quote: str = "USDT") -> float:
        """Get total equity in quote currency."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("total", {}).get(quote, 0))
        except ccxt.BaseError as e:
            logger.error("Broker: Failed to fetch equity: {}", e)
            return 0.0

    def get_open_positions(self) -> dict[str, float]:
        """Get net positions from OMS fills.

        Returns:
            Dict mapping symbol -> net position.
        """
        positions: dict[str, float] = {}
        for fill in self.oms.get_fills():
            if fill.symbol not in positions:
                positions[fill.symbol] = 0.0
            if fill.side == OrderSide.BUY:
                positions[fill.symbol] += fill.quantity
            else:
                positions[fill.symbol] -= fill.quantity

        return {sym: qty for sym, qty in positions.items() if abs(qty) > 1e-8}

    def _fetch_ticker_safe(self, symbol: str) -> dict:
        """Fetch ticker, returning empty dict on error."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.BaseError as e:
            logger.error("Broker: Failed to fetch ticker for {}: {}", symbol, e)
            return {}

    # --- Reconciliation ---

    def reconcile(self) -> dict:
        """Compare OMS state with exchange state.

        Returns:
            Dict with reconciliation info.
        """
        oms_positions = self.get_open_positions()
        oms_summary = self.oms.order_summary()

        return {
            "oms_positions": oms_positions,
            "oms_orders": oms_summary,
            "total_orders": self.oms.total_orders,
            "total_fills": self.oms.total_fills,
            "total_commission": self.oms.total_commission(),
        }
