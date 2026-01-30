"""
Cliente wrapper para Binance usando CCXT.

Funcionalidades:
- Conexión spot y futures (testnet y producción)
- Obtener OHLCV histórico con paginación
- Ejecutar órdenes market/limit
- Obtener balance y posiciones
- Rate limiting automático
- Logging de operaciones
"""

import os
import time
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class BinanceClient:
    """
    Wrapper para interactuar con Binance.

    Args:
        testnet: Si True, usa testnet de Binance
        market_type: 'spot' o 'future'
    """

    def __init__(self,
                 testnet: bool = True,
                 market_type: str = 'spot'):
        self.testnet = testnet
        self.market_type = market_type
        self.exchange = self._create_exchange()
        self.markets = {}

        logger.info(f"BinanceClient inicializado ({'testnet' if testnet else 'producción'}, {market_type})")

    def _create_exchange(self) -> ccxt.binance:
        """Crea instancia de exchange."""
        config = {
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': self.market_type,
                'adjustForTimeDifference': True
            }
        }

        if self.testnet:
            config['sandbox'] = True

        exchange = ccxt.binance(config)

        try:
            exchange.load_markets()
            self.markets = exchange.markets
            logger.info(f"Mercados cargados: {len(self.markets)} pares disponibles")
        except Exception as e:
            logger.error(f"Error cargando mercados: {e}")

        return exchange

    # ─── DATOS ────────────────────────────────────────────────────

    def fetch_ohlcv(self,
                    symbol: str,
                    timeframe: str = '1h',
                    limit: int = 500) -> pd.DataFrame:
        """
        Obtiene datos OHLCV recientes.

        Args:
            symbol: Par (ej: 'BTC/USDT')
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Número de velas (max 1000)

        Returns:
            DataFrame con columnas: open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"Obtenidas {len(df)} velas de {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error obteniendo OHLCV de {symbol}: {e}")
            return pd.DataFrame()

    def fetch_full_history(self,
                           symbol: str,
                           timeframe: str = '1h',
                           since: str = '2023-01-01') -> pd.DataFrame:
        """
        Obtiene historial completo paginando requests.

        Args:
            symbol: Par (ej: 'BTC/USDT')
            timeframe: Timeframe deseado
            since: Fecha inicio 'YYYY-MM-DD'

        Returns:
            DataFrame completo desde la fecha indicada
        """
        since_ts = self.exchange.parse8601(since + 'T00:00:00Z')
        all_ohlcv = []
        batch = 0

        logger.info(f"Descargando historial de {symbol} desde {since}")

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ts, limit=1000
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                since_ts = ohlcv[-1][0] + 1
                batch += 1

                if batch % 10 == 0:
                    logger.info(f"Descargados {len(all_ohlcv)} registros...")

                if len(ohlcv) < 1000:
                    break

                time.sleep(0.1)  # Respetar rate limits

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit alcanzado, esperando 60s...")
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error en paginación: {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        logger.info(f"Historial completo: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
        return df

    def fetch_ticker(self, symbol: str) -> Dict:
        """Obtiene precio actual y stats 24h."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage']
            }
        except Exception as e:
            logger.error(f"Error obteniendo ticker de {symbol}: {e}")
            return {}

    def fetch_multiple_ohlcv(self,
                              symbols: List[str],
                              timeframe: str = '1h',
                              limit: int = 500) -> Dict[str, pd.DataFrame]:
        """Obtiene OHLCV para múltiples pares."""
        data = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)
        return data

    # ─── ÓRDENES ──────────────────────────────────────────────────

    def place_market_order(self,
                           symbol: str,
                           side: str,
                           amount: float) -> Optional[Dict]:
        """
        Ejecuta orden de mercado.

        Args:
            symbol: Par (ej: 'BTC/USDT')
            side: 'buy' o 'sell'
            amount: Cantidad en moneda base (ej: 0.01 BTC)

        Returns:
            Dict con detalles de la orden o None si falla
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )

            logger.info(
                f"Orden market {side.upper()} ejecutada: "
                f"{amount} {symbol} @ {order.get('average', 'N/A')}"
            )
            return order

        except ccxt.InsufficientFunds:
            logger.error(f"Fondos insuficientes para {side} {amount} {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error ejecutando orden: {e}")
            return None

    def place_limit_order(self,
                          symbol: str,
                          side: str,
                          amount: float,
                          price: float) -> Optional[Dict]:
        """
        Ejecuta orden límite.

        Args:
            symbol: Par
            side: 'buy' o 'sell'
            amount: Cantidad en moneda base
            price: Precio límite

        Returns:
            Dict con detalles de la orden
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )

            logger.info(
                f"Orden limit {side.upper()} creada: "
                f"{amount} {symbol} @ {price}"
            )
            return order

        except Exception as e:
            logger.error(f"Error creando orden límite: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancela orden abierta."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Orden {order_id} cancelada")
            return True
        except Exception as e:
            logger.error(f"Error cancelando orden: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Obtiene órdenes abiertas."""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error obteniendo órdenes: {e}")
            return []

    # ─── CUENTA ───────────────────────────────────────────────────

    def get_balance(self) -> Dict[str, float]:
        """
        Obtiene balance de la cuenta.

        Returns:
            Dict con balances por moneda
        """
        try:
            balance = self.exchange.fetch_balance()

            result = {}
            for currency in ['USDT', 'BTC', 'ETH', 'BNB', 'SOL']:
                if currency in balance:
                    free = balance[currency].get('free', 0)
                    if free and float(free) > 0:
                        result[currency] = {
                            'free': float(balance[currency]['free']),
                            'used': float(balance[currency]['used']),
                            'total': float(balance[currency]['total'])
                        }

            logger.info(f"Balance obtenido: {len(result)} monedas con saldo")
            return result

        except Exception as e:
            logger.error(f"Error obteniendo balance: {e}")
            return {}

    def get_total_equity_usdt(self) -> float:
        """Estima equity total en USDT."""
        try:
            balance = self.exchange.fetch_balance()
            total = 0

            for currency, data in balance.items():
                if isinstance(data, dict) and 'total' in data:
                    amount = float(data['total'] or 0)
                    if amount <= 0:
                        continue

                    if currency == 'USDT':
                        total += amount
                    else:
                        try:
                            ticker = self.exchange.fetch_ticker(f"{currency}/USDT")
                            total += amount * ticker['last']
                        except Exception:
                            pass

            return total

        except Exception as e:
            logger.error(f"Error calculando equity: {e}")
            return 0

    # ─── UTILIDADES ───────────────────────────────────────────────

    def get_min_order_amount(self, symbol: str) -> float:
        """Obtiene monto mínimo de orden para un par."""
        if symbol in self.markets:
            limits = self.markets[symbol].get('limits', {})
            amount_limits = limits.get('amount', {})
            return float(amount_limits.get('min', 0))
        return 0

    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Obtiene comisiones de trading."""
        try:
            fees = self.exchange.fetch_trading_fee(symbol)
            return {
                'maker': fees.get('maker', 0.001),
                'taker': fees.get('taker', 0.001)
            }
        except Exception:
            return {'maker': 0.001, 'taker': 0.001}


# ─── EJEMPLO DE USO ──────────────────────────────────────────────

if __name__ == "__main__":
    # Crear cliente (testnet)
    client = BinanceClient(testnet=True, market_type='spot')

    # Obtener datos OHLCV
    btc = client.fetch_ohlcv('BTC/USDT', '1h', 100)
    if not btc.empty:
        print(f"\nBTC/USDT últimas 5 velas (1h):")
        print(btc.tail())

    # Obtener ticker
    ticker = client.fetch_ticker('BTC/USDT')
    if ticker:
        print(f"\nBTC/USDT precio actual: ${ticker['last']:,.2f}")
        print(f"Cambio 24h: {ticker['change_24h']:.2f}%")

    # Obtener balance
    balance = client.get_balance()
    if balance:
        print(f"\nBalance:")
        for currency, data in balance.items():
            print(f"  {currency}: {data['free']:.8f} (disponible)")
