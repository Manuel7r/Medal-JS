# Trading de Criptomonedas con Binance

## Configuración Inicial

### 1. Crear API Keys

1. Ir a [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Crear nueva API key
3. Permisos necesarios:
   - **Enable Reading** (obligatorio)
   - **Enable Spot & Margin Trading** (para operar)
   - **Enable Futures** (si se usa futuros)
4. Configurar IP whitelist (recomendado para producción)

### 2. Testnet (Desarrollo)

```
Binance Testnet: https://testnet.binance.vision/
Futures Testnet: https://testnet.binancefuture.com/

API Testnet:
  Base URL: https://testnet.binance.vision/api
  WebSocket: wss://testnet.binance.vision/ws
```

### 3. Variables de Entorno

```bash
# .env (NUNCA commitear este archivo)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret
BINANCE_TESTNET=true  # Cambiar a false para producción
```

---

## Conexión con CCXT

```python
import ccxt

def create_binance_client(testnet: bool = True) -> ccxt.binance:
    """Crea cliente de Binance usando CCXT."""
    config = {
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',       # 'spot', 'future', 'margin'
            'adjustForTimeDifference': True
        }
    }

    if testnet:
        config['sandbox'] = True

    exchange = ccxt.binance(config)
    exchange.load_markets()

    return exchange
```

---

## Pares Recomendados

### Para Pairs Trading (alta correlación)

| Par 1 | Par 2 | Correlación | Notas |
|-------|-------|-------------|-------|
| BTC/USDT | ETH/USDT | 0.85-0.92 | Par principal |
| BNB/USDT | SOL/USDT | 0.75-0.85 | Ecosistemas L1 |
| LINK/USDT | DOT/USDT | 0.70-0.80 | Infraestructura |
| AVAX/USDT | NEAR/USDT | 0.70-0.78 | L1 alternativas |

### Para Mean Reversion

```
Alta liquidez (recomendados):
  BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT

Media liquidez:
  AVAX/USDT, DOT/USDT, LINK/USDT, MATIC/USDT

Evitar (baja liquidez / alta manipulación):
  Memecoins, tokens < $100M market cap
```

---

## Diferencias vs Mercados Tradicionales

| Aspecto | Stocks | Crypto |
|---------|--------|--------|
| Horario | Lun-Vie 9:30-16:00 | 24/7/365 |
| Volatilidad diaria | 1-3% | 3-10% |
| Comisiones | 0.01-0.1% | 0.1% (0.075% con BNB) |
| Slippage | 0.05-0.2% | 0.1-0.5% |
| Flash crashes | Raro | Frecuente |
| Settlement | T+2 | Inmediato |
| Short selling | Requiere margen | Fácil en futuros |
| Regulación | Alta | Variable |

---

## Ajustes de Estrategia para Crypto

### Timeframes Recomendados

```
Pairs Trading:  1h o 4h (no diario como stocks)
Mean Reversion: 15min a 4h
ML Ensemble:    1h

Lookback ajustado:
  Stocks: 20-60 días
  Crypto: 48-168 horas (2-7 días en velas horarias)
```

### Parámetros Ajustados

| Parámetro | Stocks | Crypto |
|-----------|--------|--------|
| Max Drawdown | 15% | 20% |
| Daily Loss | 2% | 3% |
| Position Size | 5% | 3% |
| ATR Multiplier SL | 2.0x | 2.5x |
| ATR Multiplier TP | 3.0x | 4.0x |
| Entry Z-score | 1.5 | 2.0 |
| Stop Z-score | 3.0 | 3.5 |

---

## Obtener Datos Históricos

```python
def fetch_ohlcv(exchange, symbol: str, timeframe: str = '1h',
                limit: int = 1000) -> pd.DataFrame:
    """
    Obtiene datos OHLCV de Binance.

    Timeframes: '1m', '5m', '15m', '1h', '4h', '1d'
    Límite máximo por request: 1000 velas
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


def fetch_full_history(exchange, symbol: str, timeframe: str,
                       since: str) -> pd.DataFrame:
    """
    Obtiene historial completo paginando requests.
    """
    since_ts = exchange.parse8601(since + 'T00:00:00Z')
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe,
                                      since=since_ts, limit=1000)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        since_ts = ohlcv[-1][0] + 1  # Siguiente timestamp

        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated()]

    return df
```

---

## Ejecutar Órdenes

```python
def place_market_order(exchange, symbol: str, side: str,
                       amount: float) -> dict:
    """Ejecuta orden de mercado."""
    order = exchange.create_order(
        symbol=symbol,
        type='market',
        side=side,     # 'buy' o 'sell'
        amount=amount  # En unidades base (ej: 0.01 BTC)
    )
    return order


def place_limit_order(exchange, symbol: str, side: str,
                      amount: float, price: float) -> dict:
    """Ejecuta orden límite."""
    order = exchange.create_order(
        symbol=symbol,
        type='limit',
        side=side,
        amount=amount,
        price=price
    )
    return order


def get_balance(exchange) -> dict:
    """Obtiene balance de la cuenta."""
    balance = exchange.fetch_balance()
    return {
        'USDT': balance['USDT']['free'],
        'BTC': balance['BTC']['free'],
        'ETH': balance['ETH']['free'],
        'total_usdt': balance['total']['USDT']
    }
```

---

## Scheduler 24/7

```python
# Ajuste del scheduler para crypto (24/7)
scheduler.add_job(
    func=run_strategies,
    trigger=CronTrigger(
        minute='*/15'      # Cada 15 minutos
        # SIN restricción de hora ni día de semana
    ),
    id='crypto_strategy'
)

scheduler.add_job(
    func=monitor_positions,
    trigger=CronTrigger(
        minute='*/5'       # Cada 5 minutos
    ),
    id='crypto_monitor'
)
```

---

## Comisiones Binance

```
Spot:
  Maker: 0.10%
  Taker: 0.10%
  Con BNB: 0.075% (-25%)

Futures:
  Maker: 0.02%
  Taker: 0.04%

Descuentos VIP:
  VIP 1 (> $1M/mes): Maker 0.09%, Taker 0.09%
  VIP 2 (> $5M/mes): Maker 0.08%, Taker 0.08%
```

---

## Consideraciones de Seguridad

1. **Nunca** guardar API keys en código
2. Usar IP whitelist en producción
3. No habilitar permisos de retiro
4. No dejar todo el capital en el exchange
5. Monitorear accesos no autorizados
6. Rotar API keys periódicamente
