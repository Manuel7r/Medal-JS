# Arquitectura del Sistema

## Diagrama de Capas

```
┌─────────────────────────────────────────────────────────────┐
│                      CAPA DE DATOS                          │
│  Real-time Data → Historical OHLCV → Alternative Data       │
│                          ↓                                  │
│              Data Validation & Normalization                │
│                          ↓                                  │
│              PostgreSQL/TimescaleDB + Redis                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   CAPA DE FEATURES                          │
│  Indicadores técnicos (RSI, MACD, BB, ATR)                 │
│  Features estadísticos (z-score, skew, kurtosis)           │
│  Cointegración y correlaciones                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE ESTRATEGIAS                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Pairs   │  │   Mean   │  │    ML    │                  │
│  │ Trading  │  │ Reversion│  │ Ensemble │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       └─────────────┼─────────────┘                        │
│                     ↓                                       │
│           Signal Aggregator (Voting)                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   CAPA DE RIESGO                            │
│  Position Sizing (Kelly) → Portfolio Limits → Stop Loss    │
│                          ↓                                  │
│               Pre-Trade Validation                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE EJECUCIÓN                          │
│            Order Management System (OMS)                    │
│                          ↓                                  │
│  Interactive Brokers │ Alpaca │ Binance (CCXT)             │
│                          ↕                                  │
│               WebSocket (datos real-time crypto)            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE MONITOREO                          │
│  Dashboard (Streamlit) │ Alertas │ Analytics & Logs        │
└─────────────────────────────────────────────────────────────┘
```

## Estructura de Directorios

```
trading-system/
├── config/
│   ├── settings.yaml           # Configuración general
│   ├── strategies.yaml         # Parámetros de estrategias
│   └── risk.yaml               # Límites de riesgo
│
├── src/
│   ├── data/
│   │   ├── sources.py          # Yahoo Finance, Alpha Vantage, Binance (CCXT)
│   │   ├── pipeline.py         # ETL de datos
│   │   └── storage.py          # Interfaz con BD
│   │
│   ├── features/
│   │   ├── technical.py        # RSI, MACD, BB, ATR
│   │   └── statistical.py      # Z-score, correlaciones
│   │
│   ├── strategies/
│   │   ├── base.py             # Clase abstracta
│   │   ├── pairs_trading.py
│   │   ├── mean_reversion.py
│   │   └── ml_ensemble.py
│   │
│   ├── risk/
│   │   ├── position_sizing.py  # Kelly Criterion
│   │   ├── portfolio.py        # Límites
│   │   └── stops.py            # Stop loss dinámico
│   │
│   ├── execution/
│   │   ├── oms.py              # Order Management
│   │   ├── brokers/            # IB, Alpaca, Binance
│   │   └── scheduler.py        # APScheduler
│   │
│   ├── backtester/
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   │
│   └── monitoring/
│       ├── dashboard.py        # Streamlit
│       └── alerts.py
│
├── tests/
├── examples/
├── docs/
├── requirements.txt
└── main.py
```

## Schema de Base de Datos

```sql
CREATE TABLE ohlcv (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    close DECIMAL(18,8),
    volume DECIMAL(24,8),
    PRIMARY KEY (symbol, timestamp)
);

-- TimescaleDB hypertable para performance
SELECT create_hypertable('ohlcv', 'timestamp');

-- Índices
CREATE INDEX idx_symbol ON ohlcv(symbol);
CREATE INDEX idx_timestamp ON ohlcv(timestamp DESC);

-- Columna para distinguir mercado
ALTER TABLE ohlcv ADD COLUMN market VARCHAR(10) DEFAULT 'stock';
-- market: 'stock', 'crypto', 'forex'
CREATE INDEX idx_market ON ohlcv(market);
```
