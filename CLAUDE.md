# CLAUDE.md

## Proyecto

Sistema de trading cuantitativo automatizado basado en principios de Jim Simons / Renaissance Technologies. Soporta **stocks** (Interactive Brokers, Alpaca) y **criptomonedas** (Binance).

## Estructura

```
Medal-JS/
├── docs/           # 9 archivos .md (español)
│   ├── README.md
│   ├── 01-overview.md        # Visión general y metas
│   ├── 02-principles.md      # 7 principios de Simons
│   ├── 03-architecture.md    # Diagrama de capas y directorios
│   ├── 04-strategies.md      # Pairs Trading, Mean Reversion, ML
│   ├── 05-risk-management.md # Límites, Kelly, stops
│   ├── 06-development-phases.md
│   ├── 07-tech-stack.md      # Stack y requirements.txt
│   ├── 08-checklist.md       # Validación pre-producción
│   └── 09-crypto-binance.md  # Binance vía CCXT
├── examples/       # 6 archivos Python de referencia
│   ├── strategy_base.py
│   ├── pairs_trading.py
│   ├── backtest_engine.py
│   ├── risk_manager.py
│   ├── binance_client.py
│   └── crypto_pairs_trading.py
└── CLAUDE.md
```

## Stack

- **Lenguaje:** Python 3.10+
- **Brokers:** Interactive Brokers (`ib_insync`), Alpaca (`alpaca-trade-api`), Binance (`ccxt`)
- **ML:** scikit-learn, XGBoost, LightGBM
- **Stats:** statsmodels (cointegración), scipy
- **Data:** pandas, numpy, polars, yfinance
- **Infra:** PostgreSQL + TimescaleDB, Redis, FastAPI, Streamlit, Docker

## Convenciones

- Documentación en **español**, código en **inglés**
- Type hints en todo el código Python
- Docstrings en funciones públicas
- Clase base abstracta `BaseStrategy` en `examples/strategy_base.py`

## Estrategias

| Estrategia | Tipo | Sharpe Esperado |
|------------|------|-----------------|
| Pairs Trading | Market Neutral | 1.2-1.5 |
| Mean Reversion | Direccional | 1.0-1.3 |
| ML Ensemble | Direccional | 0.9-1.2 |

Señales se combinan via `SignalAggregator` (votación ponderada, mayoría >50%).

## Parámetros de Riesgo

| Parámetro | Stocks | Crypto |
|-----------|--------|--------|
| Max Drawdown | 15% | 20% |
| Daily Loss | 2% | 3% |
| Position Size | 5% | 3% |
| Leverage | 3x | 3x |
| ATR Stop Loss | 2.0x | 2.5x |
| ATR Take Profit | 3.0x | 4.0x |

Position sizing usa Kelly Criterion al 25% del óptimo.

## Fases de Desarrollo (Crypto First)

1. **Scaffolding e Infraestructura** — data pipeline Binance, DB, configs
2. **Motor de Backtesting** — engine con comisiones, métricas, walk-forward
3. **Pairs Trading** — primera estrategia, Sharpe > 1.0
4. **Mean Reversion** — segunda estrategia validada
5. **ML Ensemble + Agregador** — XGB/RF/LGB + votación ponderada
6. **Gestión de Riesgo** — Kelly, limits, ATR stops
7. **Ejecución y Automatización** — OMS, Binance broker, scheduler 24/7
8. **Monitoreo y Go-Live** — Streamlit dashboard, alertas, capital real

Detalle completo en `docs/06-development-phases.md`.

## Ejecución

```bash
pip install -r requirements.txt
python examples/pairs_trading.py          # Backtest stocks
python examples/crypto_pairs_trading.py   # Backtest crypto
```

Para Binance configurar variables de entorno:
```bash
export BINANCE_API_KEY=your_key
export BINANCE_SECRET=your_secret
```
