# Fases de Desarrollo — Crypto First (Binance)

## Enfoque

Mercado inicial: **Criptomonedas (Binance vía CCXT)**. Stocks (IB/Alpaca) se integran después.
Estrategias en orden: Pairs Trading → Mean Reversion → ML Ensemble.

## Resumen de Fases

| Fase | Entregable |
|------|------------|
| 1. Scaffolding e Infraestructura | Data pipeline Binance funcionando |
| 2. Motor de Backtesting | Engine validado con métricas |
| 3. Pairs Trading | Primera estrategia Sharpe > 1.0 |
| 4. Mean Reversion | Segunda estrategia validada |
| 5. ML Ensemble + Agregador | Ensemble combinado |
| 6. Gestión de Riesgo | Sistema integrado de riesgo |
| 7. Ejecución y Automatización | Paper trading 24/7 |
| 8. Monitoreo y Go-Live | Live con capital real |

---

## Fase 1: Scaffolding e Infraestructura

**Objetivo:** Proyecto ejecutable con data pipeline de Binance funcionando.

### Archivos
```
requirements.txt, .env.example, .gitignore
docker-compose.yml, Dockerfile
config/settings.yaml, config/risk.yaml
src/data/sources.py          # CCXT wrapper (Binance)
src/data/pipeline.py         # ETL: fetch → validate → store
src/data/storage.py          # PostgreSQL/TimescaleDB
src/features/technical.py    # RSI, MACD, BB, ATR
src/features/statistical.py  # Z-score, correlaciones, cointegración
tests/test_data_pipeline.py
main.py
```

### Tareas
- [ ] Crear `requirements.txt` con dependencias reales
- [ ] Crear `.env.example` con variables de Binance
- [ ] Crear `.gitignore` (Python)
- [ ] Crear `docker-compose.yml` (TimescaleDB + Redis)
- [ ] Implementar `src/data/sources.py` — fetch OHLCV de Binance vía CCXT
- [ ] Implementar `src/data/pipeline.py` — ETL con validación de datos
- [ ] Implementar `src/data/storage.py` — interface TimescaleDB
- [ ] Implementar `src/features/technical.py` — RSI, MACD, BB, ATR
- [ ] Implementar `src/features/statistical.py` — z-score, cointegración
- [ ] Crear `config/settings.yaml` y `config/risk.yaml`
- [ ] Tests del pipeline
- [ ] `main.py` entry point

### Criterios de éxito
- [ ] `docker-compose up` levanta DB + Redis
- [ ] `python main.py` descarga datos OHLCV de Binance y los almacena
- [ ] Features técnicos se calculan correctamente

---

## Fase 2: Motor de Backtesting

**Objetivo:** Engine de backtesting validado con métricas completas.

### Archivos
```
src/backtester/engine.py       # BacktestEngine
src/backtester/metrics.py      # Sharpe, Sortino, Calmar, DD, Win Rate
src/backtester/walk_forward.py # Walk-forward validation
tests/test_backtester.py
```

### Tareas
- [ ] `BacktestEngine` — simula trades con comisiones (0.1%) y slippage (0.1%)
- [ ] `metrics.py` — Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor
- [ ] `walk_forward.py` — train/test rolling windows
- [ ] Tests unitarios de métricas
- [ ] Visualización de equity curve (plotly)

### Criterios de éxito
- [ ] Backtest ejecuta en < 30s para 1 año de datos 1h
- [ ] Métricas coinciden con cálculos manuales

---

## Fase 3: Pairs Trading

**Objetivo:** Primera estrategia funcional con Sharpe > 1.0.

### Archivos
```
src/strategies/base.py           # Clase abstracta BaseStrategy
src/strategies/pairs_trading.py  # Pairs Trading crypto
```

### Tareas
- [ ] `base.py` — BaseStrategy con `generate_signal()`, `get_parameters()`
- [ ] `pairs_trading.py` — Cointegración, hedge ratio, z-score signals
- [ ] Parámetros crypto: lookback=168h, entry_z=2.0, stop_z=3.5
- [ ] Pares: BTC/ETH, BNB/SOL, LINK/DOT
- [ ] Backtest completo con walk-forward
- [ ] Optimización de parámetros

### Criterios de éxito
- [ ] Sharpe > 1.0 en backtest out-of-sample
- [ ] Max drawdown < 20%

---

## Fase 4: Mean Reversion

**Objetivo:** Segunda estrategia validada.

### Archivos
```
src/strategies/mean_reversion.py
```

### Tareas
- [ ] Mean Reversion con z-score (lookback=20 períodos)
- [ ] Filtro de volatilidad (no operar en tendencias fuertes)
- [ ] Backtest y walk-forward
- [ ] Validar Sharpe > 1.0

---

## Fase 5: ML Ensemble + Agregador

**Objetivo:** Ensemble combinado que mejora métricas vs estrategias individuales.

### Archivos
```
src/strategies/ml_ensemble.py  # XGBoost + RF + LightGBM
src/strategies/aggregator.py   # Votación ponderada
```

### Tareas
- [ ] Feature engineering (20+ features)
- [ ] XGBoost + RandomForest + LightGBM (pesos 0.4/0.3/0.3)
- [ ] Cross-validation 5-fold
- [ ] `aggregator.py` — votación ponderada de las 3 estrategias
- [ ] Backtest del ensemble combinado

### Criterios de éxito
- [ ] 2+ estrategias con Sharpe > 1.0
- [ ] Agregador mejora métricas vs individual

---

## Fase 6: Gestión de Riesgo

**Objetivo:** Sistema de riesgo integrado con todas las estrategias.

### Archivos
```
src/risk/position_sizing.py  # Kelly Criterion (25%)
src/risk/portfolio.py        # Límites de portfolio
src/risk/stops.py            # Stop loss dinámico ATR
```

### Tareas
- [ ] Kelly Criterion con fracción 25%, max 3% position (crypto)
- [ ] Portfolio limits: 20% DD, 3% daily loss, 20 max positions
- [ ] ATR stops: 2.5x SL, 4.0x TP (crypto)
- [ ] Trailing stop (activación a 2x riesgo)
- [ ] Validación pre-trade integrada
- [ ] Tests de integración

### Criterios de éxito
- [ ] Todos los límites se respetan en backtest
- [ ] Tests unitarios > 80% coverage

---

## Fase 7: Ejecución y Automatización

**Objetivo:** Paper trading 24/7 sin errores.

### Archivos
```
src/execution/oms.py             # Order Management System
src/execution/binance_broker.py  # Ejecución real en Binance
src/execution/scheduler.py       # APScheduler 24/7
```

### Tareas
- [ ] OMS — gestión de órdenes, estados, logging
- [ ] `binance_broker.py` — market/limit orders vía CCXT
- [ ] `scheduler.py` — jobs cada 1h/4h, reconciliación
- [ ] Paper trading en Binance testnet (2+ semanas)

### Criterios de éxito
- [ ] Sin errores críticos en paper trading
- [ ] Fills dentro de 0.5% slippage

---

## Fase 8: Monitoreo y Go-Live

**Objetivo:** Sistema en producción con capital real.

### Archivos
```
src/monitoring/dashboard.py  # Streamlit
src/monitoring/alerts.py     # Email/Slack
```

### Tareas
- [ ] Dashboard Streamlit: equity curve, posiciones, métricas
- [ ] Alertas: DD > 15%, daily loss > 2%, stops hit
- [ ] Go-live con capital mínimo
- [ ] Escalado gradual

### Plan de Escalado

| Período | Capital | Condición |
|---------|---------|-----------|
| Semanas 1-2 | $10K | Validar ejecución |
| Semanas 3-4 | $25K | Sharpe > 0.8 |
| Semanas 5-6 | $50K | Consistencia |
| Semana 7+ | $100K+ | Operación normal |

### Criterios de éxito final
- [ ] Sharpe > 0.8 en primeras 4 semanas live
- [ ] Max drawdown < 15%
- [ ] Dashboard actualiza en tiempo real
- [ ] Alertas funcionando
