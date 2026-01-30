# Sistema de Trading Cuantitativo - Documentación

Sistema automatizado basado en principios de Jim Simons / Renaissance Technologies.
Soporta **stocks** (Interactive Brokers, Alpaca) y **criptomonedas** (Binance).

## Objetivos

| Métrica | Mínimo | Ideal |
|---------|--------|-------|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 20% | < 12% |
| Win Rate | > 45% | > 52% |
| Retorno Anual | > 10% | > 18% |

## Índice de Documentos

1. [Resumen Ejecutivo](./01-overview.md) - Visión general y metas
2. [Principios Fundamentales](./02-principles.md) - Los 7 principios de Simons
3. [Arquitectura](./03-architecture.md) - Diagrama y estructura del sistema
4. [Estrategias](./04-strategies.md) - Pairs Trading, Mean Reversion, ML
5. [Gestión de Riesgo](./05-risk-management.md) - Límites y position sizing
6. [Fases de Desarrollo](./06-development-phases.md) - Timeline y tareas
7. [Stack Tecnológico](./07-tech-stack.md) - Herramientas y librerías
8. [Checklist](./08-checklist.md) - Validación pre-producción
9. [Crypto & Binance](./09-crypto-binance.md) - Trading de criptomonedas

## Código de Referencia

Ver carpeta [`../examples/`](../examples/) para implementaciones:
- `pairs_trading.py` - Estrategia de arbitraje estadístico
- `backtest_engine.py` - Motor de backtesting
- `risk_manager.py` - Gestión de riesgo
- `strategy_base.py` - Clase base para estrategias
- `binance_client.py` - Cliente wrapper para Binance (CCXT)
- `crypto_pairs_trading.py` - Pairs trading adaptado a crypto

## Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar backtest de ejemplo (stocks)
python examples/pairs_trading.py

# Ejecutar backtest crypto
python examples/crypto_pairs_trading.py

# Configurar Binance
export BINANCE_API_KEY=your_key
export BINANCE_SECRET=your_secret
```
