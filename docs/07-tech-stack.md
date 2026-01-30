# Stack Tecnológico

## Core

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.10+ |
| Web Framework | FastAPI |
| Dashboard | Streamlit |
| Base de Datos | PostgreSQL + TimescaleDB |
| Cache | Redis |
| Scheduling | APScheduler |
| Containers | Docker |

## Librerías Python

### Data & Analysis
```
pandas>=2.0
numpy>=1.24
polars>=0.19  # Para datos grandes
scipy>=1.11
```

### Machine Learning
```
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
statsmodels>=0.14  # Cointegración
```

### Brokers & Exchanges
```
ib_insync>=0.9        # Interactive Brokers (stocks)
alpaca-trade-api>=3.0 # Alpaca (stocks/crypto)
ccxt>=4.0             # Crypto exchanges (Binance, Bybit, etc.)
python-binance>=1.0   # Binance SDK nativo (alternativa a CCXT)
yfinance>=0.2         # Datos históricos stocks
```

### Binance Específico
```
ccxt>=4.0             # Recomendado: abstracción multi-exchange
python-binance>=1.0   # Opcional: SDK nativo con WebSocket
websockets>=12.0      # Para streaming de datos en tiempo real
```

### Visualización
```
plotly>=5.0
streamlit>=1.28
```

### Testing
```
pytest>=7.0
pytest-cov>=4.0
```

## Infraestructura

### Desarrollo Local
```
CPU: 4+ cores
RAM: 16GB
SSD: 256GB
```

### Producción (AWS)
```
EC2: t3.large ($0.0832/hr)
RDS: db.t3.medium (~$50/mes)
S3: Para backups (~$5/mes)
Total: ~$100-150/mes
```

## requirements.txt

```
# Data
pandas>=2.0.0
numpy>=1.24.0
polars>=0.19.0
scipy>=1.11.0

# ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
statsmodels>=0.14.0

# Brokers & Exchanges
ib_insync>=0.9.0
alpaca-trade-api>=3.0.0
ccxt>=4.0.0
python-binance>=1.0.0
yfinance>=0.2.0
websockets>=12.0.0

# Web
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0

# Database
psycopg2-binary>=2.9.0
redis>=5.0.0
sqlalchemy>=2.0.0

# Scheduling
apscheduler>=3.10.0

# Visualization
plotly>=5.18.0

# Utils
python-dotenv>=1.0.0
pydantic>=2.5.0
loguru>=0.7.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

## docker-compose.yml

```yaml
version: '3.8'

services:
  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: trading_system
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  app:
    build: .
    depends_on:
      - db
      - redis
    environment:
      DATABASE_URL: postgresql://trading:${DB_PASSWORD}@db:5432/trading_system
      REDIS_URL: redis://redis:6379
    ports:
      - "8000:8000"  # FastAPI
      - "8501:8501"  # Streamlit

volumes:
  postgres_data:
```

## CI/CD (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src tests/
```
