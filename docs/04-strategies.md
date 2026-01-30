# Estrategias Cuantitativas

## Resumen

| Estrategia | Tipo | Sharpe | Drawdown | Frecuencia |
|------------|------|--------|----------|------------|
| Pairs Trading | Market Neutral | 1.2-1.5 | 8-12% | 100-200/año |
| Mean Reversion | Direccional | 1.0-1.3 | 12-18% | 150-300/año |
| ML Ensemble | Direccional | 0.9-1.2 | 15-20% | 200-400/año |

---

## 1. Pairs Trading (Arbitraje Estadístico)

### Concepto
Explotar la relación de cointegración entre activos correlacionados. Cuando el spread diverge, revertirá a la media.

### Parámetros

| Parámetro | Valor |
|-----------|-------|
| Lookback | 60 días |
| Entry Z-score | 1.5 |
| Exit Z-score | 0.5 |
| Stop Z-score | 3.0 |

### Workflow

```python
# 1. Identificar pares cointegrados (p-value < 0.05)
_, p_value, _ = coint(asset1, asset2)

# 2. Calcular hedge ratio
hedge_ratio = LinearRegression().fit(asset2, asset1).coef_[0]

# 3. Calcular spread y z-score
spread = asset1 - hedge_ratio * asset2
z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

# 4. Generar señales
if z_score > 1.5:    return "SHORT_SPREAD"  # Vender A, Comprar B
if z_score < -1.5:   return "LONG_SPREAD"   # Comprar A, Vender B
if abs(z_score) < 0.5: return "EXIT"
```

### Ventajas
- Market neutral (beta ≈ 0)
- Bajo drawdown
- Poco correlacionado con mercado

---

## 2. Mean Reversion

### Concepto
Precios extremos tienden a revertir a su media histórica.

### Parámetros

| Parámetro | Valor |
|-----------|-------|
| Lookback | 20 días |
| Entry | 2.0 std |
| Exit | 0.5 std |

### Implementación

```python
def generate_signal(prices, lookback=20, entry=2.0, exit=0.5):
    mean = prices.rolling(lookback).mean().iloc[-1]
    std = prices.rolling(lookback).std().iloc[-1]
    z_score = (prices.iloc[-1] - mean) / std

    if z_score < -entry:   return "BUY"   # Sobreventa
    if z_score > entry:    return "SELL"  # Sobrecompra
    if abs(z_score) < exit: return "EXIT"
    return "HOLD"
```

### Aplicación
- Mejor en mercados ranging
- Evitar en tendencias fuertes
- Combinar con filtro de volatilidad

---

## 3. ML Ensemble

### Concepto
Combinación de múltiples modelos para predicción robusta.

### Features

```python
FEATURES = [
    # Técnicos
    'rsi_14', 'macd', 'bb_position', 'atr_14',

    # Price Action
    'returns_1d', 'returns_5d', 'returns_20d',

    # Volumen
    'volume_ma_ratio',

    # Estadísticos
    'z_score_20', 'returns_skew', 'returns_kurt',

    # Lagged
    'returns_lag_1', 'returns_lag_2', 'returns_lag_5'
]
```

### Ensemble

```python
models = {
    'xgb': XGBClassifier(max_depth=5, n_estimators=100),
    'rf': RandomForestClassifier(n_estimators=100),
    'lgb': LGBMClassifier(n_estimators=100)
}
weights = {'xgb': 0.4, 'rf': 0.3, 'lgb': 0.3}

# Predicción ponderada
prediction = sum(model.predict_proba(X)[0,1] * weights[name]
                 for name, model in models.items())

if prediction > 0.55: return "BUY"
if prediction < 0.45: return "SELL"
```

### Validación
- Walk-forward testing (train: 2 años, test: 3 meses)
- Cross-validation 5-fold
- Feature importance analysis

---

## Agregador de Señales

```python
def aggregate_signals(signals, weights):
    """
    Combina señales de múltiples estrategias.
    Requiere mayoría ponderada para actuar.
    """
    scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

    for strategy, signal in signals.items():
        scores[signal] += weights[strategy]

    total = sum(weights.values())

    if scores['BUY'] > total * 0.5:  return 'BUY'
    if scores['SELL'] > total * 0.5: return 'SELL'
    return 'HOLD'
```

---

## Aplicación en Crypto (Binance)

### Pares Recomendados

| Par 1 | Par 2 | Correlación | Uso |
|-------|-------|-------------|-----|
| BTC/USDT | ETH/USDT | 0.85-0.92 | Pairs Trading principal |
| BNB/USDT | SOL/USDT | 0.75-0.85 | L1 alternativas |
| LINK/USDT | DOT/USDT | 0.70-0.80 | Infraestructura |

### Ajustes de Parámetros para Crypto

| Parámetro | Stocks | Crypto |
|-----------|--------|--------|
| Timeframe | Diario | 1h o 4h |
| Lookback | 60 días | 168 horas (7 días) |
| Entry Z-score | 1.5 | 2.0 |
| Stop Z-score | 3.0 | 3.5 |
| Position Size | 5% | 3% |
| Max Hold | Indefinido | 168 horas |

### Consideraciones Crypto
- **24/7:** No hay cierre de mercado, scheduler sin restricción horaria
- **Volatilidad:** 3-10% diario vs 1-3% en stocks → z-scores más altos
- **Comisiones:** 0.1% maker/taker en Binance (0.075% con BNB)
- **Flash crashes:** Más frecuentes → stop más holgado
- **Liquidez:** Variable por hora, mejor en UTC 12:00-20:00

Ver [09-crypto-binance.md](./09-crypto-binance.md) para detalles completos.
