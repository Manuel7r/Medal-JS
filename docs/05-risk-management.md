# Gestión de Riesgo

## Límites Duros

| Parámetro | Límite | Acción si se excede |
|-----------|--------|---------------------|
| Max Drawdown | 15% | Suspensión automática |
| Daily Loss | 2% | Pausa hasta siguiente día |
| Position Size | 5% | Rechazo de orden |
| Leverage | 3x | Rechazo de orden |
| Correlación | 0.7 | No abrir posición |
| Max Positions | 20 | No abrir nuevas |

### Ajustes para Crypto

| Parámetro | Stocks | Crypto |
|-----------|--------|--------|
| Max Drawdown | 15% | 20% |
| Daily Loss | 2% | 3% |
| Position Size | 5% | 3% |
| ATR Mult SL | 2.0x | 2.5x |
| ATR Mult TP | 3.0x | 4.0x |

> Crypto es más volátil. Posiciones más pequeñas y stops más amplios
> compensan el mayor rango de movimiento.

---

## Kelly Criterion

### Fórmula

```
f* = (p × b - q) / b

donde:
  p = probabilidad de ganar
  q = 1 - p (probabilidad de perder)
  b = risk/reward ratio
```

### Ejemplo

```
Win rate: 51%
Risk/Reward: 1.5:1

f* = (0.51 × 1.5 - 0.49) / 1.5
f* = 0.042 = 4.2%
```

### Implementación Conservadora

```python
def kelly_position_size(equity, win_prob, risk_reward=1.5):
    """Usa 25% del Kelly óptimo para seguridad"""
    q = 1 - win_prob
    kelly = (win_prob * risk_reward - q) / risk_reward
    safe_kelly = kelly * 0.25

    # Límites adicionales
    max_position = 0.05  # 5% máximo
    return min(max(safe_kelly, 0), max_position) * equity
```

---

## Stop Loss Dinámico

### ATR-Based Stops

```python
def calculate_stops(entry_price, atr, direction='LONG'):
    """
    Stop loss: 2 × ATR
    Take profit: 3 × ATR
    """
    if direction == 'LONG':
        stop_loss = entry_price - (2 * atr)
        take_profit = entry_price + (3 * atr)
    else:
        stop_loss = entry_price + (2 * atr)
        take_profit = entry_price - (3 * atr)

    return stop_loss, take_profit
```

### Trailing Stop

```python
def update_trailing_stop(entry, current, stop, atr, direction='LONG'):
    """
    Activa trailing después de 2× el riesgo inicial
    """
    initial_risk = abs(entry - stop)
    profit = (current - entry) if direction == 'LONG' else (entry - current)

    if profit > 2 * initial_risk:
        if direction == 'LONG':
            new_stop = current - (atr * 1.0)  # Stop más ajustado
            return max(new_stop, stop)  # Solo subir
        else:
            new_stop = current + (atr * 1.0)
            return min(new_stop, stop)  # Solo bajar

    return stop
```

---

## Validación Pre-Trade

```python
def can_open_position(portfolio, proposed_order):
    """
    Checklist antes de cada operación
    """
    # 1. Verificar drawdown
    if portfolio.drawdown > 0.15:
        return False, "Max drawdown exceeded"

    # 2. Verificar pérdida diaria
    if portfolio.daily_loss > 0.02:
        return False, "Daily loss limit reached"

    # 3. Verificar número de posiciones
    if len(portfolio.positions) >= 20:
        return False, "Max positions reached"

    # 4. Verificar tamaño de posición
    position_value = proposed_order.size * proposed_order.price
    if position_value > portfolio.equity * 0.05:
        return False, "Position size exceeds 5%"

    # 5. Verificar correlación
    if check_high_correlation(portfolio, proposed_order.symbol):
        return False, "High correlation with existing positions"

    return True, "OK"
```

---

## Monitoreo de Riesgo

### Métricas Diarias

```
- Equity actual vs peak (drawdown)
- P&L del día
- Leverage actual
- Correlación entre posiciones
- VaR 95% del portfolio
- Posiciones cerca de stop loss
```

### Alertas Automáticas

| Condición | Acción |
|-----------|--------|
| Drawdown > 10% | Warning email |
| Drawdown > 15% | CRÍTICO + suspensión |
| Daily loss > 1.5% | Warning |
| Daily loss > 2% | Pausa automática |
| Position stop hit | Log + notificación |
