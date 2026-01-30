# Los 7 Principios Fundamentales

## 1. Data-First, No Intuición

**Regla:** Si no está validado estadísticamente, no se implementa.

```
Criterios de validación:
- p-value < 0.05 para todas las hipótesis
- Mínimo 50 observaciones para cualquier patrón
- Múltiples períodos de validación (in-sample + out-of-sample)
```

**MAL:** "Creo que el S&P subirá porque..."
**BIEN:** Analizar datos → Encontrar patrón → Validar estadísticamente → Implementar

## 2. Pequeños Edges Repetidos

**Concepto:** No busques un trade ganador. Busca 51% de precisión × 1,000 trades.

```
Matemáticas:
- Win rate: 51%
- Risk/Reward: 1.5:1
- PnL esperado = (0.51 × 1.5) - (0.49 × 1) = 0.275 por trade

Con 500 trades/año:
- Expectativa = 500 × 0.275 = 137.5 unidades de ganancia
```

## 3. Automatización 100%

**Regla:** Cero intervención humana en decisiones de trading.

- Señales generadas por código
- Órdenes ejecutadas automáticamente
- Stops y targets pre-programados
- Humanos solo monitorean y mantienen

## 4. Gestión de Riesgo Obsesiva

**Límites duros (no negociables):**

| Parámetro | Límite |
|-----------|--------|
| Max Drawdown | 15% |
| Daily Loss | 2% |
| Position Size | 5% max |
| Leverage | 3x máximo |
| Correlación | 0.7 máximo |

## 5. Backtesting Riguroso

**Checklist obligatorio:**

- [ ] Mínimo 10 años de datos
- [ ] Walk-forward validation
- [ ] Out-of-sample testing separado
- [ ] Costos realistas (comisiones + slippage)
- [ ] Monte Carlo simulation
- [ ] Stress testing en eventos extremos

## 6. Investigación Continua

**Los edges se degradan.** Timeline típico:

```
Mes 0:  Edge descubierto (Sharpe 1.8)
Mes 3:  Otros lo copian (Sharpe 1.2)
Mes 6:  Se satura (Sharpe 0.8)
Mes 12: Muere si no evolucionas
```

**Dedicación:**
- 20% del tiempo en investigación nueva
- 3-5 ideas testeadas por mes
- Rotación de estrategias cada 6-12 meses

## 7. Equipo Técnico

**No necesitas traders. Necesitas ingenieros.**

Composición ideal:
- 1 Quant Engineer (arquitectura + estrategias)
- 1 Data Engineer (pipelines + infraestructura)
- 0.5 PM/Risk Manager (supervisión)

Mentalidad: "Resolver el problema" en lugar de "tradear el mercado"
