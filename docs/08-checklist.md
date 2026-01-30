# Checklist de Validación

## Antes de Paper Trading

### Backtesting
- [ ] Sharpe ratio > 1.0 (mínimo 10 años)
- [ ] Max drawdown < 20%
- [ ] Win rate > 45%
- [ ] Profit factor > 1.3
- [ ] 50+ trades en período de prueba
- [ ] Walk-forward testing consistente
- [ ] Out-of-sample validado

### Técnico
- [ ] Unit tests > 80% coverage
- [ ] Integration tests pasan
- [ ] Broker connection estable
- [ ] Data pipeline sin gaps
- [ ] Logging completo configurado

---

## Antes de Live Trading

### Paper Trading (2+ semanas)
- [ ] Sin errores críticos
- [ ] Fills dentro de 0.5% slippage
- [ ] Drawdown < 15%
- [ ] Sharpe > 0.8

### Risk Management
- [ ] Kelly Criterion implementado
- [ ] Position sizing validado
- [ ] Stop losses ejecutan correctamente
- [ ] Límites de drawdown funcionan
- [ ] Correlación monitoreada

### Operacional
- [ ] Dashboard actualiza en tiempo real
- [ ] Alertas funcionando (email/Slack)
- [ ] Backup automático configurado
- [ ] Plan de disaster recovery documentado
- [ ] Runbook para operaciones diarias

---

## Go-Live Checklist

### Día Anterior
- [ ] Revisar posiciones en paper trading
- [ ] Verificar conexión con broker
- [ ] Confirmar capital disponible
- [ ] Revisar límites de riesgo

### Día de Lanzamiento
- [ ] Iniciar con capital mínimo ($10K)
- [ ] Monitorear primera hora activamente
- [ ] Verificar primeras órdenes ejecutadas
- [ ] Confirmar logs y alertas

### Primera Semana
- [ ] Review diario de métricas
- [ ] Comparar fills vs esperado
- [ ] Documentar cualquier anomalía
- [ ] Ajustar parámetros si necesario

---

## Métricas de Éxito

### Semana 1-2
- [ ] Sin errores técnicos críticos
- [ ] Fills dentro de expectativa
- [ ] Sistema estable 24/7

### Mes 1
- [ ] Sharpe > 0.8
- [ ] Drawdown < 10%
- [ ] Escalar a $25K si OK

### Mes 3
- [ ] Sharpe > 1.0
- [ ] Drawdown < 12%
- [ ] Operación completamente automática
