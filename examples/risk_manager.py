"""
Gestión de Riesgo

Implementa position sizing (Kelly Criterion), límites de portfolio
y stop loss dinámico.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Configuración de límites de riesgo."""
    max_drawdown: float = 0.15       # 15%
    daily_loss_limit: float = 0.02   # 2%
    max_position_pct: float = 0.05   # 5% por posición
    max_leverage: float = 3.0        # 3x
    max_correlation: float = 0.7     # Entre posiciones
    max_positions: int = 20          # Número máximo


class KellyPositionSizer:
    """
    Calcula tamaño de posición usando Kelly Criterion.

    Usa 25% del Kelly óptimo para mayor seguridad.
    """

    def __init__(self,
                 kelly_fraction: float = 0.25,
                 max_position_pct: float = 0.05,
                 max_leverage: float = 3.0):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage

    def calculate_kelly(self, win_prob: float,
                        risk_reward: float = 1.5) -> float:
        """
        Calcula fracción óptima de Kelly.

        f* = (p × b - q) / b

        Args:
            win_prob: Probabilidad de ganar (0-1)
            risk_reward: Ratio riesgo/recompensa

        Returns:
            Fracción de capital a arriesgar
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0

        q = 1 - win_prob
        kelly = (win_prob * risk_reward - q) / risk_reward

        # Aplicar fracción conservadora
        safe_kelly = kelly * self.kelly_fraction

        # Límites
        return max(0, min(safe_kelly, self.max_position_pct))

    def calculate_position_size(self,
                                equity: float,
                                entry_price: float,
                                stop_loss: float,
                                win_prob: float = 0.51) -> float:
        """
        Calcula tamaño de posición en unidades.

        Args:
            equity: Capital actual
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            win_prob: Probabilidad estimada de éxito

        Returns:
            Número de unidades a comprar
        """
        kelly_pct = self.calculate_kelly(win_prob)
        risk_amount = equity * kelly_pct

        # Riesgo por unidad
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0

        position_size = risk_amount / risk_per_unit

        # Aplicar límite de leverage
        max_size = (equity * self.max_leverage) / entry_price

        return min(position_size, max_size)


class PortfolioRiskManager:
    """
    Gestiona riesgo a nivel de portfolio.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.peak_equity = 0
        self.daily_start_equity = 0
        self.positions: Dict[str, Dict] = {}

    def reset_daily(self, current_equity: float):
        """Reinicia contadores diarios."""
        self.daily_start_equity = current_equity

    def update_peak(self, current_equity: float):
        """Actualiza pico de equity."""
        self.peak_equity = max(self.peak_equity, current_equity)

    def get_current_drawdown(self, current_equity: float) -> float:
        """Calcula drawdown actual."""
        if self.peak_equity == 0:
            return 0
        return 1 - (current_equity / self.peak_equity)

    def get_daily_loss(self, current_equity: float) -> float:
        """Calcula pérdida diaria."""
        if self.daily_start_equity == 0:
            return 0
        return 1 - (current_equity / self.daily_start_equity)

    def check_drawdown_limit(self, current_equity: float) -> Tuple[bool, str]:
        """Verifica límite de drawdown."""
        self.update_peak(current_equity)
        dd = self.get_current_drawdown(current_equity)

        if dd > self.limits.max_drawdown:
            return False, f"Max drawdown exceeded: {dd:.1%}"
        return True, "OK"

    def check_daily_loss_limit(self, current_equity: float) -> Tuple[bool, str]:
        """Verifica límite de pérdida diaria."""
        daily_loss = self.get_daily_loss(current_equity)

        if daily_loss > self.limits.daily_loss_limit:
            return False, f"Daily loss limit exceeded: {daily_loss:.1%}"
        return True, "OK"

    def check_position_limit(self, proposed_value: float,
                             equity: float) -> Tuple[bool, str]:
        """Verifica límite de tamaño de posición."""
        pct = proposed_value / equity

        if pct > self.limits.max_position_pct:
            return False, f"Position size exceeds limit: {pct:.1%}"
        return True, "OK"

    def check_max_positions(self) -> Tuple[bool, str]:
        """Verifica número máximo de posiciones."""
        if len(self.positions) >= self.limits.max_positions:
            return False, f"Max positions reached: {len(self.positions)}"
        return True, "OK"

    def can_open_position(self,
                          symbol: str,
                          equity: float,
                          proposed_size: float,
                          proposed_price: float) -> Tuple[bool, str]:
        """
        Validación completa pre-trade.

        Returns:
            Tuple (puede_operar, razón)
        """
        # 1. Verificar drawdown
        ok, msg = self.check_drawdown_limit(equity)
        if not ok:
            logger.warning(f"Trade rechazado: {msg}")
            return False, msg

        # 2. Verificar pérdida diaria
        ok, msg = self.check_daily_loss_limit(equity)
        if not ok:
            logger.warning(f"Trade rechazado: {msg}")
            return False, msg

        # 3. Verificar número de posiciones
        ok, msg = self.check_max_positions()
        if not ok:
            logger.warning(f"Trade rechazado: {msg}")
            return False, msg

        # 4. Verificar tamaño de posición
        position_value = proposed_size * proposed_price
        ok, msg = self.check_position_limit(position_value, equity)
        if not ok:
            logger.warning(f"Trade rechazado: {msg}")
            return False, msg

        logger.info(f"Trade aprobado para {symbol}")
        return True, "OK"


class DynamicStopManager:
    """
    Gestiona stop loss y take profit dinámicos basados en ATR.
    """

    def __init__(self,
                 atr_mult_sl: float = 2.0,
                 atr_mult_tp: float = 3.0,
                 trail_activation: float = 2.0):
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.trail_activation = trail_activation

    @staticmethod
    def calculate_atr(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      period: int = 14) -> pd.Series:
        """Calcula Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_initial_stops(self,
                                entry_price: float,
                                atr: float,
                                direction: str = 'LONG') -> Tuple[float, float]:
        """
        Calcula stop loss y take profit inicial.

        Returns:
            Tuple (stop_loss, take_profit)
        """
        if direction == 'LONG':
            stop_loss = entry_price - (atr * self.atr_mult_sl)
            take_profit = entry_price + (atr * self.atr_mult_tp)
        else:
            stop_loss = entry_price + (atr * self.atr_mult_sl)
            take_profit = entry_price - (atr * self.atr_mult_tp)

        return stop_loss, take_profit

    def update_trailing_stop(self,
                             entry_price: float,
                             current_price: float,
                             current_stop: float,
                             atr: float,
                             direction: str = 'LONG') -> float:
        """
        Actualiza trailing stop si aplica.

        Activa trailing después de 2× el riesgo inicial.
        """
        initial_risk = abs(entry_price - current_stop)

        if direction == 'LONG':
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price

        # Solo activar trailing si ganancia > trail_activation × riesgo
        if profit > initial_risk * self.trail_activation:
            if direction == 'LONG':
                new_stop = current_price - (atr * self.atr_mult_sl * 0.5)
                return max(new_stop, current_stop)  # Solo subir
            else:
                new_stop = current_price + (atr * self.atr_mult_sl * 0.5)
                return min(new_stop, current_stop)  # Solo bajar

        return current_stop


# Ejemplo de uso
if __name__ == "__main__":
    # Position Sizer
    sizer = KellyPositionSizer()

    # Ejemplo: win_prob = 55%, entry = $100, stop = $95
    size = sizer.calculate_position_size(
        equity=100_000,
        entry_price=100,
        stop_loss=95,
        win_prob=0.55
    )
    print(f"Tamaño de posición calculado: {size:.2f} unidades")

    # Risk Manager
    risk_mgr = PortfolioRiskManager()
    risk_mgr.peak_equity = 100_000
    risk_mgr.daily_start_equity = 100_000

    # Verificar si puede abrir posición
    can_trade, reason = risk_mgr.can_open_position(
        symbol="AAPL",
        equity=98_000,  # 2% drawdown
        proposed_size=50,
        proposed_price=100
    )
    print(f"¿Puede operar? {can_trade} - {reason}")

    # Stop Manager
    stop_mgr = DynamicStopManager()

    # Calcular stops con ATR = $2
    sl, tp = stop_mgr.calculate_initial_stops(
        entry_price=100,
        atr=2.0,
        direction='LONG'
    )
    print(f"Stop Loss: ${sl:.2f}, Take Profit: ${tp:.2f}")
