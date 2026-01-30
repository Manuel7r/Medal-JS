"""
Motor de Backtesting

Framework para validar estrategias con datos históricos.
Incluye modelado de comisiones, slippage y métricas de performance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representa un trade ejecutado."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    direction: str = 'LONG'  # 'LONG' o 'SHORT'
    pnl: float = 0
    commission: float = 0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def close(self, date: datetime, price: float, commission: float = 0):
        """Cierra el trade."""
        self.exit_date = date
        self.exit_price = price
        self.commission += commission

        if self.direction == 'LONG':
            self.pnl = (price - self.entry_price) * self.quantity - self.commission
        else:
            self.pnl = (self.entry_price - price) * self.quantity - self.commission


@dataclass
class BacktestResult:
    """Resultado de un backtest."""
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Motor de backtesting para estrategias de trading.

    Args:
        initial_capital: Capital inicial
        commission: Comisión por trade (porcentaje)
        slippage: Slippage estimado (porcentaje)
    """

    def __init__(self,
                 initial_capital: float = 100_000,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.equity = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []

    def reset(self):
        """Reinicia el estado del backtester."""
        self.equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.positions = {}
        self.closed_trades = []

    def run(self,
            data: pd.DataFrame,
            signal_func: Callable[[pd.DataFrame, int], Dict],
            symbols: List[str]) -> BacktestResult:
        """
        Ejecuta backtest con una función de señales.

        Args:
            data: DataFrame con datos OHLCV (MultiIndex o columnas por símbolo)
            signal_func: Función que retorna señales dado data y índice
            symbols: Lista de símbolos a operar

        Returns:
            BacktestResult con equity curve, trades y métricas
        """
        logger.info("Iniciando backtest")
        self.reset()

        for i in range(1, len(data)):
            date = data.index[i]

            # Obtener señales
            signals = signal_func(data, i)

            for symbol in symbols:
                if symbol not in signals:
                    continue

                signal = signals[symbol]
                price = self._get_price(data, symbol, i)

                if price is None:
                    continue

                # Procesar señal
                if signal == 'BUY' and symbol not in self.positions:
                    self._open_position(symbol, date, price, 'LONG')

                elif signal == 'SELL' and symbol not in self.positions:
                    self._open_position(symbol, date, price, 'SHORT')

                elif signal == 'EXIT' and symbol in self.positions:
                    self._close_position(symbol, date, price)

            # Actualizar equity (mark-to-market)
            self._update_equity(data, i)
            self.equity_curve.append(self.equity)

        # Cerrar posiciones abiertas al final
        self._close_all_positions(data.index[-1], data, len(data) - 1)

        # Calcular métricas
        metrics = self.calculate_metrics()

        return BacktestResult(
            equity_curve=self.equity_curve,
            trades=self.closed_trades,
            metrics=metrics
        )

    def _get_price(self, data: pd.DataFrame, symbol: str, idx: int) -> Optional[float]:
        """Obtiene precio de cierre para un símbolo."""
        try:
            if symbol in data.columns:
                return data[symbol].iloc[idx]
            elif ('Close', symbol) in data.columns:
                return data[('Close', symbol)].iloc[idx]
            return None
        except Exception:
            return None

    def _open_position(self, symbol: str, date: datetime,
                       price: float, direction: str):
        """Abre una posición."""
        # Aplicar slippage
        if direction == 'LONG':
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)

        # Calcular tamaño (simplificado: 5% del capital)
        position_value = self.equity * 0.05
        quantity = position_value / exec_price

        # Calcular comisión
        commission_cost = position_value * self.commission

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=exec_price,
            quantity=quantity,
            direction=direction,
            commission=commission_cost
        )

        self.positions[symbol] = trade
        logger.debug(f"Abierta posición {direction} en {symbol} @ {exec_price:.2f}")

    def _close_position(self, symbol: str, date: datetime, price: float):
        """Cierra una posición."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        # Aplicar slippage
        if trade.direction == 'LONG':
            exec_price = price * (1 - self.slippage)
        else:
            exec_price = price * (1 + self.slippage)

        # Comisión de cierre
        commission_cost = trade.quantity * exec_price * self.commission

        trade.close(date, exec_price, commission_cost)

        self.equity += trade.pnl
        self.closed_trades.append(trade)
        del self.positions[symbol]

        logger.debug(f"Cerrada posición {symbol} @ {exec_price:.2f}, P&L: {trade.pnl:.2f}")

    def _close_all_positions(self, date: datetime, data: pd.DataFrame, idx: int):
        """Cierra todas las posiciones abiertas."""
        for symbol in list(self.positions.keys()):
            price = self._get_price(data, symbol, idx)
            if price:
                self._close_position(symbol, date, price)

    def _update_equity(self, data: pd.DataFrame, idx: int):
        """Actualiza equity basado en posiciones abiertas (mark-to-market)."""
        unrealized_pnl = 0

        for symbol, trade in self.positions.items():
            price = self._get_price(data, symbol, idx)
            if price is None:
                continue

            if trade.direction == 'LONG':
                unrealized_pnl += (price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl += (trade.entry_price - price) * trade.quantity

        # El equity base más ganancias realizadas ya está actualizado
        # Solo añadimos unrealized para la curva

    def calculate_metrics(self) -> Dict:
        """Calcula métricas de performance."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Filtrar NaN e infinitos
        returns = returns[np.isfinite(returns)]

        metrics = {}

        # Retorno total
        metrics['total_return'] = (equity[-1] - self.initial_capital) / self.initial_capital

        # CAGR
        years = len(equity) / 252
        if years > 0 and equity[-1] > 0:
            metrics['cagr'] = (equity[-1] / self.initial_capital) ** (1 / years) - 1
        else:
            metrics['cagr'] = 0

        # Sharpe Ratio (anualizado, rf = 2%)
        if len(returns) > 0 and np.std(returns) > 0:
            excess_returns = returns - (0.02 / 252)
            metrics['sharpe_ratio'] = (np.mean(excess_returns) * 252) / (np.std(returns) * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            metrics['sortino_ratio'] = (np.mean(returns) * 252) / (np.std(downside_returns) * np.sqrt(252))
        else:
            metrics['sortino_ratio'] = 0

        # Max Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        metrics['max_drawdown'] = drawdown.min()

        # Calmar Ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0

        # Estadísticas de trades
        if self.closed_trades:
            pnls = [t.pnl for t in self.closed_trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]

            metrics['num_trades'] = len(self.closed_trades)
            metrics['win_rate'] = len(winning) / len(pnls) if pnls else 0
            metrics['avg_trade_pnl'] = np.mean(pnls)

            if winning and losing:
                metrics['profit_factor'] = sum(winning) / abs(sum(losing))
            else:
                metrics['profit_factor'] = float('inf') if winning else 0

            metrics['avg_winning_trade'] = np.mean(winning) if winning else 0
            metrics['avg_losing_trade'] = np.mean(losing) if losing else 0
        else:
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0
            metrics['avg_trade_pnl'] = 0
            metrics['profit_factor'] = 0

        return metrics

    def print_metrics(self, metrics: Dict):
        """Imprime métricas formateadas."""
        print("\n" + "=" * 50)
        print("MÉTRICAS DE PERFORMANCE")
        print("=" * 50)

        print(f"{'Retorno Total:':<25} {metrics['total_return']:>10.2%}")
        print(f"{'CAGR:':<25} {metrics['cagr']:>10.2%}")
        print(f"{'Sharpe Ratio:':<25} {metrics['sharpe_ratio']:>10.2f}")
        print(f"{'Sortino Ratio:':<25} {metrics['sortino_ratio']:>10.2f}")
        print(f"{'Max Drawdown:':<25} {metrics['max_drawdown']:>10.2%}")
        print(f"{'Calmar Ratio:':<25} {metrics['calmar_ratio']:>10.2f}")

        print("-" * 50)
        print(f"{'Número de Trades:':<25} {metrics['num_trades']:>10}")
        print(f"{'Win Rate:':<25} {metrics['win_rate']:>10.2%}")
        print(f"{'Profit Factor:':<25} {metrics['profit_factor']:>10.2f}")
        print(f"{'P&L Promedio:':<25} ${metrics['avg_trade_pnl']:>9.2f}")


# Ejemplo de uso
if __name__ == "__main__":
    import yfinance as yf

    # Descargar datos
    symbols = ['AAPL', 'MSFT']
    data = yf.download(symbols, start='2020-01-01', end='2024-01-01')['Close']

    # Función de señales simple (cruce de medias)
    def simple_signal(data: pd.DataFrame, idx: int) -> Dict:
        signals = {}
        for symbol in data.columns:
            if idx < 50:
                continue

            prices = data[symbol].iloc[:idx]
            fast_ma = prices.rolling(10).mean().iloc[-1]
            slow_ma = prices.rolling(50).mean().iloc[-1]

            if fast_ma > slow_ma:
                signals[symbol] = 'BUY'
            elif fast_ma < slow_ma:
                signals[symbol] = 'EXIT'

        return signals

    # Ejecutar backtest
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, simple_signal, symbols)

    # Mostrar resultados
    engine.print_metrics(result.metrics)
