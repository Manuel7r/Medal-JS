"""
Estrategia de Pairs Trading (Arbitraje Estadístico)

Explota la relación de cointegración entre activos correlacionados.
Cuando el spread diverge, espera reversión a la media.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    entry_date: str
    exit_date: str
    direction: str
    entry_spread: float
    exit_spread: float
    pnl: float
    duration_days: int


class PairsTradingStrategy:
    """
    Estrategia de Pairs Trading usando z-score del spread.

    Parámetros:
        lookback: Período para calcular media y std (default: 60)
        entry_z: Z-score para entrada (default: 1.5)
        exit_z: Z-score para salida (default: 0.5)
        stop_z: Z-score para stop loss (default: 3.0)
    """

    def __init__(self,
                 asset1: str,
                 asset2: str,
                 lookback: int = 60,
                 entry_z: float = 1.5,
                 exit_z: float = 0.5,
                 stop_z: float = 3.0):
        self.asset1 = asset1
        self.asset2 = asset2
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

        self.hedge_ratio: Optional[float] = None
        self.trades: List[TradeResult] = []

    def load_data(self, start: str, end: str) -> pd.DataFrame:
        """Carga datos históricos de Yahoo Finance."""
        logger.info(f"Cargando datos para {self.asset1} y {self.asset2}")

        data1 = yf.download(self.asset1, start=start, end=end, progress=False)
        data2 = yf.download(self.asset2, start=start, end=end, progress=False)

        df = pd.DataFrame({
            self.asset1: data1['Close'],
            self.asset2: data2['Close']
        }).dropna()

        logger.info(f"Cargadas {len(df)} filas")
        return df

    def check_cointegration(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """Verifica si el par está cointegrado."""
        _, p_value, _ = coint(df[self.asset1], df[self.asset2])

        is_coint = p_value < 0.05
        status = "COINTEGRADO" if is_coint else "NO COINTEGRADO"
        logger.info(f"Test cointegración p-value: {p_value:.4f} - {status}")

        return p_value, is_coint

    def calculate_hedge_ratio(self, df: pd.DataFrame) -> float:
        """Calcula hedge ratio usando regresión lineal."""
        model = LinearRegression()
        X = df[self.asset2].values.reshape(-1, 1)
        y = df[self.asset1].values
        model.fit(X, y)

        self.hedge_ratio = float(model.coef_[0])
        logger.info(f"Hedge ratio: {self.hedge_ratio:.4f}")

        return self.hedge_ratio

    def calculate_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calcula spread normalizado."""
        return df[self.asset1] - self.hedge_ratio * df[self.asset2]

    def calculate_z_score(self, spread: pd.Series) -> pd.Series:
        """Calcula z-score rolling del spread."""
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        return (spread - mean) / std

    def generate_signal(self, z_score: float, has_position: bool) -> str:
        """Genera señal basada en z-score."""
        if not has_position:
            if z_score > self.entry_z:
                return "SHORT_SPREAD"
            elif z_score < -self.entry_z:
                return "LONG_SPREAD"
        else:
            if abs(z_score) < self.exit_z:
                return "EXIT"
            if abs(z_score) > self.stop_z:
                return "STOP_LOSS"
        return "HOLD"

    def backtest(self, df: pd.DataFrame,
                 initial_capital: float = 100_000) -> Dict:
        """Ejecuta backtest de la estrategia."""
        logger.info("Iniciando backtest")

        spread = self.calculate_spread(df)
        z_score = self.calculate_z_score(spread)

        # Estado
        position = 0  # 1: long spread, -1: short spread, 0: flat
        entry_price = None
        entry_date = None
        equity = initial_capital
        equity_curve = [initial_capital]

        for i in range(self.lookback, len(df)):
            date = df.index[i]
            z = z_score.iloc[i]
            current_spread = spread.iloc[i]

            if np.isnan(z):
                equity_curve.append(equity)
                continue

            # Señales de entrada
            if position == 0:
                if z > self.entry_z:
                    position = -1
                    entry_price = current_spread
                    entry_date = date
                elif z < -self.entry_z:
                    position = 1
                    entry_price = current_spread
                    entry_date = date

            # Señales de salida
            elif position != 0:
                should_exit = abs(z) < self.exit_z or abs(z) > self.stop_z

                if should_exit:
                    # Calcular P&L
                    if position == 1:
                        pnl = (current_spread - entry_price) * 1000
                    else:
                        pnl = (entry_price - current_spread) * 1000

                    equity += pnl

                    self.trades.append(TradeResult(
                        entry_date=str(entry_date.date()),
                        exit_date=str(date.date()),
                        direction='LONG' if position == 1 else 'SHORT',
                        entry_spread=entry_price,
                        exit_spread=current_spread,
                        pnl=pnl,
                        duration_days=(date - entry_date).days
                    ))

                    position = 0
                    entry_price = None
                    entry_date = None

            equity_curve.append(equity)

        metrics = self._calculate_metrics(equity_curve, initial_capital)

        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': self.trades
        }

    def _calculate_metrics(self, equity_curve: List[float],
                          initial_capital: float) -> Dict:
        """Calcula métricas de performance."""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Sharpe Ratio (anualizado)
        sharpe = 0
        if np.std(returns) > 0:
            sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))

        # Max Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        # Trade stats
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl < 0]

        win_rate = len(winning) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'total_return': (equity[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': np.mean([t.pnl for t in self.trades]) if self.trades else 0
        }

    def print_results(self, results: Dict):
        """Imprime resultados formateados."""
        print("\n" + "=" * 60)
        print("RESULTADOS DEL BACKTEST")
        print("=" * 60)
        print(f"Par: {self.asset1} / {self.asset2}")
        print(f"Hedge Ratio: {self.hedge_ratio:.4f}")
        print("-" * 60)

        metrics = results['metrics']
        print(f"{'Retorno Total:':<25} {metrics['total_return']:>10.2%}")
        print(f"{'Sharpe Ratio:':<25} {metrics['sharpe_ratio']:>10.2f}")
        print(f"{'Max Drawdown:':<25} {metrics['max_drawdown']:>10.2%}")
        print(f"{'Número de Trades:':<25} {metrics['num_trades']:>10}")
        print(f"{'Win Rate:':<25} {metrics['win_rate']:>10.2%}")
        print(f"{'Profit Factor:':<25} {metrics['profit_factor']:>10.2f}")
        print(f"{'P&L Promedio por Trade:':<25} ${metrics['avg_trade_pnl']:>9.2f}")


def find_cointegrated_pairs(symbols: List[str],
                           start: str,
                           end: str,
                           p_threshold: float = 0.05) -> List[Tuple]:
    """
    Encuentra pares cointegrados en una lista de símbolos.

    Args:
        symbols: Lista de símbolos a analizar
        start: Fecha inicio
        end: Fecha fin
        p_threshold: Umbral de p-value

    Returns:
        Lista de tuplas (symbol1, symbol2, p_value)
    """
    logger.info(f"Buscando pares cointegrados entre {len(symbols)} símbolos")

    # Descargar datos
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if len(df) > 100:
                data[symbol] = df['Close']
        except Exception as e:
            logger.warning(f"Error descargando {symbol}: {e}")

    prices = pd.DataFrame(data).dropna()
    logger.info(f"Datos disponibles para {len(prices.columns)} símbolos")

    # Buscar pares cointegrados
    pairs = []
    symbols_list = list(prices.columns)

    for i in range(len(symbols_list)):
        for j in range(i + 1, len(symbols_list)):
            try:
                _, p_value, _ = coint(prices[symbols_list[i]],
                                      prices[symbols_list[j]])
                if p_value < p_threshold:
                    pairs.append((symbols_list[i], symbols_list[j], p_value))
            except Exception:
                pass

    pairs.sort(key=lambda x: x[2])
    logger.info(f"Encontrados {len(pairs)} pares cointegrados")

    return pairs


def main():
    """Ejemplo de uso."""
    # Configuración
    ASSET1 = "JPM"
    ASSET2 = "BAC"
    START = "2015-01-01"
    END = "2025-01-01"

    # Crear estrategia
    strategy = PairsTradingStrategy(
        asset1=ASSET1,
        asset2=ASSET2,
        lookback=60,
        entry_z=1.5,
        exit_z=0.5,
        stop_z=3.0
    )

    # Cargar datos
    df = strategy.load_data(START, END)

    # Verificar cointegración
    p_value, is_coint = strategy.check_cointegration(df)

    if not is_coint:
        logger.warning("Par no cointegrado - resultados pueden ser no confiables")

    # Calcular hedge ratio
    strategy.calculate_hedge_ratio(df)

    # Ejecutar backtest
    results = strategy.backtest(df)

    # Mostrar resultados
    strategy.print_results(results)

    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
