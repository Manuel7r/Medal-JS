"""
Pairs Trading Adaptado para Criptomonedas (Binance)

Diferencias con la versión de stocks:
- Timeframe: 1h o 4h (no diario)
- Mercado 24/7
- Mayor volatilidad → ATR multipliers ajustados
- Z-score de entrada más alto (2.0 vs 1.5)
- Position size más conservador (3% vs 5%)
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar cliente de Binance
try:
    from binance_client import BinanceClient
except ImportError:
    BinanceClient = None
    logger.warning("binance_client.py no encontrado. Solo disponible modo backtest.")


@dataclass
class CryptoTradeResult:
    entry_date: str
    exit_date: str
    direction: str
    entry_spread: float
    exit_spread: float
    pnl_usdt: float
    duration_hours: float
    exit_reason: str  # 'mean_reversion', 'stop_loss', 'timeout'


class CryptoPairsTradingStrategy:
    """
    Pairs Trading para crypto con ajustes de volatilidad.

    Parámetros ajustados para crypto:
    - lookback: 168 (7 días en velas horarias)
    - entry_z: 2.0 (más conservador por volatilidad)
    - exit_z: 0.5
    - stop_z: 3.5 (más holgado para flash crashes)
    - max_hold_hours: 168 (7 días máximo)
    """

    def __init__(self,
                 asset1: str = 'BTC/USDT',
                 asset2: str = 'ETH/USDT',
                 lookback: int = 168,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5,
                 stop_z: float = 3.5,
                 max_hold_hours: int = 168,
                 position_pct: float = 0.03):
        self.asset1 = asset1
        self.asset2 = asset2
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_hold_hours = max_hold_hours
        self.position_pct = position_pct

        self.hedge_ratio: Optional[float] = None
        self.trades: List[CryptoTradeResult] = []

    def load_data_from_binance(self,
                                since: str = '2023-01-01',
                                timeframe: str = '1h') -> pd.DataFrame:
        """Carga datos desde Binance."""
        if BinanceClient is None:
            raise ImportError("binance_client.py necesario para datos en vivo")

        client = BinanceClient(testnet=False, market_type='spot')

        data1 = client.fetch_full_history(self.asset1, timeframe, since)
        data2 = client.fetch_full_history(self.asset2, timeframe, since)

        df = pd.DataFrame({
            self.asset1: data1['close'],
            self.asset2: data2['close']
        }).dropna()

        logger.info(f"Cargadas {len(df)} velas horarias desde {df.index[0]}")
        return df

    def load_data_from_csv(self, path1: str, path2: str) -> pd.DataFrame:
        """Carga datos desde CSV para backtest offline."""
        data1 = pd.read_csv(path1, index_col='timestamp', parse_dates=True)
        data2 = pd.read_csv(path2, index_col='timestamp', parse_dates=True)

        df = pd.DataFrame({
            self.asset1: data1['close'],
            self.asset2: data2['close']
        }).dropna()

        return df

    def check_cointegration(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """Verifica cointegración del par crypto."""
        _, p_value, _ = coint(df[self.asset1], df[self.asset2])

        is_coint = p_value < 0.05
        logger.info(
            f"Cointegración {self.asset1}/{self.asset2}: "
            f"p-value={p_value:.4f} - "
            f"{'COINTEGRADO' if is_coint else 'NO COINTEGRADO'}"
        )
        return p_value, is_coint

    def calculate_hedge_ratio(self,
                               df: pd.DataFrame,
                               rolling_window: Optional[int] = None) -> float:
        """
        Calcula hedge ratio. En crypto, usar rolling para adaptarse a
        cambios de régimen más frecuentes.
        """
        if rolling_window:
            # Rolling hedge ratio (más adaptativo)
            recent = df.tail(rolling_window)
        else:
            recent = df

        model = LinearRegression()
        X = recent[self.asset2].values.reshape(-1, 1)
        y = recent[self.asset1].values
        model.fit(X, y)

        self.hedge_ratio = float(model.coef_[0])
        logger.info(f"Hedge ratio: {self.hedge_ratio:.4f}")
        return self.hedge_ratio

    def calculate_spread_and_zscore(self,
                                     df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calcula spread y z-score."""
        spread = df[self.asset1] - self.hedge_ratio * df[self.asset2]
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z_score = (spread - mean) / std
        return spread, z_score

    def backtest(self,
                 df: pd.DataFrame,
                 initial_capital: float = 10_000) -> Dict:
        """
        Ejecuta backtest de la estrategia crypto.

        Args:
            df: DataFrame con precios de ambos activos
            initial_capital: Capital inicial en USDT
        """
        logger.info(f"Iniciando backtest crypto: {self.asset1}/{self.asset2}")

        # Recalcular hedge ratio con datos de training
        train_size = min(self.lookback * 2, len(df) // 3)
        self.calculate_hedge_ratio(df.head(train_size))

        spread, z_score = self.calculate_spread_and_zscore(df)

        # Estado
        position = 0  # 1: long spread, -1: short spread
        entry_price = None
        entry_date = None
        equity = initial_capital
        equity_curve = [initial_capital]
        commission_rate = 0.001  # 0.1% Binance

        for i in range(self.lookback, len(df)):
            date = df.index[i]
            z = z_score.iloc[i]
            current_spread = spread.iloc[i]

            if np.isnan(z):
                equity_curve.append(equity)
                continue

            # ─── Entrada ──────────────────────────────────
            if position == 0:
                if z > self.entry_z:
                    position = -1
                    entry_price = current_spread
                    entry_date = date
                    logger.debug(f"SHORT SPREAD @ {date}, z={z:.2f}")

                elif z < -self.entry_z:
                    position = 1
                    entry_price = current_spread
                    entry_date = date
                    logger.debug(f"LONG SPREAD @ {date}, z={z:.2f}")

            # ─── Salida ───────────────────────────────────
            elif position != 0:
                exit_reason = None

                # Exit por reversión a media
                if abs(z) < self.exit_z:
                    exit_reason = 'mean_reversion'

                # Exit por stop loss
                elif abs(z) > self.stop_z:
                    exit_reason = 'stop_loss'

                # Exit por timeout
                elif entry_date:
                    hours_held = (date - entry_date).total_seconds() / 3600
                    if hours_held > self.max_hold_hours:
                        exit_reason = 'timeout'

                if exit_reason:
                    # P&L basado en tamaño de posición (3% del capital)
                    position_value = equity * self.position_pct
                    spread_return = (current_spread - entry_price) / abs(entry_price)

                    if position == 1:
                        pnl = position_value * spread_return
                    else:
                        pnl = position_value * (-spread_return)

                    # Descontar comisiones (entrada + salida)
                    commission = position_value * commission_rate * 2 * 2  # 2 legs × 2 trades
                    pnl -= commission

                    equity += pnl

                    duration = (date - entry_date).total_seconds() / 3600

                    self.trades.append(CryptoTradeResult(
                        entry_date=str(entry_date),
                        exit_date=str(date),
                        direction='LONG' if position == 1 else 'SHORT',
                        entry_spread=entry_price,
                        exit_spread=current_spread,
                        pnl_usdt=pnl,
                        duration_hours=duration,
                        exit_reason=exit_reason
                    ))

                    logger.debug(f"EXIT ({exit_reason}) @ {date}, P&L=${pnl:.2f}")
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
        returns = returns[np.isfinite(returns)]

        # Sharpe (anualizado para crypto: 24h × 365)
        hours_per_year = 8760
        periods = len(returns)
        annualization = np.sqrt(hours_per_year) if periods > 0 else 1

        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = (np.mean(returns) * hours_per_year) / (np.std(returns) * annualization)

        # Max Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        # Trade stats
        winning = [t for t in self.trades if t.pnl_usdt > 0]
        losing = [t for t in self.trades if t.pnl_usdt < 0]

        win_rate = len(winning) / len(self.trades) if self.trades else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        avg_duration = np.mean([t.duration_hours for t in self.trades]) if self.trades else 0

        return {
            'total_return': (equity[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_trade_pnl': np.mean([t.pnl_usdt for t in self.trades]) if self.trades else 0,
            'avg_duration_hours': avg_duration,
            'exit_reasons': exit_reasons,
            'total_commissions': sum(abs(t.pnl_usdt) * 0.004 for t in self.trades)
        }

    def print_results(self, results: Dict):
        """Imprime resultados formateados."""
        print("\n" + "=" * 60)
        print("RESULTADOS BACKTEST CRYPTO PAIRS TRADING")
        print("=" * 60)
        print(f"Par: {self.asset1} / {self.asset2}")
        print(f"Hedge Ratio: {self.hedge_ratio:.4f}")
        print(f"Lookback: {self.lookback}h | Entry Z: {self.entry_z} | Stop Z: {self.stop_z}")
        print("-" * 60)

        m = results['metrics']
        print(f"{'Retorno Total:':<25} {m['total_return']:>10.2%}")
        print(f"{'Sharpe Ratio:':<25} {m['sharpe_ratio']:>10.2f}")
        print(f"{'Max Drawdown:':<25} {m['max_drawdown']:>10.2%}")
        print(f"{'Número de Trades:':<25} {m['num_trades']:>10}")
        print(f"{'Win Rate:':<25} {m['win_rate']:>10.2%}")
        print(f"{'P&L Promedio:':<25} ${m['avg_trade_pnl']:>9.2f}")
        print(f"{'Duración Promedio:':<25} {m['avg_duration_hours']:>8.1f}h")

        print("\nRazones de Salida:")
        for reason, count in m.get('exit_reasons', {}).items():
            print(f"  {reason}: {count}")


def scan_crypto_pairs(symbols: List[str],
                      client: Optional['BinanceClient'] = None,
                      timeframe: str = '1h',
                      limit: int = 500) -> List[Tuple]:
    """
    Escanea pares de criptomonedas para encontrar cointegrados.

    Args:
        symbols: Lista de pares (ej: ['BTC/USDT', 'ETH/USDT', ...])
        client: Cliente de Binance
        timeframe: Timeframe para análisis
        limit: Número de velas

    Returns:
        Lista de tuplas (pair1, pair2, p_value)
    """
    logger.info(f"Escaneando {len(symbols)} pares para cointegración")

    # Obtener datos
    prices = {}
    for symbol in symbols:
        try:
            if client:
                df = client.fetch_ohlcv(symbol, timeframe, limit)
                if not df.empty:
                    prices[symbol] = df['close']
            else:
                logger.warning(f"Sin cliente para {symbol}, saltando")
        except Exception as e:
            logger.warning(f"Error con {symbol}: {e}")

    if len(prices) < 2:
        logger.warning("Menos de 2 pares con datos")
        return []

    prices_df = pd.DataFrame(prices).dropna()

    # Buscar pares cointegrados
    pairs = []
    symbols_list = list(prices_df.columns)

    for i in range(len(symbols_list)):
        for j in range(i + 1, len(symbols_list)):
            try:
                _, p_value, _ = coint(
                    prices_df[symbols_list[i]],
                    prices_df[symbols_list[j]]
                )
                if p_value < 0.05:
                    pairs.append((symbols_list[i], symbols_list[j], p_value))
            except Exception:
                pass

    pairs.sort(key=lambda x: x[2])
    logger.info(f"Encontrados {len(pairs)} pares cointegrados")

    for p in pairs[:10]:
        logger.info(f"  {p[0]} / {p[1]}: p-value={p[2]:.4f}")

    return pairs


# ─── EJEMPLO DE USO ──────────────────────────────────────────────

def main():
    """Ejemplo: backtest de crypto pairs trading."""

    # Configuración
    strategy = CryptoPairsTradingStrategy(
        asset1='BTC/USDT',
        asset2='ETH/USDT',
        lookback=168,       # 7 días en horas
        entry_z=2.0,        # Más conservador que stocks
        exit_z=0.5,
        stop_z=3.5,         # Más holgado para flash crashes
        max_hold_hours=168,  # Max 7 días
        position_pct=0.03   # 3% por trade
    )

    # Opción 1: Datos desde Binance
    try:
        df = strategy.load_data_from_binance(since='2023-01-01', timeframe='1h')
    except Exception as e:
        logger.warning(f"No se pudo conectar a Binance: {e}")
        logger.info("Generando datos sintéticos para demo...")

        # Datos sintéticos para demo
        np.random.seed(42)
        n = 8760  # 1 año de datos horarios
        dates = pd.date_range('2023-01-01', periods=n, freq='h')

        btc_base = 30000
        btc_returns = np.random.normal(0.0001, 0.01, n)
        btc_prices = btc_base * np.cumprod(1 + btc_returns)

        # ETH correlacionado con BTC
        eth_base = 2000
        eth_returns = 0.85 * btc_returns + 0.15 * np.random.normal(0, 0.01, n)
        eth_prices = eth_base * np.cumprod(1 + eth_returns)

        df = pd.DataFrame({
            'BTC/USDT': btc_prices,
            'ETH/USDT': eth_prices
        }, index=dates)

    # Verificar cointegración
    p_value, is_coint = strategy.check_cointegration(df)

    if not is_coint:
        logger.warning("Par no cointegrado. Resultados pueden no ser confiables.")

    # Calcular hedge ratio
    strategy.calculate_hedge_ratio(df)

    # Backtest
    results = strategy.backtest(df, initial_capital=10_000)

    # Resultados
    strategy.print_results(results)

    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
