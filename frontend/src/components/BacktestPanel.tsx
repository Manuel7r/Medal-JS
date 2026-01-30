import { useState, useEffect, useCallback } from 'react';
import type { BacktestMetrics } from '../types';

interface BacktestEntry {
  strategy: string;
  symbol: string;
  metrics: BacktestMetrics;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(2)}%`;
}

export default function BacktestPanel() {
  const [results, setResults] = useState<BacktestEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/backtest/all');
      if (res.ok) {
        const data = await res.json();
        setResults(data);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading) return null;
  if (results.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h2 className="text-lg font-semibold mb-4">Backtest Results</h2>
        <p className="text-slate-500 text-sm">Backtests running... results will appear shortly</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Backtest Results</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left py-2">Strategy</th>
              <th className="text-left py-2">Symbol</th>
              <th className="text-right py-2">Sharpe</th>
              <th className="text-right py-2">Return</th>
              <th className="text-right py-2">Max DD</th>
              <th className="text-right py-2">Win Rate</th>
              <th className="text-right py-2">Trades</th>
              <th className="text-right py-2">Profit Factor</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => {
              const isBest = results.length > 0 &&
                r.metrics.sharpe_ratio === Math.max(...results.map(x => x.metrics.sharpe_ratio));
              return (
              <tr key={i} className={`border-b border-slate-700/50 ${isBest ? 'bg-emerald-500/5' : ''}`}>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    r.strategy === 'MLEnsemble' ? 'bg-purple-500/20 text-purple-400' :
                    r.strategy === 'Aggregator' ? 'bg-amber-500/20 text-amber-400' :
                    r.strategy === 'PairsTrading' ? 'bg-cyan-500/20 text-cyan-400' :
                    'bg-indigo-500/20 text-indigo-400'
                  }`}>
                    {r.strategy}{isBest ? ' *' : ''}
                  </span>
                </td>
                <td className="py-2 font-medium">{r.symbol}</td>
                <td className={`text-right py-2 font-mono ${r.metrics.sharpe_ratio >= 1 ? 'text-emerald-400' : r.metrics.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {r.metrics.sharpe_ratio.toFixed(2)}
                </td>
                <td className={`text-right py-2 ${r.metrics.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {pct(r.metrics.total_return)}
                </td>
                <td className="text-right py-2 text-red-400">{pct(r.metrics.max_drawdown)}</td>
                <td className="text-right py-2">{pct(r.metrics.win_rate)}</td>
                <td className="text-right py-2 text-slate-300">{r.metrics.total_trades}</td>
                <td className={`text-right py-2 ${r.metrics.profit_factor >= 1 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {r.metrics.profit_factor === Infinity ? '∞' : r.metrics.profit_factor.toFixed(2)}
                </td>
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
