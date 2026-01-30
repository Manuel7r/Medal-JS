import { useState, useEffect, useCallback } from 'react';

interface RegimeStat {
  window_id: number;
  volatility: number;
  regime: string;
  sharpe: number;
  n_trades: number;
}

interface MonteCarlo {
  original_sharpe: number;
  p_value: number;
  percentile_5: number;
  percentile_95: number;
  n_simulations: number;
}

interface DiagnosticsData {
  avg_is_sharpe: number;
  avg_oos_sharpe: number;
  sharpe_degradation: number;
  sharpe_stability: number;
  worst_window_sharpe: number;
  best_window_sharpe: number;
  pct_profitable_windows: number;
  regime_stats: RegimeStat[];
  monte_carlo: MonteCarlo | null;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function regimeBadge(regime: string): string {
  switch (regime) {
    case 'high': return 'bg-red-500/20 text-red-400';
    case 'low': return 'bg-blue-500/20 text-blue-400';
    default: return 'bg-slate-500/20 text-slate-400';
  }
}

export default function DiagnosticsPanel() {
  const [data, setData] = useState<DiagnosticsData | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/backtest/diagnostics');
      if (res.ok) {
        const d = await res.json();
        if (d) setData(d);
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, [refresh]);

  if (!data) return null;

  const degradationColor = data.sharpe_degradation < 0.3 ? 'text-emerald-400' :
    data.sharpe_degradation < 0.5 ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Walk-Forward Diagnostics</h2>

      {/* Key metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <div className="text-xs text-slate-500 mb-1">Avg IS Sharpe</div>
          <div className="text-lg font-mono text-slate-200">{data.avg_is_sharpe.toFixed(3)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Avg OOS Sharpe</div>
          <div className="text-lg font-mono text-slate-200">{data.avg_oos_sharpe.toFixed(3)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Degradation</div>
          <div className={`text-lg font-mono ${degradationColor}`}>{pct(data.sharpe_degradation)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Profitable Windows</div>
          <div className="text-lg font-mono text-slate-200">{pct(data.pct_profitable_windows)}</div>
        </div>
      </div>

      {/* Regime analysis */}
      {data.regime_stats.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm text-slate-400 mb-2 font-medium">Regime Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {data.regime_stats.map(rs => (
              <div key={rs.window_id} className="flex items-center justify-between bg-slate-700/30 rounded px-3 py-1.5 text-sm">
                <span className="text-slate-400">W{rs.window_id}</span>
                <span className={`px-2 py-0.5 rounded text-xs ${regimeBadge(rs.regime)}`}>
                  {rs.regime}
                </span>
                <span className="font-mono text-slate-300">{rs.sharpe.toFixed(2)}</span>
                <span className="text-slate-500 text-xs">{rs.n_trades}t</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Monte Carlo */}
      {data.monte_carlo && (
        <div>
          <h3 className="text-sm text-slate-400 mb-2 font-medium">Monte Carlo ({data.monte_carlo.n_simulations} sims)</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-xs text-slate-500">Original Sharpe</div>
              <div className="font-mono">{data.monte_carlo.original_sharpe.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-500">p-value</div>
              <div className={`font-mono ${data.monte_carlo.p_value < 0.05 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                {data.monte_carlo.p_value.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">95% CI</div>
              <div className="font-mono text-slate-300">
                [{data.monte_carlo.percentile_5.toFixed(3)}, {data.monte_carlo.percentile_95.toFixed(3)}]
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
