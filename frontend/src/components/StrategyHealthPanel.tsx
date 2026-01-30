import { useContinuousBacktest, useDegradationAlerts } from '../api';

function healthColor(degraded: boolean): string {
  return degraded ? 'border-red-500/50 bg-red-500/5' : 'border-emerald-500/30 bg-emerald-500/5';
}

function healthBadge(degraded: boolean): string {
  return degraded
    ? 'bg-red-500/20 text-red-400'
    : 'bg-emerald-500/20 text-emerald-400';
}

function metricLabel(metric: string): string {
  switch (metric) {
    case 'sharpe_absolute': return 'Sharpe below threshold';
    case 'sharpe_drop': return 'Sharpe dropped significantly';
    case 'win_rate_drop': return 'Win rate declined';
    default: return metric;
  }
}

export default function StrategyHealthPanel() {
  const results = useContinuousBacktest();
  const alerts = useDegradationAlerts();

  if (results.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h2 className="text-lg font-semibold mb-2">Strategy Health</h2>
        <p className="text-sm text-slate-500">Waiting for continuous backtest cycle (every 4h)...</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Strategy Health</h2>

      {/* Strategy cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
        {results.map(r => (
          <div key={`${r.strategy}-${r.symbol}`} className={`rounded-lg border p-4 ${healthColor(r.degraded)}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-slate-200">{r.strategy}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${healthBadge(r.degraded)}`}>
                {r.degraded ? 'Degraded' : 'Healthy'}
              </span>
            </div>
            <div className="text-xs text-slate-400 mb-2">{r.symbol}</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <div className="text-xs text-slate-500">Sharpe</div>
                <div className={`font-mono ${r.sharpe_ratio > 0.5 ? 'text-emerald-400' : r.sharpe_ratio > 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {r.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Win Rate</div>
                <div className="font-mono text-slate-300">{(r.win_rate * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Return</div>
                <div className={`font-mono ${r.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(r.total_return * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Trades</div>
                <div className="font-mono text-slate-300">{r.total_trades}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Degradation alerts */}
      {alerts.length > 0 && (
        <div>
          <h3 className="text-sm text-red-400 mb-2 font-medium">Degradation Alerts</h3>
          <div className="space-y-1.5">
            {alerts.slice(0, 10).map((a, i) => (
              <div key={i} className="flex items-center gap-3 text-sm py-1.5 border-b border-slate-700/30">
                <span className="text-xs text-slate-500 whitespace-nowrap">
                  {new Date(a.timestamp).toLocaleTimeString()}
                </span>
                <span className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap">
                  {a.strategy}
                </span>
                <span className="text-slate-400">{a.symbol}</span>
                <span className="text-slate-500 text-xs">{metricLabel(a.metric)}</span>
                <span className="text-slate-400 font-mono text-xs">
                  {a.historical_value.toFixed(2)} {'\u2192'} {a.current_value.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
