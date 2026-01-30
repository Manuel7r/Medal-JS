import { useStreamingBacktest } from '../api';
import LiveEquityCurve from './LiveEquityCurve';

export default function LiveBacktestPanel() {
  const { progress, trades, completed, equityCurves, connected } = useStreamingBacktest();

  const activeKeys = Object.keys(progress);
  const hasData = activeKeys.length > 0 || completed.length > 0;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Live Backtest Stream</h2>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-500'}`} />
          <span className="text-slate-400">{connected ? 'Streaming' : 'Disconnected'}</span>
        </div>
      </div>

      {!hasData && (
        <div className="text-sm text-slate-500">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
            <span>Waiting for streaming backtest cycle to start...</span>
          </div>
        </div>
      )}

      {/* Active backtests */}
      {activeKeys.map(key => {
        const p = progress[key];
        const curve = equityCurves[key] || [];
        return (
          <div key={key} className="mb-6 border border-slate-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-sm font-medium text-emerald-400">{p.strategy}</span>
                <span className="text-slate-500 mx-2">—</span>
                <span className="text-sm text-slate-300">{p.symbol}</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">
                {p.bar_index}/{p.total_bars} bars
              </span>
            </div>

            {/* Progress bar */}
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden mb-3">
              <div
                className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                style={{ width: `${p.progress_pct}%` }}
              />
            </div>

            {/* Live equity curve */}
            {curve.length > 1 && <LiveEquityCurve data={curve} />}

            {/* Live metrics */}
            <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
              <div>
                <div className="text-xs text-slate-500">Equity</div>
                <div className={`font-mono ${p.equity >= 10000 ? 'text-emerald-400' : 'text-red-400'}`}>
                  ${p.equity.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Progress</div>
                <div className="font-mono text-slate-300">{p.progress_pct}%</div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Trades</div>
                <div className="font-mono text-slate-300">{p.total_trades}</div>
              </div>
            </div>
          </div>
        );
      })}

      {/* Recent trades */}
      {trades.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm text-slate-400 mb-2 font-medium">Recent Trades</h3>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {trades.slice(0, 10).map((t, i) => (
              <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-slate-700/30">
                <span className={`px-2 py-0.5 rounded font-medium ${
                  t.trade.side === 'LONG' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {t.trade.side}
                </span>
                <span className="text-slate-300">{t.strategy} — {t.symbol}</span>
                <span className="font-mono text-slate-400">
                  ${t.trade.entry_price} → ${t.trade.exit_price}
                </span>
                <span className={`font-mono ${t.trade.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {t.trade.pnl >= 0 ? '+' : ''}{t.trade.pnl.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Completed backtests */}
      {completed.length > 0 && (
        <div>
          <h3 className="text-sm text-slate-400 mb-2 font-medium">Completed</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {completed.map((c, i) => (
              <div key={i} className="rounded-lg border border-slate-700/50 p-3 bg-slate-800/50">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-200">{c.strategy}</span>
                  <span className="text-xs text-slate-400">{c.symbol}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <div className="text-xs text-slate-500">Sharpe</div>
                    <div className={`font-mono ${c.metrics.sharpe_ratio > 0.5 ? 'text-emerald-400' : c.metrics.sharpe_ratio > 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {c.metrics.sharpe_ratio}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Return</div>
                    <div className={`font-mono ${c.metrics.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {c.metrics.total_return}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500">Trades</div>
                    <div className="font-mono text-slate-300">{c.metrics.total_trades}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
