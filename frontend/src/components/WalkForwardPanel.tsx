import type { WalkForwardResult } from '../types';

function pct(n: number): string {
  return `${(n * 100).toFixed(2)}%`;
}

interface Props {
  data: WalkForwardResult | null;
}

export default function WalkForwardPanel({ data }: Props) {
  if (!data) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h2 className="text-lg font-semibold mb-4">Walk-Forward Validation</h2>
        <p className="text-slate-500 text-sm">Running walk-forward analysis... results will appear shortly</p>
      </div>
    );
  }

  const oos = data.oos_metrics;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-1">Walk-Forward Validation</h2>
      <p className="text-xs text-slate-500 mb-4">
        {data.strategy} on {data.symbol} — {data.n_windows} out-of-sample windows
      </p>

      {/* OOS aggregate metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
        <div>
          <p className="text-xs text-slate-500">OOS Sharpe</p>
          <p className={`text-lg font-mono font-bold ${oos.sharpe_ratio >= 1 ? 'text-emerald-400' : oos.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
            {oos.sharpe_ratio.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">OOS Return</p>
          <p className={`text-lg font-mono font-bold ${oos.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {pct(oos.total_return)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">OOS Max DD</p>
          <p className="text-lg font-mono font-bold text-red-400">{pct(oos.max_drawdown)}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">OOS Trades</p>
          <p className="text-lg font-mono font-bold text-slate-300">{oos.total_trades}</p>
        </div>
      </div>

      {/* Per-window table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left py-2">Window</th>
              <th className="text-right py-2">Train Sharpe</th>
              <th className="text-right py-2">Test Sharpe</th>
              <th className="text-right py-2">Test Return</th>
              <th className="text-right py-2">Test Trades</th>
            </tr>
          </thead>
          <tbody>
            {data.windows.map((w) => (
              <tr key={w.window_id} className="border-b border-slate-700/50">
                <td className="py-2 font-medium">#{w.window_id}</td>
                <td className="text-right py-2 font-mono text-slate-400">
                  {w.train_sharpe != null ? w.train_sharpe.toFixed(2) : '—'}
                </td>
                <td className={`text-right py-2 font-mono ${(w.test_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {w.test_sharpe != null ? w.test_sharpe.toFixed(2) : '—'}
                </td>
                <td className={`text-right py-2 ${(w.test_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {w.test_return != null ? pct(w.test_return) : '—'}
                </td>
                <td className="text-right py-2 text-slate-300">
                  {w.test_trades ?? '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
