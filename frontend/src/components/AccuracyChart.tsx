import { usePredictionAccuracy } from '../api';

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function hitRateColor(rate: number): string {
  if (rate >= 0.6) return 'text-emerald-400';
  if (rate >= 0.5) return 'text-yellow-400';
  return 'text-red-400';
}

export default function AccuracyChart() {
  const data = usePredictionAccuracy();

  if (!data || data.total_resolved === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h2 className="text-lg font-semibold mb-2">Prediction Accuracy</h2>
        <p className="text-sm text-slate-500">No resolved predictions yet. Accuracy will appear after predictions are validated against actual prices.</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Prediction Accuracy</h2>

      {/* Overall metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div>
          <div className="text-xs text-slate-500 mb-1">Overall Hit Rate</div>
          <div className={`text-xl font-mono font-bold ${hitRateColor(data.overall.hit_rate)}`}>
            {pct(data.overall.hit_rate)}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Total Predictions</div>
          <div className="text-xl font-mono text-slate-200">{data.total_resolved}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Pending</div>
          <div className="text-xl font-mono text-slate-200">{data.total_pending}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 mb-1">Calibration Error</div>
          <div className="text-xl font-mono text-slate-200">{pct(data.overall.calibration_error)}</div>
        </div>
      </div>

      {/* Per-strategy table */}
      <h3 className="text-sm text-slate-400 mb-2 font-medium">Per Strategy</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-500 text-xs border-b border-slate-700">
              <th className="text-left py-2 pr-4">Strategy</th>
              <th className="text-right py-2 px-2">Hit Rate</th>
              <th className="text-right py-2 px-2">Recent (20)</th>
              <th className="text-right py-2 px-2">Total</th>
              <th className="text-right py-2 px-2">Conf (Correct)</th>
              <th className="text-right py-2 px-2">Conf (Wrong)</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.per_strategy).map(([name, acc]) => (
              <tr key={name} className="border-b border-slate-700/30">
                <td className="py-2 pr-4 text-slate-300 font-medium">{name}</td>
                <td className={`py-2 px-2 text-right font-mono ${hitRateColor(acc.hit_rate)}`}>
                  {pct(acc.hit_rate)}
                </td>
                <td className={`py-2 px-2 text-right font-mono ${hitRateColor(acc.recent_hit_rate)}`}>
                  {pct(acc.recent_hit_rate)}
                </td>
                <td className="py-2 px-2 text-right font-mono text-slate-400">
                  {acc.total_predictions}
                </td>
                <td className="py-2 px-2 text-right font-mono text-slate-400">
                  {pct(acc.avg_confidence_when_correct)}
                </td>
                <td className="py-2 px-2 text-right font-mono text-slate-400">
                  {pct(acc.avg_confidence_when_wrong)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Ranking */}
      {data.ranking.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm text-slate-400 mb-2 font-medium">Strategy Ranking</h3>
          <div className="flex gap-2">
            {data.ranking.map((r, i) => (
              <div key={r.strategy} className={`px-3 py-1.5 rounded text-xs font-medium ${
                i === 0 ? 'bg-emerald-500/20 text-emerald-400' :
                i === 1 ? 'bg-blue-500/20 text-blue-400' :
                'bg-slate-500/20 text-slate-400'
              }`}>
                #{i + 1} {r.strategy} ({pct(r.hit_rate)})
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
