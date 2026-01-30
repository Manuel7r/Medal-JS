import type { Risk } from '../types';

interface Props {
  risk: Risk;
}

function ProgressBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span>{(value * 100).toFixed(1)}% / {(max * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RiskPanel({ risk }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Risk</h2>
      <ProgressBar
        label="Drawdown"
        value={risk.drawdown_pct}
        max={0.20}
        color={risk.drawdown_pct > 0.15 ? 'bg-red-500' : 'bg-emerald-500'}
      />
      <ProgressBar
        label="Leverage"
        value={risk.leverage / 3.0}
        max={1.0}
        color={risk.leverage > 2.5 ? 'bg-yellow-500' : 'bg-blue-500'}
      />
      <div className="flex gap-3 mt-4">
        {risk.suspended && (
          <span className="px-3 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400">
            SUSPENDED
          </span>
        )}
        {risk.paused && (
          <span className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400">
            PAUSED
          </span>
        )}
        {!risk.suspended && !risk.paused && (
          <span className="px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400">
            NORMAL
          </span>
        )}
      </div>
    </div>
  );
}
