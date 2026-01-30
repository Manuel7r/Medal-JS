import type { AlertsData } from '../types';

interface Props {
  alerts: AlertsData;
}

const levelColors: Record<string, string> = {
  INFO: 'bg-blue-500',
  WARNING: 'bg-yellow-500',
  CRITICAL: 'bg-red-500',
};

export default function AlertsList({ alerts }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Alerts</h2>
        {alerts.unacknowledged > 0 && (
          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/20 text-red-400">
            {alerts.unacknowledged} new
          </span>
        )}
      </div>
      {alerts.recent.length === 0 ? (
        <p className="text-slate-500 text-sm">No alerts</p>
      ) : (
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {[...alerts.recent].reverse().map((a, i) => (
            <div key={i} className="flex items-start gap-3">
              <span className={`w-2 h-2 mt-1.5 rounded-full flex-shrink-0 ${levelColors[a.level] || 'bg-slate-500'}`} />
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-slate-400">{a.type}</span>
                  <span className="text-xs text-slate-600">
                    {new Date(a.time).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-slate-300 truncate">{a.message}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
