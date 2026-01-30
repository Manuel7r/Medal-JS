import { useState, useEffect, useCallback } from 'react';

interface AuditEntry {
  timestamp: string;
  event_type: string;
  source: string;
  details: Record<string, unknown>;
}

function eventColor(type: string): string {
  if (type.includes('RISK_BREACH')) return 'bg-red-500/20 text-red-400';
  if (type.includes('ORDER')) return 'bg-blue-500/20 text-blue-400';
  if (type.includes('SIGNAL')) return 'bg-purple-500/20 text-purple-400';
  if (type.includes('SYSTEM')) return 'bg-amber-500/20 text-amber-400';
  return 'bg-slate-500/20 text-slate-400';
}

export default function AuditPanel() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/audit?limit=20');
      if (res.ok) setEntries(await res.json());
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, [refresh]);

  if (entries.length === 0) return null;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Audit Trail</h2>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {entries.map((e, i) => (
          <div key={i} className="flex items-start gap-3 text-sm py-1.5 border-b border-slate-700/30">
            <span className="text-xs text-slate-500 whitespace-nowrap mt-0.5">
              {new Date(e.timestamp).toLocaleTimeString()}
            </span>
            <span className={`px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap ${eventColor(e.event_type)}`}>
              {e.event_type}
            </span>
            <span className="text-slate-400 whitespace-nowrap">{e.source}</span>
            <span className="text-slate-500 truncate">
              {JSON.stringify(e.details).slice(0, 80)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
