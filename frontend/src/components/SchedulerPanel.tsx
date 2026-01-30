import type { SchedulerData } from '../types';

interface Props {
  scheduler: SchedulerData;
}

function timeAgo(iso: string | null): string {
  if (!iso) return 'Never';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m ago`;
}

export default function SchedulerPanel({ scheduler }: Props) {
  const jobs = Object.values(scheduler.jobs);

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Scheduler</h2>
        <div className="flex items-center gap-2 text-sm">
          <span className={`w-2 h-2 rounded-full ${scheduler.running ? 'bg-emerald-500' : 'bg-red-500'}`} />
          <span className="text-slate-400">{scheduler.running ? 'Running' : 'Stopped'}</span>
        </div>
      </div>
      {jobs.length === 0 ? (
        <p className="text-slate-500 text-sm">No jobs registered</p>
      ) : (
        <div className="space-y-3">
          {jobs.map(j => (
            <div key={j.name} className="flex items-center justify-between py-2 border-b border-slate-700/50 last:border-0">
              <div>
                <p className="text-sm font-medium">{j.name}</p>
                <p className="text-xs text-slate-500">{timeAgo(j.last_run)}</p>
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span>{j.run_count} runs</span>
                {j.error_count > 0 && (
                  <span className="text-red-400">{j.error_count} errors</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
