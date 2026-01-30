import type { Risk } from '../types';

interface Props {
  risk?: Risk;
  children: React.ReactNode;
  activeSection: string;
  onNavigate: (section: string) => void;
}

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'predictions', label: 'Predictions' },
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'backtest', label: 'Backtests' },
  { id: 'orders', label: 'Orders' },
  { id: 'risk', label: 'Risk' },
  { id: 'alerts', label: 'Alerts' },
  { id: 'scheduler', label: 'Scheduler' },
];

export default function Layout({ risk, children, activeSection, onNavigate }: Props) {
  const statusColor = risk?.suspended
    ? 'bg-red-500'
    : risk?.paused
      ? 'bg-yellow-500'
      : 'bg-emerald-500';

  const statusText = risk?.suspended
    ? 'SUSPENDED'
    : risk?.paused
      ? 'PAUSED'
      : 'RUNNING';

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-60 bg-slate-900 border-r border-slate-700 p-6 flex flex-col fixed h-full">
        <h1 className="text-xl font-bold text-white mb-1">Medal Trading</h1>
        <p className="text-xs text-slate-500 mb-8">Quantitative System</p>

        <nav className="space-y-1 text-sm">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`block w-full text-left px-3 py-2 rounded-lg transition-colors ${
                activeSection === item.id
                  ? 'bg-slate-800 text-white font-medium'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto">
          <div className="flex items-center gap-2 text-sm">
            <span className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
            <span className="text-slate-300">{statusText}</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 ml-60 p-6 overflow-auto space-y-6">
        {children}
      </main>
    </div>
  );
}
