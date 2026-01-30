import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { EquityPoint } from '../types';

interface Props {
  data: EquityPoint[];
}

export default function EquityChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 text-center text-slate-500">
        No equity data yet
      </div>
    );
  }

  const formatted = data.map(d => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }));

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={formatted}>
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="time" tick={{ fill: '#94a3b8', fontSize: 12 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Area type="monotone" dataKey="equity" stroke="#10b981" fill="url(#eqGrad)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
