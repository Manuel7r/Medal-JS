interface Props {
  label: string;
  value: string;
  delta?: string;
  positive?: boolean;
}

export default function StatCard({ label, value, delta, positive }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      {delta && (
        <p className={`text-sm mt-1 ${positive ? 'text-emerald-400' : 'text-red-400'}`}>
          {positive ? '▲' : '▼'} {delta}
        </p>
      )}
    </div>
  );
}
