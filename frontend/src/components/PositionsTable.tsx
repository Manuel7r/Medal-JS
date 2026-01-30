import type { Position } from '../types';

interface Props {
  positions: Position[];
}

export default function PositionsTable({ positions }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Positions</h2>
      {positions.length === 0 ? (
        <p className="text-slate-500 text-sm">No open positions</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-2">Symbol</th>
                <th className="text-right py-2">Qty</th>
                <th className="text-right py-2">Entry</th>
                <th className="text-right py-2">Current</th>
                <th className="text-right py-2">P&L</th>
                <th className="text-right py-2">P&L %</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(p => (
                <tr key={p.symbol} className="border-b border-slate-700/50">
                  <td className="py-2 font-medium">{p.symbol}</td>
                  <td className="text-right py-2">{p.quantity.toFixed(4)}</td>
                  <td className="text-right py-2">${p.entry_price.toFixed(2)}</td>
                  <td className="text-right py-2">${p.current_price.toFixed(2)}</td>
                  <td className={`text-right py-2 ${p.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    ${p.unrealized_pnl.toFixed(2)}
                  </td>
                  <td className={`text-right py-2 ${p.pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {(p.pnl_pct * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
