import { useState } from 'react';
import type { Order } from '../types';

interface Props {
  orders: Order[];
}

const statusColors: Record<string, string> = {
  FILLED: 'bg-emerald-500/20 text-emerald-400',
  REJECTED: 'bg-red-500/20 text-red-400',
  CANCELLED: 'bg-slate-500/20 text-slate-400',
  PENDING: 'bg-yellow-500/20 text-yellow-400',
  SUBMITTED: 'bg-blue-500/20 text-blue-400',
  PARTIALLY_FILLED: 'bg-cyan-500/20 text-cyan-400',
};

export default function OrdersTable({ orders }: Props) {
  const [showOpen, setShowOpen] = useState(false);

  const openStatuses = new Set(['PENDING', 'SUBMITTED', 'PARTIALLY_FILLED']);
  const filtered = showOpen ? orders.filter(o => openStatuses.has(o.status)) : orders;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Orders</h2>
        <div className="flex gap-2 text-sm">
          <button
            onClick={() => setShowOpen(false)}
            className={`px-3 py-1 rounded-lg ${!showOpen ? 'bg-slate-600 text-white' : 'text-slate-400'}`}
          >
            All
          </button>
          <button
            onClick={() => setShowOpen(true)}
            className={`px-3 py-1 rounded-lg ${showOpen ? 'bg-slate-600 text-white' : 'text-slate-400'}`}
          >
            Open
          </button>
        </div>
      </div>
      {filtered.length === 0 ? (
        <p className="text-slate-500 text-sm">No orders</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-2">ID</th>
                <th className="text-left py-2">Symbol</th>
                <th className="text-left py-2">Side</th>
                <th className="text-right py-2">Qty</th>
                <th className="text-left py-2">Status</th>
                <th className="text-right py-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(o => (
                <tr key={o.order_id} className="border-b border-slate-700/50">
                  <td className="py-2 font-mono text-xs">{o.order_id}</td>
                  <td className="py-2">{o.symbol}</td>
                  <td className={`py-2 ${o.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {o.side}
                  </td>
                  <td className="text-right py-2">{o.quantity.toFixed(4)}</td>
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColors[o.status] || ''}`}>
                      {o.status}
                    </span>
                  </td>
                  <td className="text-right py-2 text-slate-400 text-xs">
                    {new Date(o.created_at).toLocaleTimeString()}
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
