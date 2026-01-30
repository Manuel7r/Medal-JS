import { useState, useEffect, useCallback } from 'react';

interface VaRData {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
}

interface HurstData {
  hurst: number;
  regime: string;
  confidence: string;
}

interface RiskReport {
  var_historical: VaRData | null;
  var_parametric: VaRData | null;
  hurst: Record<string, HurstData> | null;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(2)}%`;
}

function regimeColor(regime: string): string {
  switch (regime) {
    case 'mean_reverting': return 'text-cyan-400';
    case 'trending': return 'text-amber-400';
    default: return 'text-slate-400';
  }
}

function regimeLabel(regime: string): string {
  switch (regime) {
    case 'mean_reverting': return 'Mean Reverting';
    case 'trending': return 'Trending';
    default: return 'Random Walk';
  }
}

export default function AdvancedRiskPanel() {
  const [data, setData] = useState<RiskReport | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/risk/advanced');
      if (res.ok) setData(await res.json());
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, [refresh]);

  if (!data) return null;

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <h2 className="text-lg font-semibold mb-4">Advanced Risk Metrics</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* VaR */}
        {data.var_historical && (
          <div>
            <h3 className="text-sm text-slate-400 mb-2 font-medium">Value at Risk (Historical)</h3>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">VaR 95%</span>
                <span className="text-red-400 font-mono">{pct(data.var_historical.var_95)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">VaR 99%</span>
                <span className="text-red-400 font-mono">{pct(data.var_historical.var_99)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">CVaR 95%</span>
                <span className="text-red-500 font-mono">{pct(data.var_historical.cvar_95)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">CVaR 99%</span>
                <span className="text-red-500 font-mono">{pct(data.var_historical.cvar_99)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Hurst */}
        {data.hurst && Object.keys(data.hurst).length > 0 && (
          <div>
            <h3 className="text-sm text-slate-400 mb-2 font-medium">Hurst Exponent (Regime)</h3>
            <div className="space-y-2">
              {Object.entries(data.hurst).map(([symbol, h]) => (
                <div key={symbol} className="flex items-center justify-between text-sm">
                  <span className="text-slate-300 font-medium">{symbol}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-slate-200">{h.hurst.toFixed(3)}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${regimeColor(h.regime)}`}>
                      {regimeLabel(h.regime)}
                    </span>
                    <span className="text-xs text-slate-500">({h.confidence})</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
