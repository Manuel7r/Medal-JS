import { useState, useEffect, useCallback } from 'react';
import type { PredictionRecord } from '../types';
import { usePredictions } from '../api';

function directionBadge(dir: string, correct: boolean | null) {
  const base = dir === 'UP'
    ? 'bg-emerald-500/20 text-emerald-400'
    : 'bg-red-500/20 text-red-400';
  return base;
}

function confidenceBar(confidence: number) {
  const pct = Math.round(confidence * 100);
  const color = confidence > 0.7 ? 'bg-emerald-500' : confidence > 0.5 ? 'bg-yellow-500' : 'bg-slate-500';
  return (
    <div className="flex items-center gap-2 min-w-[100px]">
      <div className="flex-1 bg-slate-700 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

function formatCountdown(iso: string | null): string {
  if (!iso) return '';
  const diff = new Date(iso).getTime() - Date.now();
  if (diff <= 0) return 'Running now...';
  const secs = Math.floor(diff / 1000);
  const mins = Math.floor(secs / 60);
  const remSecs = secs % 60;
  return `${mins}m ${remSecs.toString().padStart(2, '0')}s`;
}

function useNextPredictionRun() {
  const [nextRun, setNextRun] = useState<string | null>(null);
  const [, setTick] = useState(0);

  const poll = useCallback(async () => {
    try {
      const res = await fetch('/api/dashboard');
      if (!res.ok) return;
      const data = await res.json();
      const job = data?.scheduler?.jobs?.prediction_cycle;
      setNextRun(job?.next_run ?? null);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [poll]);

  // Tick every second for countdown
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return nextRun;
}

export default function PredictionPanel() {
  const predictions = usePredictions();
  const nextRun = useNextPredictionRun();
  const symbols = Object.keys(predictions);

  if (symbols.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold">Live Predictions</h2>
          {nextRun && (
            <span className="text-emerald-400 font-mono text-xs bg-emerald-500/10 px-2 py-1 rounded">
              Next cycle: {formatCountdown(nextRun)}
            </span>
          )}
        </div>
        <p className="text-sm text-slate-500">Waiting for first prediction cycle (every 15 min)...</p>
        <div className="flex items-center gap-2 mt-3">
          <div className="w-3 h-3 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-xs text-slate-400">Prediction engine initializing</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Live Predictions</h2>
        {nextRun && (
          <span className="text-emerald-400 font-mono text-xs bg-emerald-500/10 px-2 py-1 rounded">
            Next cycle: {formatCountdown(nextRun)}
          </span>
        )}
      </div>
      <div className="space-y-4">
        {symbols.map(symbol => (
          <div key={symbol}>
            <h3 className="text-sm font-medium text-slate-300 mb-2">{symbol}</h3>
            <div className="space-y-1">
              {predictions[symbol].map((p: PredictionRecord) => (
                <div key={p.prediction_id} className="flex items-center gap-3 text-sm py-1.5 border-b border-slate-700/30">
                  <span className="text-slate-400 w-28 truncate">{p.strategy}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${directionBadge(p.direction, p.correct)}`}>
                    {p.direction === 'UP' ? '\u2191' : '\u2193'} {p.direction}
                  </span>
                  {confidenceBar(p.confidence)}
                  <span className="text-slate-500 text-xs">
                    ${p.price_at_prediction.toFixed(2)}
                  </span>
                  {p.correct === null ? (
                    <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" title="Pending" />
                  ) : p.correct ? (
                    <span className="text-emerald-400 text-xs font-bold" title="Correct">OK</span>
                  ) : (
                    <span className="text-red-400 text-xs font-bold" title="Wrong">X</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
