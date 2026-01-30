import { useState, useEffect, useCallback, useRef } from 'react';
import type { DashboardData, EquityPoint, Order, WalkForwardResult, PredictionRecord, AccuracySummary, ContinuousBacktestResult, DegradationAlert, LiveBacktestProgress, LiveBacktestTrade, LiveBacktestComplete } from './types';

const BASE = '/api';

async function fetchJSON<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export function useDashboard(intervalMs = 5000) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<DashboardData>('/dashboard');
    if (d) setData(d);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return { data, loading };
}

export function useEquityCurve(intervalMs = 30000) {
  const [data, setData] = useState<EquityPoint[]>([]);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<EquityPoint[]>('/equity-curve');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function useOrders(intervalMs = 10000) {
  const [data, setData] = useState<Order[]>([]);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<Order[]>('/orders');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function useWalkForward(intervalMs = 30000) {
  const [data, setData] = useState<WalkForwardResult | null>(null);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<WalkForwardResult>('/backtest/walk-forward');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

/**
 * WebSocket hook for real-time dashboard updates.
 * Falls back to polling if WebSocket is unavailable.
 */
export function usePredictions(intervalMs = 10000) {
  const [data, setData] = useState<Record<string, PredictionRecord[]>>({});

  const refresh = useCallback(async () => {
    const d = await fetchJSON<Record<string, PredictionRecord[]>>('/predictions/live');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function usePredictionAccuracy(intervalMs = 30000) {
  const [data, setData] = useState<AccuracySummary | null>(null);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<AccuracySummary>('/predictions/accuracy');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function useContinuousBacktest(intervalMs = 60000) {
  const [data, setData] = useState<ContinuousBacktestResult[]>([]);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<ContinuousBacktestResult[]>('/backtest/continuous');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function useDegradationAlerts(intervalMs = 60000) {
  const [data, setData] = useState<DegradationAlert[]>([]);

  const refresh = useCallback(async () => {
    const d = await fetchJSON<DegradationAlert[]>('/backtest/degradation');
    if (d) setData(d);
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return data;
}

export function useStreamingBacktest() {
  const [progress, setProgress] = useState<Record<string, LiveBacktestProgress>>({});
  const [trades, setTrades] = useState<LiveBacktestTrade[]>([]);
  const [completed, setCompleted] = useState<LiveBacktestComplete[]>([]);
  const [equityCurves, setEquityCurves] = useState<Record<string, number[]>>({});
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws`;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      ws = new WebSocket(url);

      ws.onopen = () => setConnected(true);

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'backtest_progress') {
            const d = msg.data as LiveBacktestProgress;
            const key = `${d.strategy}:${d.symbol}`;
            setProgress(prev => ({ ...prev, [key]: d }));
            if (d.equity_curve) {
              setEquityCurves(prev => ({ ...prev, [key]: d.equity_curve }));
            }
          }

          if (msg.type === 'backtest_trade') {
            const d = msg.data as LiveBacktestTrade;
            setTrades(prev => [d, ...prev].slice(0, 50));
          }

          if (msg.type === 'backtest_complete') {
            const d = msg.data as LiveBacktestComplete;
            setCompleted(prev => [d, ...prev].slice(0, 20));
            // Clear progress for completed
            const key = `${d.strategy}:${d.symbol}`;
            setProgress(prev => {
              const next = { ...prev };
              delete next[key];
              return next;
            });
          }

          if (msg.type === 'backtest_cycle_start') {
            setProgress({});
            setTrades([]);
          }
        } catch { /* ignore */ }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 5000);
      };

      ws.onerror = () => ws?.close();
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  return { progress, trades, completed, equityCurves, connected };
}

export function useWebSocket(onMessage?: (data: DashboardData) => void) {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws`;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      ws = new WebSocket(url);

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'dashboard' && onMessage) {
            onMessage(msg.data);
          }
        } catch { /* ignore parse errors */ }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 5000);
      };

      ws.onerror = () => {
        ws?.close();
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [onMessage]);

  return connected;
}
