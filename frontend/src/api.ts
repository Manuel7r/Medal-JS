import { useState, useEffect, useCallback } from 'react';
import type { DashboardData, EquityPoint, Order, WalkForwardResult } from './types';

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
