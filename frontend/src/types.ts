export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  pnl_pct: number;
}

export interface Portfolio {
  equity: number;
  peak_equity: number;
  drawdown_pct: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_commission: number;
}

export interface PositionsData {
  count: number;
  leverage: number;
  positions: Position[];
}

export interface OrdersSummary {
  total_orders: number;
  total_fills: number;
  summary: Record<string, number>;
}

export interface Risk {
  drawdown_pct: number;
  leverage: number;
  suspended: boolean;
  paused: boolean;
}

export interface Alert {
  type: string;
  level: string;
  message: string;
  time: string;
  acknowledged: boolean;
}

export interface AlertsData {
  unacknowledged: number;
  recent: Alert[];
}

export interface Job {
  name: string;
  last_run: string | null;
  run_count: number;
  error_count: number;
  last_error: string | null;
}

export interface SchedulerData {
  running: boolean;
  jobs: Record<string, Job>;
}

export interface DashboardData {
  portfolio: Portfolio;
  positions: PositionsData;
  orders: OrdersSummary;
  risk: Risk;
  alerts: AlertsData;
  scheduler: SchedulerData;
  last_update: string;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown_pct: number;
  daily_pnl: number;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  price: number | null;
  status: string;
  filled_quantity: number;
  filled_price: number;
  commission: number;
  created_at: string;
  updated_at: string;
}

export interface WalkForwardWindow {
  window_id: number;
  train_sharpe: number | null;
  test_sharpe: number | null;
  test_return: number | null;
  test_trades: number | null;
}

export interface WalkForwardResult {
  symbol: string;
  strategy: string;
  n_windows: number;
  oos_metrics: BacktestMetrics;
  windows: WalkForwardWindow[];
}

export interface BacktestMetrics {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  max_drawdown_duration: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  avg_trade_return: number;
  avg_win: number;
  avg_loss: number;
  expectancy: number;
}
