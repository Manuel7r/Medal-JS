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

export interface PredictionRecord {
  prediction_id: string;
  symbol: string;
  strategy: string;
  direction: 'UP' | 'DOWN';
  confidence: number;
  price_at_prediction: number;
  timestamp: string;
  actual_direction: 'UP' | 'DOWN' | null;
  actual_price: number | null;
  resolved_at: string | null;
  correct: boolean | null;
}

export interface AccuracyMetrics {
  total_predictions: number;
  correct_predictions: number;
  hit_rate: number;
  avg_confidence_when_correct: number;
  avg_confidence_when_wrong: number;
  calibration_error: number;
  recent_hit_rate: number;
}

export interface AccuracySummary {
  overall: AccuracyMetrics;
  per_strategy: Record<string, AccuracyMetrics>;
  ranking: { strategy: string; hit_rate: number; total_predictions: number; recent_hit_rate: number }[];
  total_pending: number;
  total_resolved: number;
}

export interface ContinuousBacktestResult {
  strategy: string;
  symbol: string;
  sharpe_ratio: number;
  total_return: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  degraded: boolean;
  alerts: DegradationAlert[];
}

export interface DegradationAlert {
  strategy: string;
  symbol: string;
  metric: string;
  historical_value: number;
  current_value: number;
  threshold: number;
  timestamp: string;
}

export interface LiveBacktestProgress {
  strategy: string;
  symbol: string;
  bar_index: number;
  total_bars: number;
  progress_pct: number;
  equity: number;
  total_trades: number;
  equity_curve: number[];
}

export interface LiveBacktestTrade {
  strategy: string;
  symbol: string;
  trade: {
    side: string;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
  };
  total_trades: number;
}

export interface LiveBacktestComplete {
  strategy: string;
  symbol: string;
  metrics: {
    sharpe_ratio: number;
    total_return: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
    profit_factor: number;
  };
  final_equity: number;
  total_trades: number;
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
