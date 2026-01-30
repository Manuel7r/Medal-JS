import { useState, useRef, useCallback } from 'react';
import { useDashboard, useEquityCurve, useOrders, useWalkForward, useWebSocket } from './api';
import Layout from './components/Layout';
import PortfolioPanel from './components/PortfolioPanel';
import EquityChart from './components/EquityChart';
import PositionsTable from './components/PositionsTable';
import OrdersTable from './components/OrdersTable';
import RiskPanel from './components/RiskPanel';
import AlertsList from './components/AlertsList';
import SchedulerPanel from './components/SchedulerPanel';
import BacktestPanel from './components/BacktestPanel';
import WalkForwardPanel from './components/WalkForwardPanel';
import AdvancedRiskPanel from './components/AdvancedRiskPanel';
import DiagnosticsPanel from './components/DiagnosticsPanel';
import AuditPanel from './components/AuditPanel';

export default function App() {
  const { data, loading } = useDashboard();
  const equity = useEquityCurve();
  const orders = useOrders();
  const walkForward = useWalkForward();
  const wsConnected = useWebSocket();
  const [activeSection, setActiveSection] = useState('dashboard');

  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const handleNavigate = useCallback((section: string) => {
    setActiveSection(section);
    const el = sectionRefs.current[section];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  const setRef = useCallback((id: string) => (el: HTMLDivElement | null) => {
    sectionRefs.current[id] = el;
  }, []);

  if (loading || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-slate-400">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p>Connecting to Medal Trading System...</p>
        </div>
      </div>
    );
  }

  return (
    <Layout risk={data.risk} activeSection={activeSection} onNavigate={handleNavigate}>
      {/* Dashboard overview */}
      <div ref={setRef('dashboard')}>
        <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700 mb-6">
          <h2 className="text-lg font-semibold text-white mb-2">Medal Trading System</h2>
          <p className="text-sm text-slate-400 leading-relaxed">
            Sistema de trading cuantitativo automatizado. El sistema descarga datos de Binance,
            ejecuta <strong className="text-slate-300">backtests</strong> con estrategias de Mean Reversion, Pairs Trading,
            Momentum, Microstructure y ML Ensemble,
            y opera en modo <strong className="text-slate-300">paper trading</strong> generando ordenes simuladas cada hora.
            Los datos se actualizan automaticamente.
          </p>
          <div className="flex gap-6 mt-3 text-xs text-slate-500">
            <span>Estrategias: MeanReversion, PairsTrading, Momentum, Microstructure, MLEnsemble, Aggregator</span>
            <span>Timeframe: 1h</span>
            <span>Exchange: Binance (testnet)</span>
            <span>Capital: $10,000</span>
          </div>
        </div>
      </div>

      {/* Portfolio */}
      <div ref={setRef('portfolio')}>
        <PortfolioPanel portfolio={data.portfolio} risk={data.risk} />
      </div>

      {/* Backtests */}
      <div ref={setRef('backtest')}>
        <BacktestPanel />
        <div className="mt-6">
          <WalkForwardPanel data={walkForward} />
        </div>
        <div className="mt-6">
          <DiagnosticsPanel />
        </div>
        <div className="mt-6">
          <EquityChart data={equity} />
        </div>
      </div>

      {/* Orders */}
      <div ref={setRef('orders')}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PositionsTable positions={data.positions.positions} />
          <OrdersTable orders={orders} />
        </div>
      </div>

      {/* Risk */}
      <div ref={setRef('risk')}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <RiskPanel risk={data.risk} />
          <AdvancedRiskPanel />
        </div>
      </div>

      {/* Alerts */}
      <div ref={setRef('alerts')}>
        <AlertsList alerts={data.alerts} />
      </div>

      {/* Audit */}
      <div ref={setRef('audit')}>
        <AuditPanel />
      </div>

      {/* Scheduler */}
      <div ref={setRef('scheduler')}>
        <SchedulerPanel scheduler={data.scheduler} />
      </div>

      <div className="flex items-center justify-between text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500' : 'bg-slate-500'}`} />
          {wsConnected ? 'Live' : 'Polling'}
        </span>
        <span>Last update: {new Date(data.last_update).toLocaleString()}</span>
      </div>
    </Layout>
  );
}
