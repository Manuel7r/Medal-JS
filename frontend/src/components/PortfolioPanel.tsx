import type { Portfolio, Risk } from '../types';
import StatCard from './StatCard';

interface Props {
  portfolio: Portfolio;
  risk: Risk;
}

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function pct(n: number): string {
  return `${(n * 100).toFixed(2)}%`;
}

export default function PortfolioPanel({ portfolio, risk }: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Equity"
        value={`$${fmt(portfolio.equity)}`}
        delta={`Peak: $${fmt(portfolio.peak_equity)}`}
        positive={portfolio.equity >= portfolio.peak_equity}
      />
      <StatCard
        label="Drawdown"
        value={pct(portfolio.drawdown_pct)}
        delta="Max 20%"
        positive={portfolio.drawdown_pct < 0.15}
      />
      <StatCard
        label="Daily P&L"
        value={`$${fmt(portfolio.daily_pnl)}`}
        delta={pct(portfolio.daily_pnl_pct)}
        positive={portfolio.daily_pnl >= 0}
      />
      <StatCard
        label="Leverage"
        value={`${fmt(risk.leverage, 1)}x`}
        delta="Max 3.0x"
        positive={risk.leverage <= 3.0}
      />
    </div>
  );
}
