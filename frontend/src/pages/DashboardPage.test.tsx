import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DashboardPage } from './DashboardPage';

// ---------------------------------------------------------------------------
// Mock child components
// ---------------------------------------------------------------------------

vi.mock('../components/Dashboard/EnergyDashboard', () => ({
  EnergyDashboard: () => <div data-testid="energy-dashboard">EnergyDashboard</div>,
}));

vi.mock('../components/Dashboard/CostComparison', () => ({
  CostComparison: () => <div data-testid="cost-comparison">CostComparison</div>,
}));

vi.mock('../components/Dashboard/TraceDebugger', () => ({
  TraceDebugger: () => <div data-testid="trace-debugger">TraceDebugger</div>,
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('DashboardPage', () => {
  it('renders the Dashboard heading', () => {
    render(<DashboardPage />);
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('renders EnergyDashboard component', () => {
    render(<DashboardPage />);
    expect(screen.getByTestId('energy-dashboard')).toBeInTheDocument();
  });

  it('renders CostComparison component', () => {
    render(<DashboardPage />);
    expect(screen.getByTestId('cost-comparison')).toBeInTheDocument();
  });

  it('renders TraceDebugger component', () => {
    render(<DashboardPage />);
    expect(screen.getByTestId('trace-debugger')).toBeInTheDocument();
  });
});
