import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AgentsPage } from './AgentsPage';

// ---------------------------------------------------------------------------
// Mock API
// ---------------------------------------------------------------------------

const mockAgents = [
  {
    id: 'a1',
    name: 'Research Agent',
    agent_type: 'research',
    config: {},
    status: 'idle' as const,
    summary_memory: 'Researches topics',
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
    total_runs: 5,
    total_cost: 0.05,
  },
  {
    id: 'a2',
    name: 'Code Agent',
    agent_type: 'code',
    config: {},
    status: 'running' as const,
    summary_memory: 'Writes code',
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  },
];

vi.mock('../lib/api', () => ({
  fetchManagedAgents: vi.fn().mockResolvedValue([]),
  fetchAgentTasks: vi.fn().mockResolvedValue([]),
  fetchAgentChannels: vi.fn().mockResolvedValue([]),
  fetchAgentMessages: vi.fn().mockResolvedValue([]),
  fetchTemplates: vi.fn().mockResolvedValue([]),
  createManagedAgent: vi.fn(),
  pauseManagedAgent: vi.fn(),
  resumeManagedAgent: vi.fn(),
  deleteManagedAgent: vi.fn(),
  runManagedAgent: vi.fn(),
  recoverManagedAgent: vi.fn(),
  sendAgentMessage: vi.fn(),
  fetchLearningLog: vi.fn().mockResolvedValue([]),
  triggerLearning: vi.fn(),
  fetchAgentTraces: vi.fn().mockResolvedValue([]),
}));

import { fetchManagedAgents } from '../lib/api';

const mockFetchManagedAgents = vi.mocked(fetchManagedAgents);

// ---------------------------------------------------------------------------
// Mock Zustand store using a shared object referenced from the factory
// ---------------------------------------------------------------------------

const _state = {
  managedAgents: [] as any[],
  managedAgentsLoading: false,
  selectedAgentId: null as string | null,
};

vi.mock('../lib/store', () => {
  const getState = () => ({
    ..._state,
    setManagedAgents: (agents: any[]) => { _state.managedAgents = agents; },
    setManagedAgentsLoading: () => {},
    setSelectedAgentId: () => {},
  });

  const store = (selector: (s: any) => any) => selector(getState());

  return {
    useAppStore: Object.assign(store, {
      getState,
      setState: (partial: any) => Object.assign(_state, typeof partial === 'function' ? partial(_state) : partial),
      subscribe: () => () => {},
    }),
  };
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchManagedAgents.mockResolvedValue([]);
    _state.managedAgents = [];
    _state.selectedAgentId = null;
  });

  it('shows loading state initially', () => {
    render(<AgentsPage />);
    expect(screen.getByText('Loading agents...')).toBeInTheDocument();
  });

  it('renders the Agents heading after loading', async () => {
    render(<AgentsPage />);
    await waitFor(() => {
      expect(screen.getByText('Agents')).toBeInTheDocument();
    });
  });

  it('calls fetchManagedAgents on mount', async () => {
    render(<AgentsPage />);
    await waitFor(() => {
      expect(mockFetchManagedAgents).toHaveBeenCalled();
    });
  });

  it('shows "Create Agent" button after loading', async () => {
    render(<AgentsPage />);
    await waitFor(() => {
      expect(screen.getByText('New Agent')).toBeInTheDocument();
    });
  });

  it('displays agent cards when agents are fetched', async () => {
    mockFetchManagedAgents.mockResolvedValue(mockAgents);

    render(<AgentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Research Agent')).toBeInTheDocument();
      expect(screen.getByText('Code Agent')).toBeInTheDocument();
    });
  });
});
