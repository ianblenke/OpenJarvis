import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatPage } from './ChatPage';

// ---------------------------------------------------------------------------
// Mock child components to isolate page-level rendering
// ---------------------------------------------------------------------------

vi.mock('../components/Chat/ChatArea', () => ({
  ChatArea: () => <div data-testid="chat-area">ChatArea</div>,
}));

vi.mock('../components/Chat/SystemPanel', () => ({
  SystemPanel: () => <div data-testid="system-panel">SystemPanel</div>,
}));

vi.mock('../lib/store', () => {
  let systemPanelOpen = true;
  const store = (selector: (s: any) => any) =>
    selector({ systemPanelOpen });

  return {
    useAppStore: Object.assign(store, {
      getState: () => ({ systemPanelOpen }),
      setState: (partial: any) => {
        if ('systemPanelOpen' in partial) systemPanelOpen = partial.systemPanelOpen;
      },
      subscribe: () => () => {},
    }),
  };
});

import { useAppStore } from '../lib/store';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ChatPage', () => {
  beforeEach(() => {
    (useAppStore as any).setState({ systemPanelOpen: true });
  });

  it('renders ChatArea', () => {
    render(<ChatPage />);
    expect(screen.getByTestId('chat-area')).toBeInTheDocument();
  });

  it('renders SystemPanel when systemPanelOpen is true', () => {
    render(<ChatPage />);
    expect(screen.getByTestId('system-panel')).toBeInTheDocument();
  });

  it('hides SystemPanel when systemPanelOpen is false', () => {
    (useAppStore as any).setState({ systemPanelOpen: false });
    render(<ChatPage />);
    expect(screen.queryByTestId('system-panel')).not.toBeInTheDocument();
  });
});
