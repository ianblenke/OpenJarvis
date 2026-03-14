import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SettingsPage } from './SettingsPage';

// ---------------------------------------------------------------------------
// Mock API functions
// ---------------------------------------------------------------------------

vi.mock('../lib/api', () => ({
  checkHealth: vi.fn().mockResolvedValue(true),
  fetchSpeechHealth: vi.fn().mockResolvedValue({ available: false }),
}));

// ---------------------------------------------------------------------------
// Mock Zustand store
// ---------------------------------------------------------------------------

const mockUpdateSettings = vi.fn();
const mockLoadConversations = vi.fn();

const defaultSettings = {
  theme: 'system' as const,
  apiUrl: '',
  fontSize: 'default' as const,
  defaultModel: '',
  defaultAgent: '',
  temperature: 0.7,
  maxTokens: 4096,
  speechEnabled: false,
};

vi.mock('../lib/store', () => {
  const store = (selector: (s: any) => any) =>
    selector({
      settings: defaultSettings,
      updateSettings: mockUpdateSettings,
      conversations: [],
      serverInfo: null,
      loadConversations: mockLoadConversations,
    });

  return {
    useAppStore: Object.assign(store, {
      getState: () => ({
        settings: defaultSettings,
        updateSettings: mockUpdateSettings,
        conversations: [],
        serverInfo: null,
        loadConversations: mockLoadConversations,
      }),
      setState: () => {},
      subscribe: () => () => {},
    }),
  };
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the Settings heading', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders Appearance section with theme buttons', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Appearance')).toBeInTheDocument();
    expect(screen.getByText('Light')).toBeInTheDocument();
    expect(screen.getByText('Dark')).toBeInTheDocument();
    expect(screen.getByText('System')).toBeInTheDocument();
  });

  it('renders Connection section', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Connection')).toBeInTheDocument();
    expect(screen.getByText('Server status')).toBeInTheDocument();
    expect(screen.getByText('API URL')).toBeInTheDocument();
  });

  it('renders Model Defaults section with temperature and max tokens', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Model Defaults')).toBeInTheDocument();
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('Max tokens')).toBeInTheDocument();
  });

  it('renders Speech section', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Speech')).toBeInTheDocument();
    expect(screen.getByText('Speech-to-Text')).toBeInTheDocument();
  });

  it('renders Data section with export, import, clear buttons', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Data')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
    expect(screen.getByText('Import')).toBeInTheDocument();
    expect(screen.getByText('Clear')).toBeInTheDocument();
  });

  it('renders About section', () => {
    render(<SettingsPage />);
    expect(screen.getByText('About')).toBeInTheDocument();
    expect(screen.getByText('OpenJarvis')).toBeInTheDocument();
  });

  it('calls updateSettings when a theme button is clicked', () => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByText('Dark'));
    expect(mockUpdateSettings).toHaveBeenCalledWith({ theme: 'dark' });
  });

  it('renders font size selector', () => {
    render(<SettingsPage />);
    expect(screen.getByText('Font size')).toBeInTheDocument();
    // The select should have options
    const select = screen.getByDisplayValue('Default');
    expect(select).toBeInTheDocument();
  });
});
