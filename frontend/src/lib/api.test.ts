import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  isTauri,
  fetchModels,
  fetchSavings,
  fetchServerInfo,
  checkHealth,
  fetchEnergy,
  fetchTelemetry,
  fetchTraces,
  transcribeAudio,
  fetchSpeechHealth,
  fetchManagedAgents,
  fetchManagedAgent,
  createManagedAgent,
  updateManagedAgent,
  deleteManagedAgent,
  pauseManagedAgent,
  resumeManagedAgent,
  fetchAgentTasks,
  createAgentTask,
  fetchAgentChannels,
  fetchTemplates,
  runManagedAgent,
  recoverManagedAgent,
  fetchAgentState,
  sendAgentMessage,
  fetchAgentMessages,
  fetchErrorAgents,
  fetchLearningLog,
  triggerLearning,
  fetchAgentTraces,
  fetchAgentTrace,
  submitSavings,
  type SavingsSubmission,
} from './api';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('isTauri', () => {
  afterEach(() => {
    delete (window as any).__TAURI_INTERNALS__;
  });

  it('returns false when __TAURI_INTERNALS__ is absent', () => {
    expect(isTauri()).toBe(false);
  });

  it('returns true when __TAURI_INTERNALS__ is present', () => {
    (window as any).__TAURI_INTERNALS__ = {};
    expect(isTauri()).toBe(true);
  });
});

describe('API - chat / models / server', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── fetchModels ─────────────────────────────────────────────────────
  describe('fetchModels', () => {
    it('returns models array from /v1/models', async () => {
      const models = [{ id: 'llama3', object: 'model', created: 0, owned_by: 'local' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ data: models })));

      const result = await fetchModels();
      expect(result).toEqual(models);
    });

    it('returns empty array when data is missing', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));
      const result = await fetchModels();
      expect(result).toEqual([]);
    });

    it('throws on non-ok response', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false, 500)));
      await expect(fetchModels()).rejects.toThrow('Failed to fetch models: 500');
    });
  });

  // ── fetchSavings ────────────────────────────────────────────────────
  describe('fetchSavings', () => {
    it('returns savings data', async () => {
      const savings = { total_calls: 10, total_tokens: 500 };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(savings)));
      const result = await fetchSavings();
      expect(result).toEqual(savings);
    });

    it('throws on error', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false, 503)));
      await expect(fetchSavings()).rejects.toThrow('Failed to fetch savings');
    });
  });

  // ── fetchServerInfo ─────────────────────────────────────────────────
  describe('fetchServerInfo', () => {
    it('returns server info', async () => {
      const info = { model: 'llama3', agent: null, engine: 'ollama' };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(info)));
      const result = await fetchServerInfo();
      expect(result).toEqual(info);
    });

    it('throws on error', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false, 500)));
      await expect(fetchServerInfo()).rejects.toThrow('Failed to fetch server info');
    });
  });

  // ── checkHealth ─────────────────────────────────────────────────────
  describe('checkHealth', () => {
    it('returns true when server responds ok', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
      expect(await checkHealth()).toBe(true);
    });

    it('returns false when server responds not ok', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }));
      expect(await checkHealth()).toBe(false);
    });

    it('returns false on network error', async () => {
      vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network')));
      expect(await checkHealth()).toBe(false);
    });
  });

  // ── fetchEnergy ─────────────────────────────────────────────────────
  describe('fetchEnergy', () => {
    it('returns energy data', async () => {
      const data = { watts: 42 };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(data)));
      const result = await fetchEnergy();
      expect(result).toEqual(data);
    });
  });

  // ── fetchTelemetry ──────────────────────────────────────────────────
  describe('fetchTelemetry', () => {
    it('returns telemetry data', async () => {
      const data = { requests: 100 };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(data)));
      const result = await fetchTelemetry();
      expect(result).toEqual(data);
    });
  });

  // ── fetchTraces ─────────────────────────────────────────────────────
  describe('fetchTraces', () => {
    it('fetches traces with default limit', async () => {
      const data = { traces: [] };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(data));
      vi.stubGlobal('fetch', mockFetch);
      await fetchTraces();
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('limit=50'));
    });

    it('fetches traces with custom limit', async () => {
      const data = { traces: [] };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(data));
      vi.stubGlobal('fetch', mockFetch);
      await fetchTraces(10);
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('limit=10'));
    });
  });
});

describe('API - speech', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('transcribeAudio', () => {
    it('sends FormData with blob and returns transcription', async () => {
      const result = { text: 'hello', language: 'en', confidence: 0.95, duration_seconds: 1.2 };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(result));
      vi.stubGlobal('fetch', mockFetch);

      const blob = new Blob(['audio'], { type: 'audio/webm' });
      const transcription = await transcribeAudio(blob);

      expect(transcription).toEqual(result);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/speech/transcribe'),
        expect.objectContaining({ method: 'POST' }),
      );
    });

    it('throws on non-ok response', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false, 400)));
      const blob = new Blob(['audio'], { type: 'audio/webm' });
      await expect(transcribeAudio(blob)).rejects.toThrow('Transcription failed');
    });
  });

  describe('fetchSpeechHealth', () => {
    it('returns speech health when available', async () => {
      const health = { available: true, backend: 'whisper' };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(health)));
      const result = await fetchSpeechHealth();
      expect(result).toEqual(health);
    });

    it('returns unavailable on non-ok response', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false, 500)));
      const result = await fetchSpeechHealth();
      expect(result.available).toBe(false);
    });
  });
});

describe('API - agents', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('fetchManagedAgents', () => {
    it('returns agents array', async () => {
      const agents = [{ id: 'a1', name: 'Agent 1', agent_type: 'research', status: 'idle' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ agents })));
      const result = await fetchManagedAgents();
      expect(result).toEqual(agents);
    });

    it('returns empty array when agents key is missing', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({})));
      const result = await fetchManagedAgents();
      expect(result).toEqual([]);
    });
  });

  describe('fetchManagedAgent', () => {
    it('returns a single agent', async () => {
      const agent = { id: 'a1', name: 'Agent 1' };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(agent)));
      const result = await fetchManagedAgent('a1');
      expect(result).toEqual(agent);
    });
  });

  describe('createManagedAgent', () => {
    it('sends POST with agent config', async () => {
      const agent = { id: 'a1', name: 'New Agent' };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(agent));
      vi.stubGlobal('fetch', mockFetch);

      const result = await createManagedAgent({ name: 'New Agent' });
      expect(result).toEqual(agent);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    });
  });

  describe('updateManagedAgent', () => {
    it('sends PATCH with partial update', async () => {
      const agent = { id: 'a1', name: 'Updated' };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(agent));
      vi.stubGlobal('fetch', mockFetch);

      const result = await updateManagedAgent('a1', { name: 'Updated' });
      expect(result).toEqual(agent);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1'),
        expect.objectContaining({ method: 'PATCH' }),
      );
    });
  });

  describe('deleteManagedAgent', () => {
    it('sends DELETE request', async () => {
      const mockFetch = vi.fn().mockResolvedValue({ ok: true });
      vi.stubGlobal('fetch', mockFetch);
      await deleteManagedAgent('a1');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1'),
        expect.objectContaining({ method: 'DELETE' }),
      );
    });
  });

  describe('pauseManagedAgent', () => {
    it('sends POST to pause endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({ ok: true });
      vi.stubGlobal('fetch', mockFetch);
      await pauseManagedAgent('a1');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1/pause'),
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('resumeManagedAgent', () => {
    it('sends POST to resume endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({ ok: true });
      vi.stubGlobal('fetch', mockFetch);
      await resumeManagedAgent('a1');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1/resume'),
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('fetchAgentTasks', () => {
    it('returns tasks array', async () => {
      const tasks = [{ id: 't1', description: 'task 1' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ tasks })));
      const result = await fetchAgentTasks('a1');
      expect(result).toEqual(tasks);
    });
  });

  describe('createAgentTask', () => {
    it('sends POST with task description', async () => {
      const task = { id: 't1', description: 'research AI' };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(task));
      vi.stubGlobal('fetch', mockFetch);
      const result = await createAgentTask('a1', 'research AI');
      expect(result).toEqual(task);
    });
  });

  describe('fetchAgentChannels', () => {
    it('returns channel bindings', async () => {
      const bindings = [{ id: 'c1', channel_type: 'slack' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ bindings })));
      const result = await fetchAgentChannels('a1');
      expect(result).toEqual(bindings);
    });
  });

  describe('fetchTemplates', () => {
    it('returns templates array', async () => {
      const templates = [{ id: 'tpl1', name: 'Research' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ templates })));
      const result = await fetchTemplates();
      expect(result).toEqual(templates);
    });
  });

  describe('runManagedAgent', () => {
    it('sends POST to run endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({ ok: true });
      vi.stubGlobal('fetch', mockFetch);
      await runManagedAgent('a1');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1/run'),
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('recoverManagedAgent', () => {
    it('sends POST to recover endpoint', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ recovered: true })));
      const result = await recoverManagedAgent('a1');
      expect(result).toEqual({ recovered: true });
    });
  });

  describe('fetchAgentState', () => {
    it('returns full agent state', async () => {
      const state = { agent: {}, tasks: [], channels: [], messages: [], checkpoint: null };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(state)));
      const result = await fetchAgentState('a1');
      expect(result).toEqual(state);
    });
  });

  describe('sendAgentMessage', () => {
    it('sends message with default queued mode', async () => {
      const msg = { id: 'm1', content: 'hello' };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(msg));
      vi.stubGlobal('fetch', mockFetch);
      const result = await sendAgentMessage('a1', 'hello');
      expect(result).toEqual(msg);
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.mode).toBe('queued');
    });

    it('sends message with immediate mode', async () => {
      const msg = { id: 'm1', content: 'hi' };
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse(msg));
      vi.stubGlobal('fetch', mockFetch);
      await sendAgentMessage('a1', 'hi', 'immediate');
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.mode).toBe('immediate');
    });
  });

  describe('fetchAgentMessages', () => {
    it('returns messages array', async () => {
      const messages = [{ id: 'm1', content: 'hi' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ messages })));
      const result = await fetchAgentMessages('a1');
      expect(result).toEqual(messages);
    });
  });

  describe('fetchErrorAgents', () => {
    it('returns agents in error state', async () => {
      const agents = [{ id: 'a1', status: 'error' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ agents })));
      const result = await fetchErrorAgents();
      expect(result).toEqual(agents);
    });
  });

  describe('fetchLearningLog', () => {
    it('returns learning log entries', async () => {
      const log = [{ id: 'l1', event_type: 'learn' }];
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ learning_log: log })));
      const result = await fetchLearningLog('a1');
      expect(result).toEqual(log);
    });
  });

  describe('triggerLearning', () => {
    it('sends POST to learning/run endpoint', async () => {
      const mockFetch = vi.fn().mockResolvedValue({ ok: true });
      vi.stubGlobal('fetch', mockFetch);
      await triggerLearning('a1');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/managed-agents/a1/learning/run'),
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('fetchAgentTraces', () => {
    it('returns traces with default limit', async () => {
      const traces = [{ id: 'tr1', outcome: 'success' }];
      const mockFetch = vi.fn().mockResolvedValue(jsonResponse({ traces }));
      vi.stubGlobal('fetch', mockFetch);
      const result = await fetchAgentTraces('a1');
      expect(result).toEqual(traces);
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('limit=20'));
    });
  });

  describe('fetchAgentTrace', () => {
    it('returns trace detail', async () => {
      const trace = { id: 'tr1', steps: [] };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(trace)));
      const result = await fetchAgentTrace('a1', 'tr1');
      expect(result).toEqual(trace);
    });
  });
});

describe('API - Tauri detection', () => {
  afterEach(() => {
    delete (window as any).__TAURI_INTERNALS__;
  });

  it('checkHealth returns false in Tauri mode when invoke fails', async () => {
    (window as any).__TAURI_INTERNALS__ = {};
    // The dynamic import of @tauri-apps/api/core will fail in tests,
    // so checkHealth should catch and return false
    const result = await checkHealth();
    expect(result).toBe(false);
  });
});

describe('API - submitSavings (Supabase)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns true on successful submission (200)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200 }));
    const submission: SavingsSubmission = {
      anon_id: 'test-id',
      display_name: 'Tester',
      total_calls: 100,
      total_tokens: 5000,
      dollar_savings: 2.5,
      energy_wh_saved: 10,
      flops_saved: 1e9,
    };
    const result = await submitSavings(submission);
    expect(result).toBe(true);
  });

  it('returns true on 201 status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 201 }));
    const submission: SavingsSubmission = {
      anon_id: 'test-id',
      display_name: 'Tester',
      total_calls: 0,
      total_tokens: 0,
      dollar_savings: 0,
      energy_wh_saved: 0,
      flops_saved: 0,
    };
    const result = await submitSavings(submission);
    expect(result).toBe(true);
  });

  it('returns false on network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    const submission: SavingsSubmission = {
      anon_id: 'test-id',
      display_name: 'Tester',
      total_calls: 0,
      total_tokens: 0,
      dollar_savings: 0,
      energy_wh_saved: 0,
      flops_saved: 0,
    };
    const result = await submitSavings(submission);
    expect(result).toBe(false);
  });
});
