import { describe, it, expect, vi, beforeEach } from 'vitest';
import { streamChat, type ChatRequest } from './sse';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockResponse(chunks: string[], ok = true, status = 200) {
  let chunkIndex = 0;
  const encoder = new TextEncoder();
  const readable = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (chunkIndex < chunks.length) {
        controller.enqueue(encoder.encode(chunks[chunkIndex]));
        chunkIndex++;
      } else {
        controller.close();
      }
    },
  });

  return {
    ok,
    status,
    body: readable,
  } as unknown as Response;
}

const BASE_REQUEST: ChatRequest = {
  model: 'test-model',
  messages: [{ role: 'user', content: 'hello' }],
  stream: true,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('streamChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('yields SSE events from streamed response', async () => {
    const sseData =
      'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n' +
      'data: {"choices":[{"delta":{"content":" world"}}]}\n\n' +
      'data: [DONE]\n\n';

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createMockResponse([sseData])));

    const events: Array<{ event?: string; data: string }> = [];
    for await (const evt of streamChat(BASE_REQUEST)) {
      events.push(evt);
    }

    expect(events).toHaveLength(2);
    expect(events[0].data).toContain('Hello');
    expect(events[1].data).toContain(' world');
  });

  it('handles named events (event: field)', async () => {
    const sseData =
      'event: tool_start\n' +
      'data: {"tool":"search"}\n\n' +
      'data: [DONE]\n\n';

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createMockResponse([sseData])));

    const events: Array<{ event?: string; data: string }> = [];
    for await (const evt of streamChat(BASE_REQUEST)) {
      events.push(evt);
    }

    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('tool_start');
    expect(events[0].data).toContain('search');
  });

  it('handles chunked data split across multiple reads', async () => {
    const chunk1 = 'data: {"choices":[{"delta":{"content":"He';
    const chunk2 = 'llo"}}]}\n\ndata: [DONE]\n\n';

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createMockResponse([chunk1, chunk2])));

    const events: Array<{ event?: string; data: string }> = [];
    for await (const evt of streamChat(BASE_REQUEST)) {
      events.push(evt);
    }

    expect(events).toHaveLength(1);
    expect(events[0].data).toContain('Hello');
  });

  it('throws on non-ok response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(createMockResponse([], false, 500)),
    );

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of streamChat(BASE_REQUEST)) {
        // should not reach here
      }
    }).rejects.toThrow('Chat request failed: 500');
  });

  it('passes signal to fetch for abort support', async () => {
    const sseData = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n';
    const mockFetch = vi.fn().mockResolvedValue(createMockResponse([sseData]));
    vi.stubGlobal('fetch', mockFetch);

    const controller = new AbortController();
    const events: Array<{ event?: string; data: string }> = [];
    for await (const evt of streamChat(BASE_REQUEST, controller.signal)) {
      events.push(evt);
    }

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it('sends POST request with JSON body', async () => {
    const sseData = 'data: [DONE]\n\n';
    const mockFetch = vi.fn().mockResolvedValue(createMockResponse([sseData]));
    vi.stubGlobal('fetch', mockFetch);

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for await (const _ of streamChat(BASE_REQUEST)) {
      // drain
    }

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(BASE_REQUEST),
      }),
    );
  });

  it('returns no events for an empty stream', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createMockResponse([''])));

    const events: Array<{ event?: string; data: string }> = [];
    for await (const evt of streamChat(BASE_REQUEST)) {
      events.push(evt);
    }

    expect(events).toHaveLength(0);
  });
});
