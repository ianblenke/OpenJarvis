import type { SSEEvent } from '../types';
import { isTauri } from './api';

export interface ChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  stream: true;
}

const DESKTOP_API = 'http://127.0.0.1:8222';

function* parseSSELines(lines: string[]): Generator<SSEEvent> {
  let currentEvent: string | undefined;

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      currentEvent = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;
      yield { event: currentEvent, data };
      currentEvent = undefined;
    } else if (line.trim() === '') {
      currentEvent = undefined;
    }
  }
}

export async function* streamChat(
  request: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const base = import.meta.env.VITE_API_URL || (isTauri() ? DESKTOP_API : '');

  // In Tauri/WebKitGTK, ReadableStream-based SSE parsing is unreliable.
  // Use a non-streaming request and emit the content as a single chunk.
  if (isTauri()) {
    const response = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: false }),
      signal,
    });
    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.status}`);
    }
    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';
    if (content) {
      yield {
        event: undefined,
        data: JSON.stringify({
          choices: [{ index: 0, delta: { role: 'assistant', content } }],
          usage: data.usage,
        }),
      };
    }
    // Emit stop event
    yield {
      event: undefined,
      data: JSON.stringify({
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        usage: data.usage,
      }),
    };
    return;
  }

  const response = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  // Prefer ReadableStream for true streaming in browsers that support it
  if (response.body) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        yield* parseSSELines(lines);
      }
      if (buffer.trim()) {
        yield* parseSSELines([buffer]);
      }
    } finally {
      reader.releaseLock();
    }
  } else {
    // Fallback: read entire response
    const text = await response.text();
    yield* parseSSELines(text.split('\n'));
  }
}
