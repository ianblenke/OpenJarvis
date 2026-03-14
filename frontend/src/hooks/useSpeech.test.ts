import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSpeech } from './useSpeech';

// ---------------------------------------------------------------------------
// Mock the API module
// ---------------------------------------------------------------------------

vi.mock('../lib/api', () => ({
  transcribeAudio: vi.fn(),
  fetchSpeechHealth: vi.fn(),
}));

import { transcribeAudio, fetchSpeechHealth } from '../lib/api';

const mockTranscribeAudio = vi.mocked(transcribeAudio);
const mockFetchSpeechHealth = vi.mocked(fetchSpeechHealth);

// ---------------------------------------------------------------------------
// Mock MediaRecorder & getUserMedia
// ---------------------------------------------------------------------------

class MockMediaRecorder {
  state = 'inactive';
  ondataavailable: ((e: any) => void) | null = null;
  onstop: (() => void) | null = null;
  mimeType = 'audio/webm';

  start() {
    this.state = 'recording';
  }

  stop() {
    this.state = 'inactive';
    // Simulate delivering a data chunk before stop
    if (this.ondataavailable) {
      this.ondataavailable({ data: new Blob(['audio-data'], { type: 'audio/webm' }) });
    }
    // Call onstop async to simulate real behavior
    setTimeout(() => this.onstop?.(), 0);
  }
}

function createMockStream() {
  return {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useSpeech', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchSpeechHealth.mockResolvedValue({ available: true });

    // Install MediaRecorder mock
    (globalThis as any).MediaRecorder = MockMediaRecorder;

    // Install getUserMedia mock
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValue(createMockStream()),
      },
      writable: true,
      configurable: true,
    });
  });

  it('initializes with idle state and checks speech health', async () => {
    const { result } = renderHook(() => useSpeech());

    expect(result.current.state).toBe('idle');
    expect(result.current.isRecording).toBe(false);
    expect(result.current.isTranscribing).toBe(false);
    expect(result.current.error).toBeNull();

    // Wait for effect to run
    await vi.waitFor(() => {
      expect(mockFetchSpeechHealth).toHaveBeenCalled();
    });
  });

  it('sets available to false when speech health check fails', async () => {
    mockFetchSpeechHealth.mockRejectedValue(new Error('network'));

    const { result } = renderHook(() => useSpeech());

    await vi.waitFor(() => {
      expect(result.current.available).toBe(false);
    });
  });

  it('sets available to true when speech backend is available', async () => {
    mockFetchSpeechHealth.mockResolvedValue({ available: true, backend: 'whisper' });

    const { result } = renderHook(() => useSpeech());

    await vi.waitFor(() => {
      expect(result.current.available).toBe(true);
    });
  });

  it('transitions to recording state on startRecording', async () => {
    const { result } = renderHook(() => useSpeech());

    await act(async () => {
      await result.current.startRecording();
    });

    expect(result.current.state).toBe('recording');
    expect(result.current.isRecording).toBe(true);
  });

  it('sets error if microphone is not supported', async () => {
    Object.defineProperty(navigator, 'mediaDevices', {
      value: undefined,
      writable: true,
      configurable: true,
    });

    const { result } = renderHook(() => useSpeech());

    await act(async () => {
      await result.current.startRecording();
    });

    expect(result.current.error).toBe('Microphone not supported in this browser');
    expect(result.current.state).toBe('idle');
  });

  it('sets error when microphone access is denied', async () => {
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockRejectedValue(new Error('NotAllowedError')),
      },
      writable: true,
      configurable: true,
    });

    const { result } = renderHook(() => useSpeech());

    await act(async () => {
      await result.current.startRecording();
    });

    expect(result.current.error).toBe('Microphone access denied');
    expect(result.current.state).toBe('idle');
  });

  it('stopRecording transcribes audio and returns text', async () => {
    mockTranscribeAudio.mockResolvedValue({
      text: 'Hello world',
      language: 'en',
      confidence: 0.95,
      duration_seconds: 1.5,
    });

    const { result } = renderHook(() => useSpeech());

    await act(async () => {
      await result.current.startRecording();
    });

    let transcribedText = '';
    await act(async () => {
      transcribedText = await result.current.stopRecording();
    });

    expect(transcribedText).toBe('Hello world');
    expect(result.current.state).toBe('idle');
  });

  it('stopRecording rejects when not recording', async () => {
    const { result } = renderHook(() => useSpeech());

    await expect(
      act(async () => {
        await result.current.stopRecording();
      }),
    ).rejects.toThrow('Not recording');
  });

  it('rejects when transcription fails', async () => {
    mockTranscribeAudio.mockRejectedValue(new Error('Transcription service unavailable'));

    const { result } = renderHook(() => useSpeech());

    await act(async () => {
      await result.current.startRecording();
    });

    expect(result.current.isRecording).toBe(true);

    // stopRecording should reject when transcribeAudio fails
    let rejected = false;
    await act(async () => {
      try {
        await result.current.stopRecording();
      } catch {
        rejected = true;
      }
    });

    expect(rejected).toBe(true);
    expect(result.current.state).toBe('idle');
  });
});
