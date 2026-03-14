import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageBubble } from './MessageBubble';
import type { ChatMessage } from '../../types';

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
});

describe('MessageBubble', () => {
  it('renders a user message with right-aligned styling', () => {
    const message: ChatMessage = {
      id: 'msg1',
      role: 'user',
      content: 'Hello assistant',
      timestamp: Date.now(),
    };

    const { container } = render(<MessageBubble message={message} />);
    expect(screen.getByText('Hello assistant')).toBeInTheDocument();
    // User messages should be right-aligned (flex justify-end)
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain('justify-end');
  });

  it('renders an assistant message with markdown', () => {
    const message: ChatMessage = {
      id: 'msg2',
      role: 'assistant',
      content: 'This is **bold** text',
      timestamp: Date.now(),
    };

    render(<MessageBubble message={message} />);
    // The bold text should be rendered
    expect(screen.getByText('bold')).toBeInTheDocument();
  });

  it('strips <think> tags from assistant content', () => {
    const message: ChatMessage = {
      id: 'msg3',
      role: 'assistant',
      content: '<think>internal reasoning</think>Visible answer',
      timestamp: Date.now(),
    };

    render(<MessageBubble message={message} />);
    expect(screen.getByText('Visible answer')).toBeInTheDocument();
    expect(screen.queryByText('internal reasoning')).not.toBeInTheDocument();
  });

  it('displays token usage when present', () => {
    const message: ChatMessage = {
      id: 'msg4',
      role: 'assistant',
      content: 'Response',
      timestamp: Date.now(),
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    };

    render(<MessageBubble message={message} />);
    expect(screen.getByText('30 tokens')).toBeInTheDocument();
  });

  it('renders tool call cards when present', () => {
    const message: ChatMessage = {
      id: 'msg5',
      role: 'assistant',
      content: 'Found results',
      timestamp: Date.now(),
      toolCalls: [
        {
          id: 'tc1',
          tool: 'web_search',
          arguments: '{"query":"test"}',
          status: 'success',
          result: 'Found 5 results',
          latency: 150,
        },
      ],
    };

    render(<MessageBubble message={message} />);
    expect(screen.getByText('Found results')).toBeInTheDocument();
    // Tool call card should show tool name
    expect(screen.getByText('web_search')).toBeInTheDocument();
  });

  it('handles empty assistant content after think tag removal', () => {
    const message: ChatMessage = {
      id: 'msg6',
      role: 'assistant',
      content: '<think>only thinking</think>',
      timestamp: Date.now(),
    };

    // Should not crash
    const { container } = render(<MessageBubble message={message} />);
    expect(container).toBeDefined();
  });
});
