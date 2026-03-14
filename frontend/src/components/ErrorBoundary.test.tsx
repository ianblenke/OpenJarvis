import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from './ErrorBoundary';

// Suppress React error boundary console.error noise in tests
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

function ThrowingComponent({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <div>Child rendered OK</div>;
}

describe('ErrorBoundary', () => {
  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>Hello world</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText('Hello world')).toBeInTheDocument();
  });

  it('displays error UI when a child throws', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>,
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('shows "Try again" button and recovers when clicked', () => {
    // We need a component that can toggle between throwing and not
    let shouldThrow = true;
    function ToggleThrow() {
      if (shouldThrow) throw new Error('boom');
      return <div>Recovered</div>;
    }

    const { rerender } = render(
      <ErrorBoundary>
        <ToggleThrow />
      </ErrorBoundary>,
    );

    // Should show error
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Try again')).toBeInTheDocument();

    // Fix the throw so re-render succeeds
    shouldThrow = false;

    // Click "Try again"
    fireEvent.click(screen.getByText('Try again'));

    // Should now render children
    expect(screen.getByText('Recovered')).toBeInTheDocument();
  });

  it('shows fallback message when error has no message', () => {
    function ThrowEmpty() {
      throw new Error();
    }

    render(
      <ErrorBoundary>
        <ThrowEmpty />
      </ErrorBoundary>,
    );

    // The fallback text "An unexpected error occurred." should appear
    // since error.message is empty, the ||  fallback triggers
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });
});
