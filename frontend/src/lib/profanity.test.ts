import { describe, it, expect } from 'vitest';
import { isProfane } from './profanity';

describe('isProfane', () => {
  it('returns true for an exact blocked word', () => {
    expect(isProfane('fuck')).toBe(true);
  });

  it('is case-insensitive', () => {
    expect(isProfane('FUCK')).toBe(true);
    expect(isProfane('Damn')).toBe(true);
  });

  it('detects blocked words embedded in other words (substring match)', () => {
    expect(isProfane('motherfucker')).toBe(true);
  });

  it('strips non-alpha characters before checking', () => {
    expect(isProfane('f-u-c-k')).toBe(false); // each letter alone is not blocked
    expect(isProfane('sh!t')).toBe(false); // '!' removed, becomes "sh t" - no match
    expect(isProfane('shit!')).toBe(true); // becomes "shit"
  });

  it('returns false for clean text', () => {
    expect(isProfane('good morning')).toBe(false);
    expect(isProfane('great day today')).toBe(false);
    expect(isProfane('open source software')).toBe(false);
  });

  it('returns false for an empty string', () => {
    expect(isProfane('')).toBe(false);
  });

  it('detects profanity in a sentence', () => {
    expect(isProfane('What the hell is going on')).toBe(true);
  });

  it('detects partial match within a word', () => {
    // "asshole" contains "ass"
    expect(isProfane('asshole')).toBe(true);
  });

  it('handles multiple spaces and punctuation', () => {
    expect(isProfane('  good   morning  ')).toBe(false);
    expect(isProfane('great, damn, day')).toBe(true);
  });
});
