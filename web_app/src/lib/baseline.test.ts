import { describe, expect, it } from 'vitest';
import { compareWithBaseline } from './baseline';

describe('baseline comparison', () => {
  it('reports exact match when labels align', () => {
    const comparison = compareWithBaseline(['1m', '2m'], {
      imageName: 'sample.png',
      expectedOrderedTiles: ['1m', '2m'],
      notes: [],
    });

    expect(comparison.exactMatch).toBe(true);
    expect(comparison.mismatches).toHaveLength(0);
  });

  it('reports positional mismatches', () => {
    const comparison = compareWithBaseline(['1m', '3m'], {
      imageName: 'sample.png',
      expectedOrderedTiles: ['1m', '2m'],
      notes: [],
    });

    expect(comparison.exactMatch).toBe(false);
    expect(comparison.mismatches).toHaveLength(1);
    expect(comparison.mismatches[0]).toEqual({ index: 1, expected: '2m', actual: '3m' });
  });
});
