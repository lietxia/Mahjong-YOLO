import { describe, expect, it } from 'vitest';
import { countRedDora, hasUnsupportedTiles, sortTilesLeftToRight, type RecognizedTile } from './tile';

describe('tile helpers', () => {
  it('sorts detections by horizontal order', () => {
    const tiles: RecognizedTile[] = [
      { classId: 1, label: '2m', confidence: 0.9, bbox: [50, 0, 60, 10], centerX: 55, centerY: 5 },
      { classId: 2, label: '1m', confidence: 0.8, bbox: [10, 0, 20, 10], centerX: 15, centerY: 5 },
    ];

    expect(sortTilesLeftToRight(tiles).map((tile) => tile.label)).toEqual(['1m', '2m']);
  });

  it('counts red dora by tile code', () => {
    expect(countRedDora(['0m', '5m', '0s'])).toBe(2);
  });

  it('detects unsupported labels', () => {
    expect(hasUnsupportedTiles(['1m', 'UNKNOWN'])).toBe(true);
    expect(hasUnsupportedTiles(['1m', '9s'])).toBe(false);
  });
});
