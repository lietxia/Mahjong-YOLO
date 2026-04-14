import { describe, expect, it } from 'vitest';
import { countRedDora, hasUnsupportedTiles, selectPrimaryHorizontalRow, sortTilesLeftToRight, type RecognizedTile } from './tile';

describe('tile helpers', () => {
  it('sorts detections by horizontal order', () => {
    const tiles: RecognizedTile[] = [
      { classId: 1, label: '2m', confidence: 0.9, bbox: [50, 0, 60, 10], centerX: 55, centerY: 5 },
      { classId: 2, label: '1m', confidence: 0.8, bbox: [10, 0, 20, 10], centerX: 15, centerY: 5 },
    ];

    expect(sortTilesLeftToRight(tiles).map((tile) => tile.label)).toEqual(['1m', '2m']);
  });

  it('prefers the dominant horizontal row over an off-row outlier', () => {
    const tiles: RecognizedTile[] = [
      { classId: 1, label: '1m', confidence: 0.9, bbox: [10, 10, 30, 50], centerX: 20, centerY: 30 },
      { classId: 2, label: '2m', confidence: 0.92, bbox: [40, 12, 60, 52], centerX: 50, centerY: 32 },
      { classId: 3, label: '3m', confidence: 0.93, bbox: [70, 9, 90, 49], centerX: 80, centerY: 29 },
      { classId: 4, label: '9p', confidence: 0.88, bbox: [110, 80, 130, 120], centerX: 120, centerY: 100 },
    ];

    expect(selectPrimaryHorizontalRow(tiles).map((tile) => tile.label)).toEqual(['1m', '2m', '3m']);
    expect(sortTilesLeftToRight(tiles).map((tile) => tile.label)).toEqual(['1m', '2m', '3m']);
  });

  it('uses confidence sum to break equal row counts', () => {
    const tiles: RecognizedTile[] = [
      { classId: 1, label: '1m', confidence: 0.3, bbox: [10, 10, 30, 50], centerX: 20, centerY: 30 },
      { classId: 2, label: '2m', confidence: 0.31, bbox: [40, 12, 60, 52], centerX: 50, centerY: 32 },
      { classId: 3, label: '7p', confidence: 0.95, bbox: [15, 90, 35, 130], centerX: 25, centerY: 110 },
      { classId: 4, label: '8p', confidence: 0.96, bbox: [45, 88, 65, 128], centerX: 55, centerY: 108 },
    ];

    expect(selectPrimaryHorizontalRow(tiles).map((tile) => tile.label)).toEqual(['7p', '8p']);
  });

  it('counts red dora by tile code', () => {
    expect(countRedDora(['0m', '5m', '0s'])).toBe(2);
  });

  it('detects unsupported labels', () => {
    expect(hasUnsupportedTiles(['1m', 'UNKNOWN'])).toBe(true);
    expect(hasUnsupportedTiles(['1m', '9s'])).toBe(false);
  });
});
