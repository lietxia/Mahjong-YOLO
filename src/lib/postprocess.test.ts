import { describe, expect, it } from 'vitest';
import { computeIou, decodeClassicChannels, nonMaximumSuppression, clusterByRow, filterSizeOutliers, boxArea } from './postprocess';
import type { RawDetection } from './postprocess';

describe('postprocess helpers', () => {
  it('decodes classic channel-first yolo output', () => {
    const anchors = 2;
    const channels = 6;
    const data = [
      100, 0,
      120, 0,
      20, 0,
      40, 0,
      0.9, 0.1,
      0.1, 0.2,
    ];

    const decoded = decodeClassicChannels(data, channels, anchors);
    expect(decoded).toHaveLength(2);
    expect(decoded[0].classId).toBe(0);
    expect(decoded[0].score).toBeCloseTo(0.9, 5);
  });

  it('suppresses overlapping boxes across classes (class-agnostic NMS)', () => {
    const kept = nonMaximumSuppression(
      [
        { classId: 0, score: 0.9, bbox: [0, 0, 10, 10] },
        { classId: 0, score: 0.8, bbox: [1, 1, 11, 11] },
        { classId: 1, score: 0.7, bbox: [1, 1, 11, 11] },
      ],
      0.5,
    );

    expect(kept).toHaveLength(1);
    expect(kept[0].classId).toBe(0);
    expect(kept[0].score).toBeCloseTo(0.9, 5);
  });

  it('computes box IoU', () => {
    expect(computeIou([0, 0, 10, 10], [5, 5, 15, 15])).toBeCloseTo(25 / 175, 5);
  });
});

describe('boxArea', () => {
  it('computes area of a bounding box', () => {
    expect(boxArea([10, 20, 30, 50])).toBe(20 * 30);
  });

  it('returns 0 for degenerate box', () => {
    expect(boxArea([10, 20, 5, 50])).toBe(0);
  });
});

describe('clusterByRow', () => {
  const makeDet = (classId: number, score: number, bbox: [number, number, number, number]): RawDetection => ({
    classId,
    score,
    bbox,
  });

  it('clusters same-row detections together', () => {
    const detections = [
      makeDet(0, 0.9, [10, 100, 50, 140]),
      makeDet(1, 0.8, [60, 100, 100, 140]),
      makeDet(2, 0.7, [110, 100, 150, 140]),
    ];
    const rows = clusterByRow(detections);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveLength(3);
  });

  it('splits detections into separate rows', () => {
    const detections = [
      makeDet(0, 0.9, [10, 100, 50, 140]),
      makeDet(1, 0.8, [60, 300, 100, 340]),
    ];
    const rows = clusterByRow(detections);
    expect(rows).toHaveLength(2);
  });

  it('handles empty input', () => {
    expect(clusterByRow([])).toHaveLength(0);
  });

  it('handles single detection', () => {
    const rows = clusterByRow([makeDet(0, 0.9, [10, 100, 50, 140])]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveLength(1);
  });
});

describe('filterSizeOutliers', () => {
  const makeDet = (classId: number, score: number, bbox: [number, number, number, number]) => ({
    classId,
    score,
    bbox: bbox as [number, number, number, number],
  });

  it('removes abnormally small detections in same row', () => {
    // Row of 5 tiles: 4 normal-sized + 1 tiny fragment
    const normal = makeDet(0, 0.9, [0, 100, 40, 140]);    // area 1600
    const normal2 = makeDet(1, 0.85, [50, 100, 90, 140]);  // area 1600
    const normal3 = makeDet(2, 0.8, [100, 100, 140, 140]); // area 1600
    const normal4 = makeDet(3, 0.75, [150, 100, 190, 140]);// area 1600
    const tiny = makeDet(4, 0.6, [200, 110, 210, 118]);    // area 80 (< 50% of 1600)

    const filtered = filterSizeOutliers([normal, normal2, normal3, normal4, tiny], 0.5);
    expect(filtered).toHaveLength(4);
    expect(filtered.every((d) => d.classId !== 4)).toBe(true);
  });

  it('preserves all detections when sizes are consistent', () => {
    const d1 = makeDet(0, 0.9, [0, 100, 40, 140]);
    const d2 = makeDet(1, 0.85, [50, 100, 90, 140]);
    const d3 = makeDet(2, 0.8, [100, 100, 140, 140]);

    const filtered = filterSizeOutliers([d1, d2, d3], 0.5);
    expect(filtered).toHaveLength(3);
  });

  it('skips filtering when fewer than 3 detections total', () => {
    const d1 = makeDet(0, 0.9, [0, 100, 40, 140]);
    const tiny = makeDet(4, 0.6, [200, 110, 210, 118]);

    const filtered = filterSizeOutliers([d1, tiny], 0.5);
    expect(filtered).toHaveLength(2);
  });

  it('skips filtering for rows with fewer than 3 detections', () => {
    const d1 = makeDet(0, 0.9, [0, 100, 40, 140]);   // row 1
    const d2 = makeDet(1, 0.85, [0, 400, 40, 440]);    // row 2
    const tiny = makeDet(4, 0.6, [50, 400, 55, 404]);  // row 2, tiny

    const filtered = filterSizeOutliers([d1, d2, tiny], 0.5);
    expect(filtered).toHaveLength(3); // row 2 only has 2 items, no filtering
  });
});
