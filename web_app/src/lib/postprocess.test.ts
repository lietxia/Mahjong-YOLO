import { describe, expect, it } from 'vitest';
import { computeIou, decodeClassicChannels, nonMaximumSuppression } from './postprocess';

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

  it('suppresses overlapping boxes of same class', () => {
    const kept = nonMaximumSuppression(
      [
        { classId: 0, score: 0.9, bbox: [0, 0, 10, 10] },
        { classId: 0, score: 0.8, bbox: [1, 1, 11, 11] },
        { classId: 1, score: 0.7, bbox: [1, 1, 11, 11] },
      ],
      0.5,
    );

    expect(kept).toHaveLength(2);
    expect(kept.map((item) => item.classId)).toEqual([0, 1]);
  });

  it('computes box IoU', () => {
    expect(computeIou([0, 0, 10, 10], [5, 5, 15, 15])).toBeCloseTo(25 / 175, 5);
  });
});
