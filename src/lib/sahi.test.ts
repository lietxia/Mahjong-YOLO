import { describe, expect, it } from 'vitest';
import { computeSliceGrid, mergeDetections, remapDetections } from './sahi';
import type { RawDetection } from './postprocess';

describe('computeSliceGrid', () => {
  it('returns single region when image fits in one slice', () => {
    const regions = computeSliceGrid(200, 150, 320, 0.2);
    expect(regions).toHaveLength(1);
    expect(regions[0]).toEqual({ x: 0, y: 0, width: 200, height: 150 });
  });

  it('produces grid for normal image with all regions within bounds', () => {
    const regions = computeSliceGrid(1280, 720, 320, 0.2);
    expect(regions.length).toBeGreaterThan(1);

    for (const r of regions) {
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.width).toBeLessThanOrEqual(1280);
      expect(r.y + r.height).toBeLessThanOrEqual(720);
    }
  });

  it('each region has width/height >= sliceSize or touches image boundary', () => {
    const imageW = 1280;
    const imageH = 720;
    const sliceSize = 320;
    const regions = computeSliceGrid(imageW, imageH, sliceSize, 0.2);

    for (const r of regions) {
      const touchesRight = r.x + r.width === imageW;
      const touchesBottom = r.y + r.height === imageH;
      const touchesLeft = r.x === 0;
      const touchesTop = r.y === 0;
      const touchesBoundary = touchesRight || touchesBottom || touchesLeft || touchesTop;

      expect(
        (r.width >= sliceSize && r.height >= sliceSize) || touchesBoundary,
      ).toBe(true);
    }
  });

  it('boundary reverse-pull: last tile starts at max(0, imageW - sliceSize)', () => {
    const regions = computeSliceGrid(1000, 800, 512, 0.2);
    const rightmost = regions.reduce((max, r) => (r.x > max.x ? r : max), regions[0]);
    expect(rightmost.x).toBe(Math.max(0, 1000 - 512));
    expect(rightmost.width).toBe(512);

    const bottommost = regions.reduce((max, r) => (r.y > max.y ? r : max), regions[0]);
    expect(bottommost.y).toBe(Math.max(0, 800 - 512));
    expect(bottommost.height).toBe(512);
  });

  it('no duplicate regions after boundary reverse-pull', () => {
    const regions = computeSliceGrid(1000, 800, 512, 0.2);
    const keys = regions.map((r) => `${r.x},${r.y},${r.width},${r.height}`);
    expect(new Set(keys).size).toBe(regions.length);
  });

  it('exact divisibility: 640x640 sliceSize=320 overlap=0.2', () => {
    const regions = computeSliceGrid(640, 640, 320, 0.2);
    for (const r of regions) {
      expect(r.x + r.width).toBeLessThanOrEqual(640);
      expect(r.y + r.height).toBeLessThanOrEqual(640);
    }

    const stride = Math.round(320 * (1 - 0.2));
    const cols = Math.ceil((640 - 320) / stride) + 1;
    const rows = Math.ceil((640 - 320) / stride) + 1;
    expect(regions.length).toBeLessThanOrEqual(cols * rows);
  });
});

describe('remapDetections', () => {
  it('shifts bbox coordinates by offset', () => {
    const detections: RawDetection[] = [
      { classId: 0, score: 0.9, bbox: [10, 20, 30, 40] },
      { classId: 1, score: 0.8, bbox: [5, 15, 25, 35] },
    ];

    const remapped = remapDetections(detections, 100, 200);

    expect(remapped).toHaveLength(2);
    expect(remapped[0].bbox).toEqual([110, 220, 130, 240]);
    expect(remapped[1].bbox).toEqual([105, 215, 125, 235]);
    expect(remapped[0].classId).toBe(0);
    expect(remapped[0].score).toBeCloseTo(0.9);
  });

  it('does not mutate original detections', () => {
    const detections: RawDetection[] = [
      { classId: 0, score: 0.9, bbox: [10, 20, 30, 40] },
    ];

    remapDetections(detections, 100, 200);

    expect(detections[0].bbox).toEqual([10, 20, 30, 40]);
  });

  it('zero offset returns equivalent detections', () => {
    const detections: RawDetection[] = [
      { classId: 2, score: 0.7, bbox: [1, 2, 3, 4] },
    ];

    const remapped = remapDetections(detections, 0, 0);

    expect(remapped[0].bbox).toEqual([1, 2, 3, 4]);
  });
});

describe('mergeDetections', () => {
  it('merges overlapping detections from different slices via NMS', () => {
    const slice1: RawDetection[] = [
      { classId: 0, score: 0.95, bbox: [10, 10, 50, 50] },
    ];
    const slice2: RawDetection[] = [
      { classId: 0, score: 0.85, bbox: [12, 12, 52, 52] },
    ];

    const merged = mergeDetections([slice1, slice2], 0.5);

    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBeCloseTo(0.95);
  });

  it('suppresses overlapping detections of different classes (class-agnostic)', () => {
    const slice1: RawDetection[] = [
      { classId: 0, score: 0.9, bbox: [10, 10, 50, 50] },
    ];
    const slice2: RawDetection[] = [
      { classId: 1, score: 0.8, bbox: [10, 10, 50, 50] },
    ];

    const merged = mergeDetections([slice1, slice2], 0.5);

    expect(merged).toHaveLength(1);
    expect(merged[0].classId).toBe(0);
    expect(merged[0].score).toBeCloseTo(0.9, 5);
  });

  it('handles empty detection arrays', () => {
    const merged = mergeDetections([[], []], 0.5);
    expect(merged).toHaveLength(0);
  });
});
