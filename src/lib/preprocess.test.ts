import { describe, expect, it } from 'vitest';
import { buildLetterboxedTensorData, computeLetterboxPlacement } from './preprocess';

describe('preprocess helpers', () => {
  it('computes centered letterbox placement', () => {
    const placement = computeLetterboxPlacement(1280, 720, 640);

    expect(placement.scale).toBeCloseTo(0.5, 5);
    expect(placement.drawWidth).toBeCloseTo(640, 5);
    expect(placement.drawHeight).toBeCloseTo(360, 5);
    expect(placement.padX).toBeCloseTo(0, 5);
    expect(placement.padY).toBeCloseTo(140, 5);
  });

  it('builds deterministic tensor data from RGBA pixels', () => {
    const rgba = new Uint8ClampedArray([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      255, 255, 255, 255,
    ]);
    const placement = computeLetterboxPlacement(2, 2, 2);
    const tensor = buildLetterboxedTensorData(rgba, 2, 2, 2, placement);

    expect(Array.from(tensor.slice(0, 4))).toEqual([1, 0, 0, 1]);
    expect(Array.from(tensor.slice(4, 8))).toEqual([0, 1, 0, 1]);
    expect(Array.from(tensor.slice(8, 12))).toEqual([0, 0, 1, 1]);
  });
});
