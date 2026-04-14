import type { BaselineSample, ModelAssets } from './manifest';

export type BaselineComparison = {
  sample: BaselineSample;
  actual: string[];
  expected: string[];
  exactMatch: boolean;
  expectedCount: number;
  actualCount: number;
  mismatches: Array<{
    index: number;
    expected: string | null;
    actual: string | null;
  }>;
};

export function findBaselineSample(imageName: string, assets: ModelAssets | null): BaselineSample | null {
  if (!assets) {
    return null;
  }

  return assets.baselines.find((sample) => sample.imageName === imageName) ?? null;
}

export function compareWithBaseline(actual: string[], sample: BaselineSample): BaselineComparison {
  const expected = sample.expectedOrderedTiles;
  const maxLength = Math.max(actual.length, expected.length);
  const mismatches: BaselineComparison['mismatches'] = [];

  for (let index = 0; index < maxLength; index += 1) {
    const expectedLabel = expected[index] ?? null;
    const actualLabel = actual[index] ?? null;
    if (expectedLabel !== actualLabel) {
      mismatches.push({
        index,
        expected: expectedLabel,
        actual: actualLabel,
      });
    }
  }

  return {
    sample,
    actual,
    expected,
    exactMatch: mismatches.length === 0,
    expectedCount: expected.length,
    actualCount: actual.length,
    mismatches,
  };
}
