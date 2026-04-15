import { describe, expect, it } from 'vitest';
import { createLabelPalette, getLabelColor } from './labels';

describe('labels helpers', () => {
  it('creates deterministic suit colors', () => {
    const palette = createLabelPalette(['1m', '1p', '1s', '1z']);
    expect(palette['1m']).toBe('#b91c1c');
    expect(palette['1p']).toBe('#1d4ed8');
    expect(palette['1s']).toBe('#15803d');
    expect(palette['1z']).toBe('#7c3aed');
  });

  it('falls back to a neutral color for unknown labels', () => {
    expect(getLabelColor('UNKNOWN', {})).toBe('#111827');
  });
});
