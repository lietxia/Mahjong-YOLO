import { describe, expect, it } from 'vitest';
import {
  getCameraErrorMessage,
  getCameraVideoConstraints,
  getFrameSampleIntervalMs,
  isLowEndDevice,
  LOW_END_SAMPLE_INTERVAL_MS,
  STANDARD_SAMPLE_INTERVAL_MS,
  WASM_SAMPLE_INTERVAL_MS,
} from './live-camera';

describe('live camera helpers', () => {
  it('detects low-end devices by hardware concurrency', () => {
    expect(isLowEndDevice(4)).toBe(true);
    expect(isLowEndDevice(8)).toBe(false);
    expect(isLowEndDevice(null)).toBe(false);
  });

  it('slows frame sampling for wasm fallback and low-end devices', () => {
    expect(getFrameSampleIntervalMs('webgpu', 8)).toBe(STANDARD_SAMPLE_INTERVAL_MS);
    expect(getFrameSampleIntervalMs('webgpu', 4)).toBe(LOW_END_SAMPLE_INTERVAL_MS);
    expect(getFrameSampleIntervalMs('wasm', 8)).toBe(WASM_SAMPLE_INTERVAL_MS);
  });

  it('returns lower camera constraints on low-end devices', () => {
    expect(getCameraVideoConstraints(4)).toEqual({
      facingMode: 'environment',
      width: { ideal: 960 },
      height: { ideal: 540 },
      frameRate: { ideal: 15, max: 20 },
    });
  });

  it('maps permission failures to a readable message', () => {
    const error = new DOMException('Permission denied', 'NotAllowedError');
    expect(getCameraErrorMessage(error)).toContain('权限被拒绝');
  });
});
