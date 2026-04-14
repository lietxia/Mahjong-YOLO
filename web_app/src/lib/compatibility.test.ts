import { describe, expect, it } from 'vitest';
import { getBrowserCompatibility, getWebGpuAvailabilityState, isGpuAdapterUnavailableReason } from './compatibility';

const mockWorker = (() => undefined) as unknown as typeof Worker;
const mockFetch = (async () => new Response('{}')) as typeof fetch;
const mockCreateImageBitmap = (async () => ({}) as ImageBitmap) as typeof createImageBitmap;
const mockWasm = {} as typeof WebAssembly;

describe('browser compatibility helpers', () => {
  it('allows inference but reports wasm fallback when webgpu is missing', () => {
    const compatibility = getBrowserCompatibility({
      Worker: mockWorker,
      fetch: mockFetch,
      createImageBitmap: mockCreateImageBitmap,
      WebAssembly: mockWasm,
      navigator: {
        mediaDevices: {
          getUserMedia: async () => ({}) as MediaStream,
        },
      } as Navigator,
      location: { hostname: 'localhost' },
      isSecureContext: true,
    });

    expect(compatibility.inferenceSupported).toBe(true);
    expect(compatibility.cameraSupported).toBe(true);
    expect(compatibility.notices[0]).toContain('WebGPU');
  });

  it('blocks inference when worker and wasm are unavailable', () => {
    const compatibility = getBrowserCompatibility({
      fetch: mockFetch,
      location: { hostname: 'example.com' },
      isSecureContext: false,
    });

    expect(compatibility.inferenceSupported).toBe(false);
    expect(compatibility.blockingIssues.join('；')).toContain('Web Worker');
    expect(compatibility.blockingIssues.join('；')).toContain('WebAssembly');
    expect(compatibility.cameraSupported).toBe(false);
  });

  it('warns that camera needs a secure context', () => {
    const compatibility = getBrowserCompatibility({
      Worker: mockWorker,
      fetch: mockFetch,
      createImageBitmap: mockCreateImageBitmap,
      WebAssembly: mockWasm,
      navigator: {
        mediaDevices: {
          getUserMedia: async () => ({}) as MediaStream,
        },
        gpu: {},
      } as Navigator & { gpu?: unknown },
      location: { hostname: 'mahjong.example.com' },
      isSecureContext: false,
    });

    expect(compatibility.inferenceSupported).toBe(true);
    expect(compatibility.cameraSupported).toBe(false);
    expect(compatibility.notices.join('；')).toContain('HTTPS');
  });

  it('distinguishes api support from adapter acquisition failure', () => {
    expect(getWebGpuAvailabilityState(true, null, null)).toBe('supported');
    expect(getWebGpuAvailabilityState(true, 'wasm', 'Failed to get GPU adapter')).toBe('adapter-unavailable');
    expect(getWebGpuAvailabilityState(true, 'webgpu', null)).toBe('active');
    expect(getWebGpuAvailabilityState(false, null, null)).toBe('unsupported');
  });

  it('detects adapter unavailable fallback reason', () => {
    expect(isGpuAdapterUnavailableReason('Failed to get GPU adapter.')).toBe(true);
    expect(isGpuAdapterUnavailableReason('WebGPU session init failed')).toBe(false);
  });
});
