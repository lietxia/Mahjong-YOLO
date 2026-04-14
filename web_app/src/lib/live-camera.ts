import type { RuntimeBackend } from './yolo.messages';

export const STANDARD_SAMPLE_INTERVAL_MS = 350;
export const LOW_END_SAMPLE_INTERVAL_MS = 650;
export const WASM_SAMPLE_INTERVAL_MS = 1000;
export const LOW_END_HARDWARE_CONCURRENCY = 4;

export function isLowEndDevice(hardwareConcurrency: number | null): boolean {
  return hardwareConcurrency !== null && hardwareConcurrency <= LOW_END_HARDWARE_CONCURRENCY;
}

export function getFrameSampleIntervalMs(backend: RuntimeBackend | null, hardwareConcurrency: number | null): number {
  if (backend === 'wasm') {
    return WASM_SAMPLE_INTERVAL_MS;
  }

  if (isLowEndDevice(hardwareConcurrency)) {
    return LOW_END_SAMPLE_INTERVAL_MS;
  }

  return STANDARD_SAMPLE_INTERVAL_MS;
}

export function getCameraVideoConstraints(hardwareConcurrency: number | null): MediaTrackConstraints {
  if (isLowEndDevice(hardwareConcurrency)) {
    return {
      facingMode: 'environment',
      width: { ideal: 960 },
      height: { ideal: 540 },
      frameRate: { ideal: 15, max: 20 },
    };
  }

  return {
    facingMode: 'environment',
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 24, max: 30 },
  };
}

export function getCameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
        return '摄像头权限被拒绝，请允许浏览器访问摄像头后重试。';
      case 'NotFoundError':
        return '没有检测到可用摄像头设备。';
      case 'NotReadableError':
      case 'AbortError':
        return '摄像头当前不可读，可能正被其他应用占用。';
      case 'OverconstrainedError':
        return '当前设备无法满足默认视频参数，请更换设备后重试。';
      case 'SecurityError':
        return '当前页面环境不允许访问摄像头，请使用 localhost 或 HTTPS。';
      default:
        return `打开摄像头失败：${error.message || error.name}`;
    }
  }

  return error instanceof Error ? error.message : '打开摄像头失败。';
}
