type CompatibilityEnvironment = {
  Worker?: typeof Worker;
  fetch?: typeof fetch;
  createImageBitmap?: typeof createImageBitmap;
  WebAssembly?: typeof WebAssembly;
  navigator?: Navigator & { gpu?: unknown };
  location?: Pick<Location, 'hostname'>;
  isSecureContext?: boolean;
};

export type BrowserCompatibility = {
  supportsWorker: boolean;
  supportsFetch: boolean;
  supportsImageBitmap: boolean;
  supportsWasm: boolean;
  supportsWebGpu: boolean;
  supportsCameraApi: boolean;
  secureContext: boolean;
  inferenceSupported: boolean;
  cameraSupported: boolean;
  blockingIssues: string[];
  notices: string[];
};

function isLocalhost(hostname: string | undefined): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1';
}

export function getBrowserCompatibility(
  environment: CompatibilityEnvironment = globalThis as CompatibilityEnvironment,
): BrowserCompatibility {
  const supportsWorker = typeof environment.Worker === 'function';
  const supportsFetch = typeof environment.fetch === 'function';
  const supportsImageBitmap = typeof environment.createImageBitmap === 'function';
  const supportsWasm = typeof environment.WebAssembly === 'object';
  const supportsWebGpu = Boolean(environment.navigator?.gpu);
  const supportsCameraApi = Boolean(environment.navigator?.mediaDevices?.getUserMedia);
  const secureContext = Boolean(environment.isSecureContext || isLocalhost(environment.location?.hostname));

  const blockingIssues: string[] = [];
  const notices: string[] = [];

  if (!supportsWorker) {
    blockingIssues.push('当前浏览器不支持 Web Worker，无法保持现有的后台推理链路。');
  }

  if (!supportsFetch) {
    blockingIssues.push('当前浏览器不支持 fetch，无法读取模型和静态资源。');
  }

  if (!supportsImageBitmap) {
    blockingIssues.push('当前浏览器不支持 createImageBitmap，无法把图像高效传给推理 Worker。');
  }

  if (!supportsWasm) {
    blockingIssues.push('当前浏览器不支持 WebAssembly，无法启用 ONNX Runtime Web。');
  }

  if (!supportsWebGpu && supportsWasm) {
    notices.push('未检测到 WebGPU，模型初始化后会自动回退到 WASM，推理速度会更慢。');
  }

  if (!supportsCameraApi) {
    notices.push('当前浏览器不支持 getUserMedia，仍可使用图片上传推理。');
  } else if (!secureContext) {
    notices.push('摄像头能力需要 HTTPS 或 localhost；当前环境下仅建议使用图片上传推理。');
  }

  return {
    supportsWorker,
    supportsFetch,
    supportsImageBitmap,
    supportsWasm,
    supportsWebGpu,
    supportsCameraApi,
    secureContext,
    inferenceSupported: blockingIssues.length === 0,
    cameraSupported: supportsCameraApi && secureContext,
    blockingIssues,
    notices,
  };
}
