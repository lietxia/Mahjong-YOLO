/// <reference lib="webworker" />

import type * as OrtTypes from 'onnxruntime-web';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmAsyncifyUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortWasmAsyncifyMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import type { ModelAssets } from '../lib/manifest';
import { decodeOutput } from '../lib/postprocess';
import { preprocessImageBitmap } from '../lib/preprocess';
import type {
  RuntimeBackend,
  WorkerInferenceResult,
  WorkerInitResult,
  YoloWorkerRequest,
  YoloWorkerResponse,
} from '../lib/yolo.messages';

type OrtModule = typeof import('onnxruntime-web');

let workerAssets: ModelAssets | null = null;
let ortModule: OrtModule | null = null;
let session: OrtTypes.InferenceSession | null = null;
let backend: RuntimeBackend | null = null;
let fallbackReason: string | null = null;

function postToMain(message: YoloWorkerResponse) {
  self.postMessage(message);
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Worker 推理失败。';
}

function configureOrtModule(runtime: OrtModule, targetBackend: RuntimeBackend) {
  runtime.env.wasm.proxy = false;
  runtime.env.wasm.numThreads = 1;
  runtime.env.wasm.wasmPaths =
    targetBackend === 'webgpu'
      ? {
          wasm: ortWasmAsyncifyUrl,
          mjs: ortWasmAsyncifyMjsUrl,
        }
      : {
          wasm: ortWasmUrl,
          mjs: ortWasmMjsUrl,
        };
}

async function loadOrtModule(targetBackend: RuntimeBackend): Promise<OrtModule> {
  const runtime =
    targetBackend === 'webgpu'
      ? await import('onnxruntime-web/webgpu')
      : await import('onnxruntime-web');

  configureOrtModule(runtime, targetBackend);
  return runtime;
}

async function ensureSession(nextAssets?: ModelAssets): Promise<WorkerInitResult> {
  if (nextAssets) {
    workerAssets = nextAssets;
  }

  if (session && backend) {
    return { backend, fallbackReason };
  }

  if (!workerAssets) {
    throw new Error('Worker 尚未收到模型资源。');
  }

  const modelUrl = new URL(`/model/${workerAssets.manifest.modelFile}`, self.location.origin).toString();

  try {
    const webgpuOrt = await loadOrtModule('webgpu');
    session = await webgpuOrt.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
    });
    ortModule = webgpuOrt;
    backend = 'webgpu';
    fallbackReason = null;
    return { backend, fallbackReason };
  } catch (error) {
    fallbackReason = error instanceof Error ? error.message : 'WebGPU 会话初始化失败。';
    const wasmOrt = await loadOrtModule('wasm');
    session = await wasmOrt.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    ortModule = wasmOrt;
    backend = 'wasm';
    return { backend, fallbackReason };
  }
}

async function inferImage(image: ImageBitmap): Promise<WorkerInferenceResult> {
  const initResult = await ensureSession();
  const currentAssets = workerAssets;
  const currentSession = session;
  const currentOrt = ortModule;

  if (!currentAssets || !currentSession || !currentOrt) {
    throw new Error('推理会话未初始化。');
  }

  const startedAt = performance.now();

  try {
    const { data, meta } = preprocessImageBitmap(image, currentAssets.manifest.inputSize);
    const afterPreprocess = performance.now();
    const inputTensor = new currentOrt.Tensor('float32', data, [1, 3, currentAssets.manifest.inputSize, currentAssets.manifest.inputSize]);
    const outputs = await currentSession.run({
      [currentSession.inputNames[0]]: inputTensor,
    });
    const afterInference = performance.now();
    const outputTensor = outputs[currentSession.outputNames[0]];
    const detections = decodeOutput(outputTensor, meta, currentAssets);
    const afterPostprocess = performance.now();

    return {
      detections,
      backend: initResult.backend,
      rawShape: [...outputTensor.dims],
      fallbackReason,
      timings: {
        preprocessMs: afterPreprocess - startedAt,
        inferenceMs: afterInference - afterPreprocess,
        postprocessMs: afterPostprocess - afterInference,
        totalMs: afterPostprocess - startedAt,
      },
    };
  } finally {
    image.close();
  }
}

self.onmessage = async (event: MessageEvent<YoloWorkerRequest>) => {
  const message = event.data;

  try {
    if (message.type === 'init') {
      const result = await ensureSession(message.assets);
      postToMain({
        id: message.id,
        type: 'init-result',
        result,
      });
      return;
    }

    if (message.type === 'infer') {
      const result = await inferImage(message.image);
      postToMain({
        id: message.id,
        type: 'infer-result',
        result,
      });
    }
  } catch (error) {
    postToMain({
      id: message.id,
      type: 'error',
      message: toErrorMessage(error),
    });
  }
};

export {};
