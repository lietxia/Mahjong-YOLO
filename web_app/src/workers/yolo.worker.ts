/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web/webgpu';
import ortWasmJsepUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import ortWasmJsepMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';
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

ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  wasm: ortWasmJsepUrl,
  mjs: ortWasmJsepMjsUrl,
};

let workerAssets: ModelAssets | null = null;
let session: ort.InferenceSession | null = null;
let backend: RuntimeBackend | null = null;
let fallbackReason: string | null = null;

function postToMain(message: YoloWorkerResponse) {
  self.postMessage(message);
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Worker 推理失败。';
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
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
    });
    backend = 'webgpu';
    fallbackReason = null;
    return { backend, fallbackReason };
  } catch (error) {
    fallbackReason = error instanceof Error ? error.message : 'WebGPU 会话初始化失败。';
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    backend = 'wasm';
    return { backend, fallbackReason };
  }
}

async function inferImage(image: ImageBitmap): Promise<WorkerInferenceResult> {
  const initResult = await ensureSession();
  const currentAssets = workerAssets;
  const currentSession = session;

  if (!currentAssets || !currentSession) {
    throw new Error('推理会话未初始化。');
  }

  const startedAt = performance.now();

  try {
    const { data, meta } = preprocessImageBitmap(image, currentAssets.manifest.inputSize);
    const afterPreprocess = performance.now();
    const inputTensor = new ort.Tensor('float32', data, [1, 3, currentAssets.manifest.inputSize, currentAssets.manifest.inputSize]);
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
