/// <reference lib="webworker" />

import type * as OrtTypes from 'onnxruntime-web';
import type { ModelAssets } from '../lib/manifest';
import { resolveSahiConfig } from '../lib/manifest';
import { computeSliceGrid, remapDetections, mergeDetections } from '../lib/sahi';
import type { SliceRegion } from '../lib/sahi';
import { decodeOutput, decodeRows, restoreBoundingBox, filterSizeOutliers } from '../lib/postprocess';
import type { RawDetection } from '../lib/postprocess';
import { preprocessPixels } from '../lib/preprocess';
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

const WASM_CDN = 'https://pub-e3b5792ae4f24700b2b0f0a495d5256b.r2.dev';

function configureOrtModule(runtime: OrtModule, targetBackend: RuntimeBackend) {
  runtime.env.wasm.proxy = false;
  runtime.env.wasm.numThreads = 1;
  runtime.env.wasm.wasmPaths =
    targetBackend === 'webgpu'
      ? {
          wasm: `${WASM_CDN}/ort-wasm-simd-threaded.asyncify.wasm`,
          mjs: `${WASM_CDN}/ort-wasm-simd-threaded.asyncify.mjs`,
        }
      : {
          wasm: `${WASM_CDN}/ort-wasm-simd-threaded.wasm`,
          mjs: `${WASM_CDN}/ort-wasm-simd-threaded.mjs`,
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

  const modelFile = workerAssets.manifest.modelFile;
  const modelUrl = modelFile.startsWith('http') ? modelFile : new URL(`/model/${modelFile}`, self.location.origin).toString();

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

function extractSlicePixels(fullPixels: Uint8ClampedArray, fullWidth: number, slice: SliceRegion): Uint8ClampedArray {
  const result = new Uint8ClampedArray(slice.width * slice.height * 4);
  for (let y = 0; y < slice.height; y++) {
    const srcOffset = ((slice.y + y) * fullWidth + slice.x) * 4;
    const dstOffset = y * slice.width * 4;
    result.set(fullPixels.subarray(srcOffset, srcOffset + slice.width * 4), dstOffset);
  }
  return result;
}

async function inferSlice(
  slicePixels: Uint8ClampedArray,
  sliceWidth: number,
  sliceHeight: number,
): Promise<{ detections: RawDetection[]; rawShape: number[] }> {
  const currentAssets = workerAssets;
  const currentSession = session;
  const currentOrt = ortModule;
  if (!currentAssets || !currentSession || !currentOrt) {
    throw new Error('推理会话未初始化。');
  }

  const { data, meta } = preprocessPixels(slicePixels, sliceWidth, sliceHeight, currentAssets.manifest.inputSize);
  const inputTensor = new currentOrt.Tensor('float32', data, [1, 3, currentAssets.manifest.inputSize, currentAssets.manifest.inputSize]);
  const outputs = await currentSession.run({ [currentSession.inputNames[0]]: inputTensor });
  const outputTensor = outputs[currentSession.outputNames[0]];

  const detections = decodeRows(outputTensor.dims, outputTensor.data as Float32Array | number[], currentAssets.classes.length);
  const filtered = detections.filter((d) => d.score >= currentAssets.manifest.confidenceThreshold);

  const restored = filtered.map((d) => ({
    ...d,
    bbox: restoreBoundingBox(d.bbox, meta),
  }));

  return { detections: restored, rawShape: [...outputTensor.dims] };
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
  const sahi = resolveSahiConfig(currentAssets.manifest);

  try {
    const fullCanvas = new OffscreenCanvas(image.width, image.height);
    const fullCtx = fullCanvas.getContext('2d');
    if (!fullCtx) throw new Error('无法创建预处理 canvas。');
    fullCtx.drawImage(image, 0, 0);
    const fullPixels = fullCtx.getImageData(0, 0, image.width, image.height).data;

    const afterPreprocess = performance.now();

    if (!sahi.enabled) {
      const { data, meta } = preprocessPixels(fullPixels, image.width, image.height, currentAssets.manifest.inputSize);
      const inputTensor = new currentOrt.Tensor('float32', data, [1, 3, currentAssets.manifest.inputSize, currentAssets.manifest.inputSize]);
      const outputs = await currentSession.run({ [currentSession.inputNames[0]]: inputTensor });
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
    }

    const allRawDetections: RawDetection[][] = [];
    let capturedRawShape: number[] = [];

    if (sahi.includeFullImage) {
      const fullResult = await inferSlice(fullPixels, image.width, image.height);
      allRawDetections.push(fullResult.detections);
      capturedRawShape = fullResult.rawShape;
    }

    const slices = computeSliceGrid(image.width, image.height, sahi.sliceSize, sahi.overlapRatio);
    for (const slice of slices) {
      if (sahi.includeFullImage && slice.x === 0 && slice.y === 0 && slice.width === image.width && slice.height === image.height) {
        continue;
      }
      const slicePixels = extractSlicePixels(fullPixels, image.width, slice);
      const sliceResult = await inferSlice(slicePixels, slice.width, slice.height);
      const remapped = remapDetections(sliceResult.detections, slice.x, slice.y);
      allRawDetections.push(remapped);
      capturedRawShape = sliceResult.rawShape;
    }

    const afterInference = performance.now();

    const mergedRaw = mergeDetections(allRawDetections, currentAssets.manifest.iouThreshold);
    const sizeFiltered = currentAssets.manifest.sizeFilterEnabled !== false
      ? filterSizeOutliers(mergedRaw, currentAssets.manifest.sizeRatioThreshold ?? 0.5)
      : mergedRaw;
    const afterPostprocess = performance.now();

    const detections = sizeFiltered.map((d: RawDetection) => ({
      classId: d.classId,
      label: currentAssets.classes[d.classId] ?? `class_${d.classId}`,
      confidence: d.score,
      bbox: d.bbox,
      centerX: (d.bbox[0] + d.bbox[2]) / 2,
      centerY: (d.bbox[1] + d.bbox[3]) / 2,
      rotationDegrees: null,
    }));

    return {
      detections,
      backend: initResult.backend,
      rawShape: capturedRawShape,
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
