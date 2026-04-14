import * as ort from 'onnxruntime-web/webgpu';
import ortWasmJsepUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import ortWasmJsepMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';
import type { ModelAssets } from './model';
import type { RecognizedTile } from './tile';

ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  wasm: ortWasmJsepUrl,
  mjs: ortWasmJsepMjsUrl,
};

type LetterboxMeta = {
  scale: number;
  padX: number;
  padY: number;
  originalWidth: number;
  originalHeight: number;
};

export type InferenceOutcome = {
  detections: RecognizedTile[];
  backend: 'webgpu' | 'wasm';
  rawShape: number[];
};

type BoundingBox = [number, number, number, number];

export class YoloBrowserRunner {
  private session: ort.InferenceSession | null = null;
  private backend: 'webgpu' | 'wasm' | null = null;

  constructor(private readonly assets: ModelAssets) {}

  async ensureSession(): Promise<'webgpu' | 'wasm'> {
    if (this.session && this.backend) {
      return this.backend;
    }

    const modelUrl = `/model/${this.assets.manifest.modelFile}`;

    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu'],
      });
      this.backend = 'webgpu';
      return this.backend;
    } catch (error) {
      console.warn('WebGPU session creation failed, falling back to WASM.', error);
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });
      this.backend = 'wasm';
      return this.backend;
    }
  }

  async infer(image: HTMLImageElement): Promise<InferenceOutcome> {
    const backend = await this.ensureSession();
    const session = this.session;

    if (!session) {
      throw new Error('推理会话未初始化。');
    }

    const { tensor, meta } = preprocessImage(image, this.assets.manifest.inputSize);
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: tensor,
    };
    const outputs = await session.run(feeds);
    const output = outputs[session.outputNames[0]];
    const detections = decodeOutput(output, meta, this.assets);

    return {
      detections,
      backend,
      rawShape: [...output.dims],
    };
  }
}

function preprocessImage(image: HTMLImageElement, inputSize: number): { tensor: ort.Tensor; meta: LetterboxMeta } {
  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error('无法创建预处理 canvas。');
  }

  context.fillStyle = 'rgb(114, 114, 114)';
  context.fillRect(0, 0, inputSize, inputSize);

  const scale = Math.min(inputSize / image.naturalWidth, inputSize / image.naturalHeight);
  const drawWidth = image.naturalWidth * scale;
  const drawHeight = image.naturalHeight * scale;
  const padX = (inputSize - drawWidth) / 2;
  const padY = (inputSize - drawHeight) / 2;

  context.drawImage(image, padX, padY, drawWidth, drawHeight);

  const imageData = context.getImageData(0, 0, inputSize, inputSize).data;
  const floatData = new Float32Array(1 * 3 * inputSize * inputSize);
  const planeSize = inputSize * inputSize;

  for (let pixelIndex = 0; pixelIndex < planeSize; pixelIndex += 1) {
    const rgbaOffset = pixelIndex * 4;
    floatData[pixelIndex] = imageData[rgbaOffset] / 255;
    floatData[planeSize + pixelIndex] = imageData[rgbaOffset + 1] / 255;
    floatData[planeSize * 2 + pixelIndex] = imageData[rgbaOffset + 2] / 255;
  }

  return {
    tensor: new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]),
    meta: {
      scale,
      padX,
      padY,
      originalWidth: image.naturalWidth,
      originalHeight: image.naturalHeight,
    },
  };
}

function decodeOutput(output: ort.Tensor, meta: LetterboxMeta, assets: ModelAssets): RecognizedTile[] {
  const detections = decodeRows(output, assets.classes.length);
  const filtered = detections.filter((detection) => detection.score >= assets.manifest.confidenceThreshold);
  const nmsDetections = nonMaximumSuppression(filtered, assets.manifest.iouThreshold);

  return nmsDetections.map((detection) => {
    const bbox = restoreBoundingBox(detection.bbox, meta);
    const centerX = (bbox[0] + bbox[2]) / 2;
    const centerY = (bbox[1] + bbox[3]) / 2;
    return {
      classId: detection.classId,
      label: assets.classes[detection.classId] ?? `class_${detection.classId}`,
      confidence: detection.score,
      bbox,
      centerX,
      centerY,
    } satisfies RecognizedTile;
  });
}

type RawDetection = {
  classId: number;
  score: number;
  bbox: BoundingBox;
};

function decodeRows(output: ort.Tensor, labelCount: number): RawDetection[] {
  const dims = output.dims;
  const data = Array.from(output.data as Float32Array | number[]);

  if (dims.length === 3 && dims[0] === 1) {
    const [_, second, third] = dims;
    if (second >= 5 && third > 0) {
      if (third <= 16 && second > third) {
        return decodeEndToEndRows(data, second, third);
      }

      if (second <= 16 && third > second) {
        return decodeEndToEndRows(data, third, second);
      }

      if (second === labelCount + 4 || second > third) {
        return decodeClassicChannels(data, second, third);
      }

      return decodeClassicRows(data, second, third);
    }
  }

  if (dims.length === 2 && dims[1] >= 6) {
    return decodeEndToEndRows(data, dims[0], dims[1]);
  }

  throw new Error(`暂不支持的输出形状: ${JSON.stringify(dims)}`);
}

function decodeClassicChannels(data: number[], channels: number, anchors: number): RawDetection[] {
  const detections: RawDetection[] = [];

  for (let anchorIndex = 0; anchorIndex < anchors; anchorIndex += 1) {
    const x = data[anchorIndex];
    const y = data[anchors + anchorIndex];
    const width = data[anchors * 2 + anchorIndex];
    const height = data[anchors * 3 + anchorIndex];

    let classId = -1;
    let score = -Infinity;
    for (let classOffset = 4; classOffset < channels; classOffset += 1) {
      const candidate = data[anchors * classOffset + anchorIndex];
      if (candidate > score) {
        score = candidate;
        classId = classOffset - 4;
      }
    }

    if (classId >= 0) {
      detections.push({
        classId,
        score,
        bbox: xywhToXyxy([x, y, width, height]),
      });
    }
  }

  return detections;
}

function decodeClassicRows(data: number[], rows: number, columns: number): RawDetection[] {
  const detections: RawDetection[] = [];

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const base = rowIndex * columns;
    const x = data[base];
    const y = data[base + 1];
    const width = data[base + 2];
    const height = data[base + 3];
    let classId = -1;
    let score = -Infinity;
    for (let classOffset = 4; classOffset < columns; classOffset += 1) {
      const candidate = data[base + classOffset];
      if (candidate > score) {
        score = candidate;
        classId = classOffset - 4;
      }
    }
    if (classId >= 0) {
      detections.push({ classId, score, bbox: xywhToXyxy([x, y, width, height]) });
    }
  }

  return detections;
}

function decodeEndToEndRows(data: number[], rows: number, columns: number): RawDetection[] {
  const detections: RawDetection[] = [];

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const base = rowIndex * columns;
    const x1 = data[base];
    const y1 = data[base + 1];
    const x2 = data[base + 2];
    const y2 = data[base + 3];
    const score = data[base + 4];
    const classId = Math.round(data[base + 5] ?? 0);
    detections.push({ classId, score, bbox: [x1, y1, x2, y2] });
  }

  return detections;
}

function xywhToXyxy([x, y, width, height]: [number, number, number, number]): BoundingBox {
  return [x - width / 2, y - height / 2, x + width / 2, y + height / 2];
}

function restoreBoundingBox(box: BoundingBox, meta: LetterboxMeta): BoundingBox {
  const [x1, y1, x2, y2] = box;
  const restored: BoundingBox = [
    (x1 - meta.padX) / meta.scale,
    (y1 - meta.padY) / meta.scale,
    (x2 - meta.padX) / meta.scale,
    (y2 - meta.padY) / meta.scale,
  ];

  return [
    clamp(restored[0], 0, meta.originalWidth),
    clamp(restored[1], 0, meta.originalHeight),
    clamp(restored[2], 0, meta.originalWidth),
    clamp(restored[3], 0, meta.originalHeight),
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function computeIou(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);
  const intersection = width * height;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaximumSuppression(detections: RawDetection[], iouThreshold: number): RawDetection[] {
  const kept: RawDetection[] = [];
  const sorted = [...detections].sort((left, right) => right.score - left.score);

  while (sorted.length > 0) {
    const current = sorted.shift();
    if (!current) {
      break;
    }
    kept.push(current);
    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      if (sorted[index].classId === current.classId && computeIou(sorted[index].bbox, current.bbox) > iouThreshold) {
        sorted.splice(index, 1);
      }
    }
  }

  return kept;
}
