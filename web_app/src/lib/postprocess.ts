import type * as ort from 'onnxruntime-web/webgpu';
import type { ModelAssets } from './manifest';
import type { LetterboxMeta } from './preprocess';
import type { RecognizedTile } from './tile';

export type BoundingBox = [number, number, number, number];

export type RawDetection = {
  classId: number;
  score: number;
  bbox: BoundingBox;
};

export function decodeOutput(output: ort.Tensor, meta: LetterboxMeta, assets: ModelAssets): RecognizedTile[] {
  const detections = decodeRows(output.dims, output.data as Float32Array | number[], assets.classes.length);
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
      rotationDegrees: null,
    } satisfies RecognizedTile;
  });
}

export function decodeRows(dims: readonly number[], outputData: Float32Array | number[], labelCount: number): RawDetection[] {
  const data = Array.from(outputData);

  if (dims.length === 3 && dims[0] === 1) {
    const [, second, third] = dims;
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

export function decodeClassicChannels(data: number[], channels: number, anchors: number): RawDetection[] {
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

export function decodeClassicRows(data: number[], rows: number, columns: number): RawDetection[] {
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

export function decodeEndToEndRows(data: number[], rows: number, columns: number): RawDetection[] {
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

export function xywhToXyxy([x, y, width, height]: [number, number, number, number]): BoundingBox {
  return [x - width / 2, y - height / 2, x + width / 2, y + height / 2];
}

export function restoreBoundingBox(box: BoundingBox, meta: LetterboxMeta): BoundingBox {
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

export function computeIou(a: BoundingBox, b: BoundingBox): number {
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

export function nonMaximumSuppression(detections: RawDetection[], iouThreshold: number): RawDetection[] {
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
