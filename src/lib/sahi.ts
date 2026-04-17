import type { RawDetection } from './postprocess';
import { nonMaximumSuppression } from './postprocess';

export type SliceRegion = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export function computeSliceGrid(
  imageW: number,
  imageH: number,
  sliceSize: number,
  overlapRatio: number,
): SliceRegion[] {
  if (imageW <= sliceSize && imageH <= sliceSize) {
    return [{ x: 0, y: 0, width: imageW, height: imageH }];
  }

  const stride = Math.round(sliceSize * (1 - overlapRatio));
  const seen = new Set<string>();
  const regions: SliceRegion[] = [];

  const addUnique = (region: SliceRegion): void => {
    const key = `${region.x},${region.y},${region.width},${region.height}`;
    if (!seen.has(key)) {
      seen.add(key);
      regions.push(region);
    }
  };

  for (let yMin = 0; yMin < imageH; yMin += stride) {
    for (let xMin = 0; xMin < imageW; xMin += stride) {
      let xMax = Math.min(xMin + sliceSize, imageW);
      let yMax = Math.min(yMin + sliceSize, imageH);

      // Reverse-pull: ensure boundary tiles are full sliceSize
      const clampedXMin = Math.max(0, xMax - sliceSize);
      const clampedYMin = Math.max(0, yMax - sliceSize);

      addUnique({
        x: clampedXMin,
        y: clampedYMin,
        width: xMax - clampedXMin,
        height: yMax - clampedYMin,
      });
    }
  }

  return regions;
}

export function remapDetections(
  detections: RawDetection[],
  offsetX: number,
  offsetY: number,
): RawDetection[] {
  return detections.map((det) => ({
    classId: det.classId,
    score: det.score,
    bbox: [
      det.bbox[0] + offsetX,
      det.bbox[1] + offsetY,
      det.bbox[2] + offsetX,
      det.bbox[3] + offsetY,
    ] as [number, number, number, number],
  }));
}

export function mergeDetections(
  allDetections: RawDetection[][],
  iouThreshold: number,
): RawDetection[] {
  const flat = allDetections.flat();
  return nonMaximumSuppression(flat, iouThreshold);
}
