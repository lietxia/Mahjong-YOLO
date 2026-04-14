import * as ort from 'onnxruntime-web/webgpu';

export type LetterboxMeta = {
  scale: number;
  padX: number;
  padY: number;
  originalWidth: number;
  originalHeight: number;
  inputSize: number;
};

export type PreprocessResult = {
  tensor: ort.Tensor;
  meta: LetterboxMeta;
};

export type PreprocessedImageData = {
  data: Float32Array;
  meta: LetterboxMeta;
};

const PADDING_VALUE = 114 / 255;

export type LetterboxPlacement = {
  scale: number;
  drawWidth: number;
  drawHeight: number;
  padX: number;
  padY: number;
  padRight: number;
  padBottom: number;
};

export function computeLetterboxPlacement(width: number, height: number, inputSize: number): LetterboxPlacement {
  const scale = Math.min(inputSize / width, inputSize / height);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const deltaWidth = inputSize - drawWidth;
  const deltaHeight = inputSize - drawHeight;
  const padX = Math.round(deltaWidth / 2 - 0.1);
  const padY = Math.round(deltaHeight / 2 - 0.1);
  const padRight = Math.round(deltaWidth / 2 + 0.1);
  const padBottom = Math.round(deltaHeight / 2 + 0.1);

  return { scale, drawWidth, drawHeight, padX, padY, padRight, padBottom };
}

export function preprocessImage(image: HTMLImageElement, inputSize: number): PreprocessResult {
  const processed = preprocessPixels(readHtmlImagePixels(image), image.naturalWidth, image.naturalHeight, inputSize);

  return {
    tensor: new ort.Tensor('float32', processed.data, [1, 3, inputSize, inputSize]),
    meta: processed.meta,
  };
}

export function preprocessImageBitmap(image: ImageBitmap, inputSize: number): PreprocessedImageData {
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error('无法创建 worker 预处理 canvas。');
  }

  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, image.width, image.height).data;
  return preprocessPixels(pixels, image.width, image.height, inputSize);
}

export function preprocessPixels(
  sourceImageData: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  inputSize: number,
): PreprocessedImageData {
  const placement = computeLetterboxPlacement(sourceWidth, sourceHeight, inputSize);
  const floatData = buildLetterboxedTensorData(sourceImageData, sourceWidth, sourceHeight, inputSize, placement);

  return {
    data: floatData,
    meta: {
      scale: placement.scale,
      padX: placement.padX,
      padY: placement.padY,
      originalWidth: sourceWidth,
      originalHeight: sourceHeight,
      inputSize,
    },
  };
}

export function buildLetterboxedTensorData(
  sourceImageData: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  inputSize: number,
  placement: LetterboxPlacement,
): Float32Array {
  const planeSize = inputSize * inputSize;
  const floatData = new Float32Array(3 * planeSize);
  floatData.fill(PADDING_VALUE);

  for (let targetY = 0; targetY < placement.drawHeight; targetY += 1) {
    const sourceY = mapTargetToSource(targetY, placement.drawHeight, sourceHeight);
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(y0 + 1, sourceHeight - 1);
    const yWeight = sourceY - y0;

    for (let targetX = 0; targetX < placement.drawWidth; targetX += 1) {
      const sourceX = mapTargetToSource(targetX, placement.drawWidth, sourceWidth);
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(x0 + 1, sourceWidth - 1);
      const xWeight = sourceX - x0;

      const topLeft = readPixel(sourceImageData, sourceWidth, x0, y0);
      const topRight = readPixel(sourceImageData, sourceWidth, x1, y0);
      const bottomLeft = readPixel(sourceImageData, sourceWidth, x0, y1);
      const bottomRight = readPixel(sourceImageData, sourceWidth, x1, y1);

      const red = bilinearInterpolate(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0], xWeight, yWeight) / 255;
      const green = bilinearInterpolate(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1], xWeight, yWeight) / 255;
      const blue = bilinearInterpolate(topLeft[2], topRight[2], bottomLeft[2], bottomRight[2], xWeight, yWeight) / 255;

      const canvasX = placement.padX + targetX;
      const canvasY = placement.padY + targetY;
      const pixelIndex = canvasY * inputSize + canvasX;
      floatData[pixelIndex] = red;
      floatData[planeSize + pixelIndex] = green;
      floatData[planeSize * 2 + pixelIndex] = blue;
    }
  }

  return floatData;
}

function mapTargetToSource(targetIndex: number, targetLength: number, sourceLength: number): number {
  const mapped = ((targetIndex + 0.5) * sourceLength) / targetLength - 0.5;
  return Math.min(Math.max(mapped, 0), sourceLength - 1);
}

function readPixel(sourceImageData: Uint8ClampedArray, sourceWidth: number, x: number, y: number): [number, number, number] {
  const offset = (y * sourceWidth + x) * 4;
  return [sourceImageData[offset], sourceImageData[offset + 1], sourceImageData[offset + 2]];
}

function bilinearInterpolate(
  topLeft: number,
  topRight: number,
  bottomLeft: number,
  bottomRight: number,
  xWeight: number,
  yWeight: number,
): number {
  const top = topLeft * (1 - xWeight) + topRight * xWeight;
  const bottom = bottomLeft * (1 - xWeight) + bottomRight * xWeight;
  return top * (1 - yWeight) + bottom * yWeight;
}

function readHtmlImagePixels(image: HTMLImageElement): Uint8ClampedArray {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error('无法创建预处理 canvas。');
  }

  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, image.naturalWidth, image.naturalHeight).data;
}
