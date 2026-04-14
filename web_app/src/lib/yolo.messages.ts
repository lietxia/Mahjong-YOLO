import type { ModelAssets } from './manifest';
import type { RecognizedTile } from './tile';

export type RuntimeBackend = 'webgpu' | 'wasm';

export type InferenceTimings = {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
};

export type WorkerInitResult = {
  backend: RuntimeBackend;
  fallbackReason: string | null;
};

export type WorkerInferenceResult = {
  detections: RecognizedTile[];
  backend: RuntimeBackend;
  rawShape: number[];
  timings: InferenceTimings;
  fallbackReason: string | null;
};

export type YoloWorkerRequest =
  | {
      id: number;
      type: 'init';
      assets: ModelAssets;
    }
  | {
      id: number;
      type: 'infer';
      image: ImageBitmap;
    };

export type YoloWorkerResponse =
  | {
      id: number;
      type: 'init-result';
      result: WorkerInitResult;
    }
  | {
      id: number;
      type: 'infer-result';
      result: WorkerInferenceResult;
    }
  | {
      id: number;
      type: 'error';
      message: string;
    };
