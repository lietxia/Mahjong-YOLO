import type { ModelAssets } from './manifest';
import type { WorkerInferenceResult, WorkerInitResult, YoloWorkerRequest, YoloWorkerResponse } from './yolo.messages';

export type InferenceOutcome = WorkerInferenceResult;

type PendingRequest = {
  expectedType: 'init-result' | 'infer-result';
  resolve: (value: WorkerInitResult | WorkerInferenceResult) => void;
  reject: (reason?: unknown) => void;
};

export class YoloBrowserRunner {
  private readonly worker: Worker;
  private requestId = 0;
  private initResult: WorkerInitResult | null = null;
  private pending = new Map<number, PendingRequest>();

  constructor(private readonly assets: ModelAssets) {
    this.worker = new Worker(new URL('../workers/yolo.worker.ts', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', this.handleMessage);
    this.worker.addEventListener('error', this.handleWorkerError);
  }

  async init(): Promise<WorkerInitResult> {
    if (this.initResult) {
      return this.initResult;
    }

    const result = await this.sendRequest<WorkerInitResult>(
      {
        id: this.nextRequestId(),
        type: 'init',
        assets: cloneAssetsForWorker(this.assets),
      },
      'init-result',
    );
    this.initResult = result;
    return result;
  }

  async infer(image: ImageBitmap): Promise<InferenceOutcome> {
    await this.init();
    const result = await this.sendRequest<WorkerInferenceResult>(
      {
        id: this.nextRequestId(),
        type: 'infer',
        image,
      },
      'infer-result',
      [image],
    );
    this.initResult = {
      backend: result.backend,
      fallbackReason: result.fallbackReason,
    };
    return result;
  }

  dispose() {
    this.worker.removeEventListener('message', this.handleMessage);
    this.worker.removeEventListener('error', this.handleWorkerError);
    this.worker.terminate();

    for (const pending of this.pending.values()) {
      pending.reject(new Error('Worker 已被释放。'));
    }
    this.pending.clear();
  }

  private nextRequestId(): number {
    this.requestId += 1;
    return this.requestId;
  }

  private sendRequest<T extends WorkerInitResult | WorkerInferenceResult>(
    request: YoloWorkerRequest,
    expectedType: 'init-result' | 'infer-result',
    transfer: Transferable[] = [],
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(request.id, {
        expectedType,
        resolve: (value) => resolve(value as T),
        reject,
      });
      this.worker.postMessage(request, transfer);
    });
  }

  private readonly handleMessage = (event: MessageEvent<YoloWorkerResponse>) => {
    const message = event.data;
    const pending = this.pending.get(message.id);

    if (!pending) {
      return;
    }

    this.pending.delete(message.id);

    if (message.type === 'error') {
      pending.reject(new Error(message.message));
      return;
    }

    if (message.type !== pending.expectedType) {
      pending.reject(new Error(`Worker 返回了意外消息类型：${message.type}`));
      return;
    }

    pending.resolve(message.result);
  };

  private readonly handleWorkerError = (event: ErrorEvent) => {
    const error = new Error(event.message || 'Worker 发生未处理异常。');

    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  };
}

function cloneAssetsForWorker(assets: ModelAssets): ModelAssets {
  return JSON.parse(JSON.stringify(assets)) as ModelAssets;
}

export type { WorkerInitResult } from './yolo.messages';
