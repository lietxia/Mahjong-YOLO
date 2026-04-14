<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue';
import DetectionCanvas from './components/DetectionCanvas.vue';
import LiveDetectionCanvas from './components/LiveDetectionCanvas.vue';
import { clearRegisteredAssetCaches, registerAssetCacheServiceWorker, type AssetCacheStatus } from './lib/cache';
import { compareWithBaseline, findBaselineSample } from './lib/baseline';
import { getBrowserCompatibility } from './lib/compatibility';
import { getCameraErrorMessage, getCameraVideoConstraints, getFrameSampleIntervalMs, isLowEndDevice } from './lib/live-camera';
import { calculateMahjongScore, type AgariType, type ScoreContext, type Wind } from './lib/mahjong';
import { summarizeModelAssets } from './lib/labels';
import { loadModelAssets, type ModelAssets } from './lib/manifest';
import { countRedDora, sortTilesLeftToRight, toOrderedTileLabels, type RecognizedTile } from './lib/tile';
import type { InferenceTimings } from './lib/yolo.messages';
import type { InferenceOutcome, YoloBrowserRunner } from './lib/yolo';

type ResultSource = 'upload' | 'camera';

type LiveDetectionCanvasExpose = {
  captureFrameBitmap: () => Promise<ImageBitmap | null>;
  captureScreenshot: () => string | null;
  hasReadyFrame: () => boolean;
};

const modelAssets = ref<ModelAssets | null>(null);
const modelRunner = ref<YoloBrowserRunner | null>(null);
const imageUrl = ref<string | null>(null);
const selectedFile = ref<File | null>(null);
const selectedFileName = ref('');
const loadingAssets = ref(true);
const preparingModel = ref(false);
const runningInference = ref(false);
const assetError = ref<string | null>(null);
const inferenceError = ref<string | null>(null);
const statusMessage = ref('正在读取模型清单...');
const backendUsed = ref<'webgpu' | 'wasm' | null>(null);
const backendFallbackReason = ref<string | null>(null);
const outputShape = ref<number[]>([]);
const timingMetrics = ref<InferenceTimings | null>(null);
const modelReady = ref(false);
const uploadInferenceAttempted = ref(false);
const cameraInferenceAttempted = ref(false);
const cacheStatus = ref<AssetCacheStatus | null>(null);
const clearingCache = ref(false);
const uploadDetections = ref<RecognizedTile[]>([]);
const cameraDetections = ref<RecognizedTile[]>([]);
const resultSource = ref<ResultSource>('upload');
const scoreMessage = ref('');
const selectedImageName = ref('');
const liveCanvasRef = ref<LiveDetectionCanvasExpose | null>(null);
const cameraStream = ref<MediaStream | null>(null);
const cameraActive = ref(false);
const cameraPaused = ref(false);
const cameraStarting = ref(false);
const cameraError = ref<string | null>(null);
const cameraStatusMessage = ref('尚未启动摄像头实时检测。模型会在首次使用时按需初始化。');
const hardwareConcurrency = ref<number | null>(typeof navigator !== 'undefined' ? navigator.hardwareConcurrency ?? null : null);
const compatibility = ref(getBrowserCompatibility());

let cameraAnimationFrameId = 0;
let cameraInferenceInFlight = false;
let lastCameraSampleStartedAt = 0;

const scoreContext = reactive<ScoreContext>({
  fieldWind: 'east',
  seatWind: 'east',
  agariType: 'ron',
  riichi: false,
  ippatsu: false,
  agariIndex: 13,
  doraIndicators: [],
  uraIndicators: [],
});

const activeDetections = computed(() => (resultSource.value === 'camera' ? cameraDetections.value : uploadDetections.value));
const uploadOrderedDetections = computed(() => sortTilesLeftToRight(uploadDetections.value));
const cameraOrderedDetections = computed(() => sortTilesLeftToRight(cameraDetections.value));
const orderedTiles = computed(() => toOrderedTileLabels(activeDetections.value));
const redDoraCount = computed(() => countRedDora(orderedTiles.value));

const scoring = computed(() => calculateMahjongScore(orderedTiles.value, scoreContext));

const resultSourceLabel = computed(() => (resultSource.value === 'camera' ? '摄像头实时流' : '上传图片'));
const cameraSupported = computed(() => compatibility.value.cameraSupported);
const lowEndDevice = computed(() => isLowEndDevice(hardwareConcurrency.value));
const cameraSampleIntervalMs = computed(() => getFrameSampleIntervalMs(backendUsed.value, hardwareConcurrency.value));
const liveCanvasDetections = computed(() => (cameraPaused.value ? [] : cameraOrderedDetections.value));
const inferenceSupported = computed(() => compatibility.value.inferenceSupported);
const browserBlockingMessage = computed(() => compatibility.value.blockingIssues.join('；'));
const compatibilityNotices = computed(() => compatibility.value.notices);
const cacheStatusMessage = computed(() => cacheStatus.value?.message ?? '正在检查静态资源缓存策略...');
const modelPreparationLabel = computed(() => {
  if (preparingModel.value) {
    return '初始化中';
  }

  if (modelReady.value) {
    return '已就绪';
  }

  if (loadingAssets.value) {
    return '读取清单中';
  }

  if (assetError.value) {
    return '初始化失败';
  }

  return '按需加载';
});
const executionPathLabel = computed(() => {
  if (backendUsed.value === 'webgpu') {
    return 'Web Worker + WebGPU';
  }

  if (backendUsed.value === 'wasm') {
    return 'Web Worker + WASM';
  }

  return modelReady.value ? '等待推理请求' : '尚未建立会话';
});
const cameraEnvironmentLabel = computed(() => {
  if (!compatibility.value.supportsCameraApi) {
    return '浏览器不支持';
  }

  if (!compatibility.value.secureContext) {
    return '需要 HTTPS/localhost';
  }

  return '可用';
});
const modelLifecycleMessage = computed(() => {
  if (loadingAssets.value) {
    return '正在读取模型清单、类别和基线数据。首屏不会立即下载大体积 ONNX 模型。';
  }

  if (preparingModel.value) {
    return '正在按需初始化 Web Worker 与模型会话，完成后会自动继续当前检测路径。';
  }

  if (modelReady.value) {
    return backendUsed.value === 'webgpu'
      ? '模型会话已就绪，当前优先使用 WebGPU；后续上传推理和实时检测会复用同一条 Worker 通路。'
      : '模型会话已就绪，但当前运行在 WASM 回退路径；页面会继续保留既有功能，只是速度更慢。';
  }

  return '当前只预加载元数据；首次点击“运行 YOLO 推理”或“启动摄像头”时才会下载模型并建立 Worker 会话。';
});
const uploadEmptyMessage = computed(() => {
  if (!imageUrl.value) {
    return '请选择一张麻将手牌图片后再运行推理。';
  }

  if (runningInference.value) {
    return preparingModel.value ? '正在准备模型并执行图片推理，请稍候。' : '正在执行图片推理，请稍候。';
  }

  if (inferenceError.value) {
    return '';
  }

  if (!inferenceSupported.value) {
    return browserBlockingMessage.value;
  }

  if (preparingModel.value) {
    return '正在按需初始化模型与 Worker 会话。';
  }

  if (!uploadInferenceAttempted.value) {
    return modelReady.value ? '图片已选择，点击“运行 YOLO 推理”即可开始检测。' : '图片已选择；模型会在首次推理时按需初始化。';
  }

  if (uploadDetections.value.length === 0) {
    return '本次没有识别到麻将牌，可尝试更清晰、更正面的单排手牌图片。';
  }

  return '';
});
const cameraHintMessage = computed(() => {
  if (cameraError.value) {
    return '';
  }

  if (!cameraActive.value) {
    return '';
  }

  if (cameraStarting.value) {
    return '正在请求摄像头权限并等待视频流。';
  }

  if (!cameraInferenceAttempted.value && !cameraPaused.value) {
    return '摄像头已启动，正在等待首帧进入推理队列。';
  }

  if (cameraPaused.value && cameraDetections.value.length === 0) {
    return '实时检测已暂停，当前还没有可展示的检测结果。';
  }

  if (!cameraPaused.value && cameraInferenceAttempted.value && cameraDetections.value.length === 0) {
    return '当前采样帧未识别到麻将牌，可调整距离、光线或摆放角度。';
  }

  return '';
});
const liveCanvasPlaceholderMessage = computed(() => {
  if (!cameraSupported.value) {
    return compatibility.value.supportsCameraApi
      ? '当前环境缺少 HTTPS 或 localhost，无法启动摄像头。'
      : '当前浏览器不支持摄像头接口，只能使用图片上传推理。';
  }

  if (cameraStarting.value) {
    return '正在请求摄像头权限并连接视频流...';
  }

  if (!cameraActive.value) {
    return modelReady.value ? '启动摄像头后会在这里显示实时检测画面。' : '启动摄像头后会按需初始化模型并显示实时检测画面。';
  }

  if (cameraPaused.value) {
    return '实时检测已暂停，可恢复推理或继续截图。';
  }

  if (!cameraInferenceAttempted.value) {
    return '实时画面已连接，正在等待首帧推理结果...';
  }

  if (cameraDetections.value.length === 0) {
    return '当前采样帧未识别到麻将牌，请调整画面。';
  }

  return '实时检测画面';
});

const cameraCapabilityMessage = computed(() => {
  if (!cameraSupported.value) {
    return compatibility.value.supportsCameraApi
      ? '摄像头能力需要 HTTPS 或 localhost，当前环境下无法启用实时检测。'
      : '当前环境不支持 getUserMedia，无法启用实时摄像头检测。';
  }

  if (!compatibility.value.supportsWebGpu && !backendUsed.value) {
    return '当前浏览器未提供 WebGPU，首次模型初始化后会直接走 WASM 回退路径。';
  }

  if (backendUsed.value === 'wasm') {
    return `Worker 当前运行在 WASM 回退路径，实时检测已自动降到约 ${(cameraSampleIntervalMs.value / 1000).toFixed(1)} 秒一帧。`;
  }

  if (lowEndDevice.value) {
    return `检测到较低并发设备，实时检测会按约 ${(cameraSampleIntervalMs.value / 1000).toFixed(1)} 秒一帧采样，避免堆帧。`;
  }

  return '';
});

const readinessMessage = computed(() => {
  if (orderedTiles.value.length === 0) {
    return '还没有识别到可计算的牌。';
  }
  return scoring.value.message;
});

const modelInfo = computed(() => (modelAssets.value ? summarizeModelAssets(modelAssets.value) : null));

const matchedBaseline = computed(() => {
  if (resultSource.value !== 'upload') {
    return null;
  }

  return findBaselineSample(selectedImageName.value, modelAssets.value);
});

const baselineComparison = computed(() => {
  if (!matchedBaseline.value) {
    return null;
  }

  return compareWithBaseline(orderedTiles.value, matchedBaseline.value);
});

function parseIndicatorList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

watch(orderedTiles, (tiles, previousTiles) => {
  if (tiles.length === 0) {
    scoreContext.agariIndex = 0;
    return;
  }

  if (previousTiles.length === 0) {
    scoreContext.agariIndex = Math.min(tiles.length - 1, 13);
    return;
  }

  scoreContext.agariIndex = Math.min(scoreContext.agariIndex, tiles.length - 1);
});

function applyInferenceOutcome(source: ResultSource, outcome: InferenceOutcome) {
  if (source === 'upload') {
    uploadInferenceAttempted.value = true;
    uploadDetections.value = outcome.detections;
    statusMessage.value =
      outcome.backend === 'webgpu'
        ? '上传图片推理已在 Web Worker 中通过 WebGPU 完成。'
        : '上传图片推理当前来自 Web Worker 内的 WASM 回退路径。';
    scoreMessage.value = `识别完成：${outcome.detections.length} 个框，已按从左到右提取 ${toOrderedTileLabels(outcome.detections).length} 张牌。`;
  } else {
    cameraInferenceAttempted.value = true;
    cameraDetections.value = outcome.detections;
    cameraError.value = null;
    cameraStatusMessage.value =
      outcome.backend === 'webgpu'
        ? `实时检测运行中，当前按约 ${cameraSampleIntervalMs.value} ms 采样。`
        : `实时检测运行中，但 Worker 已回退到 WASM，当前按约 ${cameraSampleIntervalMs.value} ms 采样。`;
    scoreMessage.value = `实时检测：${outcome.detections.length} 个框，当前采样间隔约 ${cameraSampleIntervalMs.value} ms。`;
  }

  resultSource.value = source;
  backendUsed.value = outcome.backend;
  backendFallbackReason.value = outcome.fallbackReason;
  outputShape.value = outcome.rawShape;
  timingMetrics.value = outcome.timings;
}

function stopCameraLoop() {
  if (cameraAnimationFrameId !== 0) {
    window.cancelAnimationFrame(cameraAnimationFrameId);
    cameraAnimationFrameId = 0;
  }

  lastCameraSampleStartedAt = 0;
}

function releaseCameraStream() {
  stopCameraLoop();

  const stream = cameraStream.value;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  cameraStream.value = null;
  cameraActive.value = false;
  cameraPaused.value = false;
}

async function inferCurrentCameraFrame() {
  if (cameraInferenceInFlight || !modelRunner.value) {
    return;
  }

  const liveCanvas = liveCanvasRef.value;
  const streamAtStart = cameraStream.value;
  if (!liveCanvas || !streamAtStart) {
    return;
  }

  cameraInferenceInFlight = true;

  try {
    const imageBitmap = await liveCanvas.captureFrameBitmap();
    if (!imageBitmap) {
      return;
    }

    const outcome = await modelRunner.value.infer(imageBitmap);
    if (cameraStream.value !== streamAtStart) {
      return;
    }

    applyInferenceOutcome('camera', outcome);
  } catch (error) {
    cameraError.value = error instanceof Error ? error.message : '摄像头推理失败';
    cameraPaused.value = true;
    cameraStatusMessage.value = '实时检测已暂停，请恢复或重新启动摄像头。';
  } finally {
    cameraInferenceInFlight = false;
  }
}

function startCameraLoop() {
  stopCameraLoop();

  const tick = (now: number) => {
    if (!cameraActive.value || !cameraStream.value) {
      cameraAnimationFrameId = 0;
      return;
    }

    cameraAnimationFrameId = window.requestAnimationFrame(tick);

    if (cameraPaused.value || cameraInferenceInFlight || !modelRunner.value) {
      return;
    }

    const liveCanvas = liveCanvasRef.value;
    if (!liveCanvas?.hasReadyFrame()) {
      return;
    }

    if (now - lastCameraSampleStartedAt < cameraSampleIntervalMs.value) {
      return;
    }

    lastCameraSampleStartedAt = now;
    void inferCurrentCameraFrame();
  };

  cameraAnimationFrameId = window.requestAnimationFrame(tick);
}

async function startCamera() {
  if (!cameraSupported.value) {
    cameraError.value = cameraCapabilityMessage.value;
    cameraStatusMessage.value = cameraError.value;
    return;
  }

  const modelReadyForCamera = await ensureModelReady('camera');
  if (!modelReadyForCamera || !modelRunner.value) {
    cameraError.value = assetError.value ?? '模型尚未初始化完成，请稍后再试。';
    cameraStatusMessage.value = cameraError.value;
    return;
  }

  cameraStarting.value = true;
  cameraError.value = null;
  cameraStatusMessage.value = '正在请求摄像头权限...';

  try {
    releaseCameraStream();
    cameraDetections.value = [];
    cameraInferenceAttempted.value = false;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: getCameraVideoConstraints(hardwareConcurrency.value),
      audio: false,
    });

    cameraStream.value = stream;
    cameraActive.value = true;
    cameraPaused.value = false;
    cameraStatusMessage.value = '摄像头已启动，正在等待首帧推理结果。';
    startCameraLoop();
  } catch (error) {
    releaseCameraStream();
    cameraError.value = getCameraErrorMessage(error);
    cameraStatusMessage.value = cameraError.value;
  } finally {
    cameraStarting.value = false;
  }
}

function pauseCamera() {
  if (!cameraActive.value) {
    return;
  }

  cameraPaused.value = true;
  cameraStatusMessage.value = '已暂停实时检测，画面继续显示，可恢复推理或截图。';
}

function resumeCamera() {
  if (!cameraActive.value) {
    return;
  }

  cameraPaused.value = false;
  cameraError.value = null;
  lastCameraSampleStartedAt = 0;
  cameraStatusMessage.value = `已恢复实时检测，当前按约 ${cameraSampleIntervalMs.value} ms 采样。`;
}

function stopCamera() {
  releaseCameraStream();
  cameraDetections.value = [];
  cameraInferenceAttempted.value = false;
  cameraError.value = null;
  cameraStatusMessage.value = '摄像头已关闭。';

  if (resultSource.value === 'camera') {
    resultSource.value = 'upload';
    outputShape.value = [];
    timingMetrics.value = null;
    scoreMessage.value = uploadDetections.value.length > 0 ? '摄像头已关闭，当前已切回上传图片结果。' : '';
  }
}

function captureCameraScreenshot() {
  const screenshot = liveCanvasRef.value?.captureScreenshot();
  if (!screenshot) {
    cameraError.value = '当前还没有可截图的实时画面。';
    return;
  }

  const link = document.createElement('a');
  link.href = screenshot;
  link.download = `mahjong-camera-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.click();
  cameraStatusMessage.value = '已保存当前实时画面截图。';
}

async function loadAssetsMetadata() {
  loadingAssets.value = true;
  assetError.value = null;
  statusMessage.value = '正在读取模型清单、类别和基线数据...';

  try {
    modelAssets.value = await loadModelAssets();
    statusMessage.value = '页面已就绪。模型会在首次推理或启动摄像头时按需初始化。';
  } catch (error) {
    assetError.value = error instanceof Error ? error.message : '初始化模型资源失败';
    statusMessage.value = '模型元数据读取失败，请检查静态资源是否已完整部署。';
  } finally {
    loadingAssets.value = false;
  }
}

async function ensureModelReady(source: ResultSource): Promise<boolean> {
  if (!inferenceSupported.value) {
    assetError.value = browserBlockingMessage.value || '当前浏览器环境不支持 Web 推理。';
    statusMessage.value = '当前浏览器环境不满足推理要求。';
    return false;
  }

  if (!modelAssets.value) {
    await loadAssetsMetadata();
  }

  if (!modelAssets.value) {
    return false;
  }

  if (modelRunner.value && modelReady.value) {
    return true;
  }

  preparingModel.value = true;
  assetError.value = null;
  statusMessage.value =
    source === 'camera' ? '正在按需初始化模型与 Worker，会在完成后启动实时检测...' : '正在按需初始化模型与 Worker，会在完成后继续图片推理...';

  const { YoloBrowserRunner: LazyYoloBrowserRunner } = await import('./lib/yolo');
  const runner = modelRunner.value ?? new LazyYoloBrowserRunner(modelAssets.value);

  try {
    const initResult = await runner.init();
    modelRunner.value = runner;
    modelReady.value = true;
    backendUsed.value = initResult.backend;
    backendFallbackReason.value = initResult.fallbackReason;
    statusMessage.value =
      initResult.backend === 'webgpu'
        ? '模型初始化完成，当前使用 Web Worker + WebGPU。'
        : '模型初始化完成，但当前运行在 Web Worker 内的 WASM 回退路径。';
    return true;
  } catch (error) {
    runner.dispose();
    modelRunner.value = null;
    modelReady.value = false;
    backendUsed.value = null;
    backendFallbackReason.value = null;
    assetError.value = error instanceof Error ? error.message : '初始化模型资源失败';
    statusMessage.value = '模型初始化失败，请稍后重试或清理静态缓存。';
    return false;
  } finally {
    preparingModel.value = false;
  }
}

async function initializeCacheStrategy() {
  cacheStatus.value = await registerAssetCacheServiceWorker();
}

async function clearAssetCache() {
  clearingCache.value = true;

  try {
    const message = await clearRegisteredAssetCaches();
    cacheStatus.value = {
      supported: true,
      enabled: false,
      message,
    };
    statusMessage.value = '静态缓存已清理；如需重新建立缓存，请刷新页面。';
  } finally {
    clearingCache.value = false;
  }
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }

  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }

  imageUrl.value = URL.createObjectURL(file);
  selectedFile.value = file;
  selectedFileName.value = file.name;
  selectedImageName.value = file.name;
  uploadInferenceAttempted.value = false;
  uploadDetections.value = [];
  inferenceError.value = null;

  if (resultSource.value === 'upload') {
    outputShape.value = [];
    timingMetrics.value = null;
    scoreMessage.value = '';
  }
}

async function runInference() {
  if (!selectedFile.value) {
    return;
  }

  uploadInferenceAttempted.value = true;
  inferenceError.value = null;
  const ready = await ensureModelReady('upload');

  if (!ready || !modelRunner.value) {
    return;
  }

  runningInference.value = true;
  statusMessage.value = '正在通过 Web Worker 运行 YOLO 推理...';

  try {
    const imageBitmap = await createImageBitmap(selectedFile.value);
    const outcome: InferenceOutcome = await modelRunner.value.infer(imageBitmap);
    applyInferenceOutcome('upload', outcome);
  } catch (error) {
    inferenceError.value = error instanceof Error ? error.message : '推理失败';
  } finally {
    runningInference.value = false;
  }
}

function resetUpload() {
  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }
  imageUrl.value = null;
  selectedFile.value = null;
  selectedFileName.value = '';
  selectedImageName.value = '';
  uploadInferenceAttempted.value = false;
  uploadDetections.value = [];
  inferenceError.value = null;

  if (resultSource.value === 'upload') {
    statusMessage.value =
      modelReady.value
        ? backendUsed.value === 'webgpu'
          ? '已清空当前图片与识别结果，Worker 仍保持 WebGPU 就绪。'
          : '已清空当前图片与识别结果，Worker 当前仍使用 WASM 路径。'
        : '已清空当前图片与识别结果；模型会继续保持按需初始化。';
    outputShape.value = [];
    timingMetrics.value = null;
    scoreMessage.value = '';
  }
}

function formatTiming(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function updateFieldWind(event: Event) {
  scoreContext.fieldWind = (event.target as HTMLSelectElement).value as Wind;
}

function updateSeatWind(event: Event) {
  scoreContext.seatWind = (event.target as HTMLSelectElement).value as Wind;
}

function updateAgariType(event: Event) {
  scoreContext.agariType = (event.target as HTMLSelectElement).value as AgariType;
}

function updateAgariIndex(event: Event) {
  scoreContext.agariIndex = Number((event.target as HTMLInputElement).value);
}

function updateDoraIndicators(event: Event) {
  scoreContext.doraIndicators = parseIndicatorList((event.target as HTMLInputElement).value);
}

function updateUraIndicators(event: Event) {
  scoreContext.uraIndicators = parseIndicatorList((event.target as HTMLInputElement).value);
}

onMounted(() => {
  void loadAssetsMetadata();
  void initializeCacheStrategy();
});

onUnmounted(() => {
  releaseCameraStream();
  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }
  modelRunner.value?.dispose();
});
</script>

<template>
  <div class="page">
    <header class="page-header">
      <h1>Mahjong YOLO Phase 5 Web App</h1>
      <p>
        当前阶段保持既有的上传推理、实时摄像头、基线比对与麻将计分能力，同时补上按需模型初始化、缓存落点、兼容性说明与更完整的状态反馈。
      </p>
      <div class="status-banner">{{ statusMessage }}</div>
    </header>

    <div class="layout">
      <section>
        <div class="panel">
          <h2>图片上传与推理</h2>
          <div class="upload-row">
            <input type="file" accept="image/*" @change="onFileChange" />
            <span v-if="selectedFileName">当前图片：{{ selectedFileName }}</span>
          </div>

          <div class="action-row">
            <button :disabled="!imageUrl || runningInference || loadingAssets || preparingModel || !inferenceSupported" @click="runInference">
              {{ runningInference ? '推理中...' : preparingModel ? '初始化中...' : '运行 YOLO 推理' }}
            </button>
            <button class="secondary" :disabled="!imageUrl && uploadDetections.length === 0" @click="resetUpload">
              清空
            </button>
          </div>

          <div v-if="uploadEmptyMessage" class="message">{{ uploadEmptyMessage }}</div>

          <h3>实时摄像头检测</h3>
          <p class="footnote">
            继续复用 Phase 3 worker 推理通路，只在主线程做视频绘制；实时路径按固定频率采样，并且同一时间只允许一个推理在途。
          </p>

          <div class="action-row">
            <button :disabled="cameraActive || cameraStarting || loadingAssets || preparingModel || !cameraSupported || !inferenceSupported" @click="startCamera">
              {{ cameraStarting ? '启动中...' : preparingModel ? '初始化中...' : '启动摄像头' }}
            </button>
            <button class="secondary" :disabled="!cameraActive" @click="cameraPaused ? resumeCamera() : pauseCamera()">
              {{ cameraPaused ? '恢复检测' : '暂停检测' }}
            </button>
            <button class="secondary" :disabled="!cameraActive" @click="captureCameraScreenshot">截图</button>
            <button class="secondary" :disabled="!cameraActive" @click="stopCamera">关闭摄像头</button>
          </div>

          <div class="meta-grid">
            <div class="meta-item">
              <span class="meta-label">实时状态</span>
              {{ cameraActive ? (cameraPaused ? '已暂停' : '运行中') : '未启动' }}
            </div>
            <div class="meta-item">
              <span class="meta-label">采样频率</span>
              约 {{ cameraSampleIntervalMs }} ms / 帧
            </div>
            <div class="meta-item">
              <span class="meta-label">设备策略</span>
              {{ lowEndDevice ? '低负载模式' : '标准模式' }}
            </div>
          </div>

          <div class="meta-grid">
            <div class="meta-item">
              <span class="meta-label">模型准备</span>
              {{ modelPreparationLabel }}
            </div>
            <div class="meta-item">
              <span class="meta-label">当前执行路径</span>
              {{ executionPathLabel }}
            </div>
            <div class="meta-item">
              <span class="meta-label">WebGPU 能力</span>
              {{ compatibility.supportsWebGpu ? '浏览器可尝试 WebGPU' : '将走 WASM 回退' }}
            </div>
            <div class="meta-item">
              <span class="meta-label">摄像头环境</span>
              {{ cameraEnvironmentLabel }}
            </div>
            <div class="meta-item">
              <span class="meta-label">静态缓存</span>
              {{ cacheStatus?.enabled ? 'Service Worker 已启用' : '未启用 SW 缓存' }}
            </div>
          </div>

          <div class="message">{{ modelLifecycleMessage }}</div>
          <div class="message">{{ cacheStatusMessage }}</div>
          <div class="action-row compact">
            <button class="secondary" :disabled="clearingCache || loadingAssets || preparingModel || runningInference || cameraStarting" @click="clearAssetCache">
              {{ clearingCache ? '清理中...' : '清理静态缓存' }}
            </button>
          </div>

          <div v-if="browserBlockingMessage" class="message warning">当前浏览器环境不满足推理要求：{{ browserBlockingMessage }}</div>
          <div v-else-if="compatibilityNotices.length > 0" class="message">
            <strong>兼容性提示</strong>
            <ul class="result-list">
              <li v-for="notice in compatibilityNotices" :key="notice">
                {{ notice }}
              </li>
            </ul>
          </div>

          <div v-if="assetError" class="message warning">模型资源初始化失败：{{ assetError }}</div>
          <div v-if="inferenceError" class="message warning">推理失败：{{ inferenceError }}</div>
          <div class="message">{{ cameraStatusMessage }}</div>
          <div v-if="cameraHintMessage" class="message">{{ cameraHintMessage }}</div>
          <div v-if="cameraError" class="message warning">摄像头错误：{{ cameraError }}</div>
          <div v-else-if="cameraCapabilityMessage" class="message warning">{{ cameraCapabilityMessage }}</div>
          <div v-if="scoreMessage" class="message">{{ scoreMessage }}</div>

          <div v-if="modelInfo" class="meta-grid">
            <div class="meta-item">
              <span class="meta-label">模型文件</span>
              {{ modelInfo.model }}
            </div>
            <div class="meta-item">
              <span class="meta-label">输入尺寸</span>
              {{ modelInfo.inputSize }}
            </div>
            <div class="meta-item">
              <span class="meta-label">类别数</span>
              {{ modelInfo.classCount }}
            </div>
            <div class="meta-item">
              <span class="meta-label">阈值</span>
              conf={{ modelInfo.confidence }}, iou={{ modelInfo.iou }}
            </div>
            <div class="meta-item">
              <span class="meta-label">后端</span>
              {{ backendUsed ?? '未运行' }}
            </div>
            <div class="meta-item">
              <span class="meta-label">执行线程</span>
              Web Worker
            </div>
            <div class="meta-item">
              <span class="meta-label">输出形状</span>
              {{ outputShape.length > 0 ? JSON.stringify(outputShape) : '未返回' }}
            </div>
          </div>

          <div v-if="timingMetrics" class="meta-grid">
            <div class="meta-item metric-item">
              <span class="meta-label">预处理</span>
              <span class="metric-value">{{ formatTiming(timingMetrics.preprocessMs) }}</span>
              <span class="meta-helper">worker 内图像 letterbox 与 tensor 化</span>
            </div>
            <div class="meta-item metric-item">
              <span class="meta-label">模型推理</span>
              <span class="metric-value">{{ formatTiming(timingMetrics.inferenceMs) }}</span>
              <span class="meta-helper">ONNX Runtime Web 执行耗时</span>
            </div>
            <div class="meta-item metric-item">
              <span class="meta-label">后处理</span>
              <span class="metric-value">{{ formatTiming(timingMetrics.postprocessMs) }}</span>
              <span class="meta-helper">decode / 阈值过滤 / NMS</span>
            </div>
            <div class="meta-item metric-item">
              <span class="meta-label">总耗时</span>
              <span class="metric-value">{{ formatTiming(timingMetrics.totalMs) }}</span>
              <span class="meta-helper">不含主线程绘制时间</span>
            </div>
          </div>

          <div v-if="backendFallbackReason" class="message warning">
            WebGPU 未能在 worker 中启用，当前已回退到 WASM：{{ backendFallbackReason }}
          </div>

          <div v-if="orderedTiles.length > 0" class="message">
            当前结果来源：{{ resultSourceLabel }}。仍沿用单排手牌假设，按检测框中心点从左到右排序。
          </div>

          <div v-if="baselineComparison" class="message" :class="{ warning: !baselineComparison.exactMatch }">
            <template v-if="baselineComparison.exactMatch">
              基线比对通过：{{ baselineComparison.sample.imageName }} 的识别序列与当前内置基线一致。
            </template>
            <template v-else>
              基线比对未通过：当前 {{ baselineComparison.actualCount }} 张，基线 {{ baselineComparison.expectedCount }} 张，差异 {{ baselineComparison.mismatches.length }} 处。
            </template>
          </div>
        </div>

        <DetectionCanvas :image-url="imageUrl" :detections="uploadOrderedDetections" />
        <LiveDetectionCanvas
          ref="liveCanvasRef"
          :stream="cameraStream"
          :detections="liveCanvasDetections"
          :placeholder-message="liveCanvasPlaceholderMessage"
        />

        <div class="panel">
          <h2>识别到的牌</h2>
          <p>
            当前来源：{{ resultSourceLabel }}，共识别 {{ orderedTiles.length }} 张牌。当前只支持 <strong>14 张闭门手牌</strong>
            的最小和牌计算。
          </p>
          <div class="tile-list">
            <span
              v-for="(tile, index) in orderedTiles"
              :key="`${tile}-${index}`"
              class="tile-chip"
              :class="{
                unknown: !/^(?:[1-9]|0)[mpsz]$/.test(tile),
                agari: index === scoreContext.agariIndex,
              }"
            >
              {{ tile }}
            </span>
          </div>
          <div class="footnote">赤宝牌计数（按 0m/0p/0s 自动统计）：{{ redDoraCount }}</div>
          <div v-if="baselineComparison && baselineComparison.mismatches.length > 0" class="footnote">
            基线差异：
            {{ baselineComparison.mismatches.slice(0, 5).map((item) => `#${item.index}: ${item.expected ?? '∅'} → ${item.actual ?? '∅'}`).join(' / ') }}
          </div>
        </div>
      </section>

      <aside>
        <div class="panel">
          <h2>和牌上下文</h2>
          <div class="form-grid">
            <label class="field">
              <span>场风</span>
              <select :value="scoreContext.fieldWind" @change="updateFieldWind">
                <option value="east">东场</option>
                <option value="south">南场</option>
                <option value="west">西场</option>
                <option value="north">北场</option>
              </select>
            </label>

            <label class="field">
              <span>自风</span>
              <select :value="scoreContext.seatWind" @change="updateSeatWind">
                <option value="east">东家</option>
                <option value="south">南家</option>
                <option value="west">西家</option>
                <option value="north">北家</option>
              </select>
            </label>

            <label class="field">
              <span>和牌方式</span>
              <select :value="scoreContext.agariType" @change="updateAgariType">
                <option value="ron">荣和</option>
                <option value="tsumo">自摸</option>
              </select>
            </label>

            <label class="field">
              <span>和牌索引（0-based）</span>
              <input
                type="number"
                min="0"
                :max="Math.max(orderedTiles.length - 1, 0)"
                :value="scoreContext.agariIndex"
                @input="updateAgariIndex"
              />
            </label>

            <label class="field full">
              <span>宝牌指示牌（逗号分隔，如 3m,7p）</span>
              <input type="text" @input="updateDoraIndicators" placeholder="可留空" />
            </label>

            <label class="field full">
              <span>里宝牌指示牌（逗号分隔）</span>
              <input type="text" @input="updateUraIndicators" placeholder="可留空" />
            </label>

            <div class="field full">
              <span>额外状态</span>
              <div class="checkbox-row">
                <label>
                  <input v-model="scoreContext.riichi" type="checkbox" />
                  立直
                </label>
                <label>
                  <input v-model="scoreContext.ippatsu" type="checkbox" />
                  一发
                </label>
              </div>
            </div>
          </div>

          <div class="message" :class="{ warning: scoring.status === 'incomplete' }">
            {{ readinessMessage }}
          </div>
        </div>

        <div class="panel">
          <h2>和牌计算结果</h2>
          <template v-if="scoring.status === 'ready' && scoring.result">
            <div class="meta-grid">
              <div class="meta-item">
                <span class="meta-label">番数</span>
                {{ scoring.result.han }}
              </div>
              <div class="meta-item">
                <span class="meta-label">符数</span>
                {{ scoring.result.fu }}
              </div>
              <div class="meta-item">
                <span class="meta-label">点数 1</span>
                {{ scoring.result.point1 }}
              </div>
              <div class="meta-item">
                <span class="meta-label">点数 2</span>
                {{ scoring.result.point2 }}
              </div>
            </div>

            <h3>役种</h3>
            <ul class="result-list">
              <li v-for="(yaku, index) in scoring.result.yaku" :key="`${yaku}-${index}`">
                {{ yaku }}
              </li>
            </ul>

            <div v-if="scoring.result.fuMessages.length > 0" class="footnote">
              符说明：{{ scoring.result.fuMessages.join(' / ') }}
            </div>
            <div v-if="scoring.warnings.length > 0" class="message warning">
              {{ scoring.warnings.join('；') }}
            </div>
          </template>

          <template v-else>
            <div class="message warning">
              {{ scoring.message }}
            </div>
          </template>

          <div class="footnote">
            限制：Phase 5 仍不自动识别副露、宝牌指示牌、里宝牌、立直信息或桌面完整局况，需人工补录必要上下文。
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>
