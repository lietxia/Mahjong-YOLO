<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue';
import DetectionCanvas from './components/DetectionCanvas.vue';
import { clearRegisteredAssetCaches, registerAssetCacheServiceWorker, type AssetCacheStatus } from './lib/cache';
import { compareWithBaseline, findBaselineSample } from './lib/baseline';
import { getBrowserCompatibility, getWebGpuAvailabilityState, isGpuAdapterUnavailableReason } from './lib/compatibility';
import { calculateMahjongScore, type AgariType, type ScoreContext, type Wind } from './lib/mahjong';
import { summarizeModelAssets } from './lib/labels';
import { loadModelAssets, type ModelAssets } from './lib/manifest';
import { countRedDora, toOrderedTileLabels, type RecognizedTile } from './lib/tile';
import type { InferenceTimings } from './lib/yolo.messages';
import type { InferenceOutcome, YoloBrowserRunner } from './lib/yolo';

const modelAssets = ref<ModelAssets | null>(null);
const modelRunner = ref<YoloBrowserRunner | null>(null);
const selectedModelId = ref('');
const imageUrl = ref<string | null>(null);
const selectedFile = ref<File | null>(null);
const selectedFileName = ref('');
const selectedImageName = ref('');
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
const cacheStatus = ref<AssetCacheStatus | null>(null);
const clearingCache = ref(false);
const uploadDetections = ref<RecognizedTile[]>([]);
const scoreMessage = ref('');
const compatibility = ref(getBrowserCompatibility());

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

const modelOptions = computed(() => modelAssets.value?.manifest.models ?? []);
const orderedTiles = computed(() => toOrderedTileLabels(uploadDetections.value));
const redDoraCount = computed(() => countRedDora(orderedTiles.value));
const scoring = computed(() => calculateMahjongScore(orderedTiles.value, scoreContext));
const inferenceSupported = computed(() => compatibility.value.inferenceSupported);
const browserBlockingMessage = computed(() => compatibility.value.blockingIssues.join('；'));
const compatibilityNotices = computed(() =>
  compatibility.value.notices.filter((notice) => !/getUserMedia|HTTPS 或 localhost|摄像头/.test(notice)),
);
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
const webGpuAvailabilityState = computed(() =>
  getWebGpuAvailabilityState(compatibility.value.supportsWebGpu, backendUsed.value, backendFallbackReason.value),
);
const webGpuCapabilityLabel = computed(() => {
  switch (webGpuAvailabilityState.value) {
    case 'active':
      return 'WebGPU 已启用';
    case 'adapter-unavailable':
      return '支持 API，但当前拿不到 GPU adapter';
    case 'supported':
      return '浏览器支持 WebGPU API';
    case 'unsupported':
      return '浏览器未提供 WebGPU API';
  }
});
const activeModelLabel = computed(() => modelAssets.value?.manifest.activeModel.label ?? '未选择');
const activeModelFile = computed(() => modelAssets.value?.manifest.modelFile ?? '未加载');
const modelLifecycleMessage = computed(() => {
  if (loadingAssets.value) {
    return '正在读取模型清单、可选模型、类别和基线数据。首屏不会立即下载 ONNX 模型。';
  }

  if (preparingModel.value) {
    return `正在按需初始化 ${activeModelLabel.value} 的 Web Worker 与模型会话，完成后会继续当前图片推理。`;
  }

  if (modelReady.value) {
    return backendUsed.value === 'webgpu'
      ? `模型会话已就绪，当前模型为 ${activeModelLabel.value}，并优先使用 WebGPU。`
      : isGpuAdapterUnavailableReason(backendFallbackReason.value)
        ? `当前模型为 ${activeModelLabel.value}；浏览器已暴露 WebGPU API，但当前环境没有返回可用 GPU adapter，已回退到 WASM。`
        : `当前模型为 ${activeModelLabel.value}；模型会话已就绪，但当前运行在 WASM 回退路径。`;
  }

  return '当前只预加载元数据；首次点击“运行 YOLO 推理”时才会下载所选模型并建立 Worker 会话。';
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
    return modelReady.value
      ? `图片已选择，点击“运行 YOLO 推理”即可使用 ${activeModelLabel.value} 开始检测。`
      : `图片已选择；${activeModelLabel.value} 会在首次推理时按需初始化。`;
  }

  if (uploadDetections.value.length === 0) {
    return '本次没有识别到麻将牌，可尝试更清晰、更正面的单排手牌图片。';
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

const matchedBaseline = computed(() => findBaselineSample(selectedImageName.value, modelAssets.value));

const baselineComparison = computed(() => {
  if (!uploadInferenceAttempted.value || !matchedBaseline.value) {
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

function applyInferenceOutcome(outcome: InferenceOutcome) {
  uploadInferenceAttempted.value = true;
  uploadDetections.value = outcome.detections;
  statusMessage.value =
    outcome.backend === 'webgpu'
      ? `上传图片推理已在 Web Worker 中通过 WebGPU 完成，当前模型为 ${activeModelLabel.value}。`
      : `上传图片推理当前来自 Web Worker 内的 WASM 回退路径，当前模型为 ${activeModelLabel.value}。`;
  scoreMessage.value = `识别完成：${outcome.detections.length} 个框，画布展示全部检测框；已按主排从左到右提取 ${toOrderedTileLabels(outcome.detections).length} 张牌。`;
  backendUsed.value = outcome.backend;
  backendFallbackReason.value = outcome.fallbackReason;
  outputShape.value = outcome.rawShape;
  timingMetrics.value = outcome.timings;
}

function resetModelSession() {
  modelRunner.value?.dispose();
  modelRunner.value = null;
  modelReady.value = false;
  backendUsed.value = null;
  backendFallbackReason.value = null;
  outputShape.value = [];
  timingMetrics.value = null;
}

async function loadAssetsMetadata(modelId?: string): Promise<boolean> {
  loadingAssets.value = true;
  assetError.value = null;
  statusMessage.value = '正在读取模型清单、类别和基线数据...';

  try {
    const assets = await loadModelAssets(modelId);
    modelAssets.value = assets;
    selectedModelId.value = assets.manifest.activeModel.id;
    statusMessage.value = `页面已就绪。当前模型：${assets.manifest.activeModel.label}；首次推理时按需初始化。`;
    return true;
  } catch (error) {
    assetError.value = error instanceof Error ? error.message : '初始化模型资源失败';
    statusMessage.value = '模型元数据读取失败，请检查静态资源是否已完整部署。';
    return false;
  } finally {
    loadingAssets.value = false;
  }
}

async function ensureModelReady(): Promise<boolean> {
  if (!inferenceSupported.value) {
    assetError.value = browserBlockingMessage.value || '当前浏览器环境不支持 Web 推理。';
    statusMessage.value = '当前浏览器环境不满足推理要求。';
    return false;
  }

  if (!modelAssets.value) {
    const loaded = await loadAssetsMetadata(selectedModelId.value || undefined);
    if (!loaded) {
      return false;
    }
  }

  if (!modelAssets.value) {
    return false;
  }

  if (modelRunner.value && modelReady.value) {
    return true;
  }

  preparingModel.value = true;
  assetError.value = null;
  statusMessage.value = `正在按需初始化模型 ${modelAssets.value.manifest.activeModel.label} 与 Worker...`;

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
        ? `模型初始化完成，当前使用 ${modelAssets.value.manifest.activeModel.label} + Web Worker + WebGPU。`
        : `模型初始化完成，当前使用 ${modelAssets.value.manifest.activeModel.label} + Web Worker 内的 WASM 回退路径。`;
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

async function onModelChange(event: Event) {
  const nextModelId = (event.target as HTMLSelectElement).value;
  if (!nextModelId || nextModelId === selectedModelId.value) {
    return;
  }

  resetModelSession();
  uploadInferenceAttempted.value = false;
  uploadDetections.value = [];
  inferenceError.value = null;
  scoreMessage.value = '';
  statusMessage.value = '正在切换模型...';

  const loaded = await loadAssetsMetadata(nextModelId);
  if (loaded && modelAssets.value) {
    statusMessage.value = `已切换到 ${modelAssets.value.manifest.activeModel.label}，请重新运行推理。`;
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
  outputShape.value = [];
  timingMetrics.value = null;
  scoreMessage.value = '';
}

async function runInference() {
  if (!selectedFile.value) {
    return;
  }

  uploadInferenceAttempted.value = true;
  inferenceError.value = null;
  const ready = await ensureModelReady();

  if (!ready || !modelRunner.value) {
    return;
  }

  runningInference.value = true;
  statusMessage.value = `正在通过 Web Worker 运行 ${activeModelLabel.value} 的 YOLO 推理...`;

  try {
    const imageBitmap = await createImageBitmap(selectedFile.value);
    const outcome: InferenceOutcome = await modelRunner.value.infer(imageBitmap);
    applyInferenceOutcome(outcome);
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
  outputShape.value = [];
  timingMetrics.value = null;
  scoreMessage.value = '';
  statusMessage.value =
    modelReady.value && modelAssets.value
      ? backendUsed.value === 'webgpu'
        ? `已清空当前图片与识别结果，${modelAssets.value.manifest.activeModel.label} 仍保持 WebGPU 就绪。`
        : `已清空当前图片与识别结果，${modelAssets.value.manifest.activeModel.label} 当前仍使用 WASM 路径。`
      : '已清空当前图片与识别结果；模型会继续保持按需初始化。';
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
      <p>当前阶段保留上传推理、基线比对与麻将计分能力，已移除实时摄像头识别路径，并支持在多个 ONNX 模型之间切换。</p>
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

          <div v-if="modelOptions.length > 0" class="form-grid">
            <label class="field full">
              <span>推理模型</span>
              <select :value="selectedModelId" :disabled="loadingAssets || preparingModel || runningInference" @change="onModelChange">
                <option v-for="model in modelOptions" :key="model.id" :value="model.id">
                  {{ model.label }}
                </option>
              </select>
            </label>
          </div>

          <div class="action-row">
            <button :disabled="!imageUrl || runningInference || loadingAssets || preparingModel || !inferenceSupported" @click="runInference">
              {{ runningInference ? '推理中...' : preparingModel ? '初始化中...' : '运行 YOLO 推理' }}
            </button>
            <button class="secondary" :disabled="!imageUrl && uploadDetections.length === 0" @click="resetUpload">清空</button>
          </div>

          <div v-if="uploadEmptyMessage" class="message">{{ uploadEmptyMessage }}</div>

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
              {{ webGpuCapabilityLabel }}
            </div>
            <div class="meta-item">
              <span class="meta-label">静态缓存</span>
              {{ cacheStatus?.enabled ? 'Service Worker 已启用' : '未启用 SW 缓存' }}
            </div>
          </div>

          <div class="message">{{ modelLifecycleMessage }}</div>
          <div class="message">{{ cacheStatusMessage }}</div>
          <div class="action-row compact">
            <button class="secondary" :disabled="clearingCache || loadingAssets || preparingModel || runningInference" @click="clearAssetCache">
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
          <div v-if="scoreMessage" class="message">{{ scoreMessage }}</div>

          <div v-if="modelInfo" class="meta-grid">
            <div class="meta-item">
              <span class="meta-label">当前模型</span>
              {{ modelInfo.model }}
            </div>
            <div class="meta-item">
              <span class="meta-label">模型文件</span>
              {{ modelInfo.modelFile }}
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
            <template v-if="isGpuAdapterUnavailableReason(backendFallbackReason)">
              浏览器已暴露 WebGPU API，但当前环境没有返回可用 GPU adapter，应用已回退到 WASM：{{ backendFallbackReason }}
            </template>
            <template v-else>
              WebGPU 未能在 worker 中启用，当前已回退到 WASM：{{ backendFallbackReason }}
            </template>
          </div>

          <div v-if="orderedTiles.length > 0" class="message">
            当前结果来自上传图片。画布显示全部后处理检测框；计分与基线比对仍沿用单排手牌假设，并按检测框中心点从左到右排序主排结果。
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

        <DetectionCanvas :image-url="imageUrl" :detections="uploadDetections" />

        <div class="panel">
          <h2>识别到的牌</h2>
          <p>
            当前来源：上传图片，共识别 {{ orderedTiles.length }} 张牌。当前只支持 <strong>14 张闭门手牌</strong>
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
          <div class="footnote">当前模型文件：{{ activeModelFile }}</div>
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

            <div v-if="scoring.result.fuMessages.length > 0" class="footnote">符说明：{{ scoring.result.fuMessages.join(' / ') }}</div>
            <div v-if="scoring.warnings.length > 0" class="message warning">
              {{ scoring.warnings.join('；') }}
            </div>
          </template>

          <template v-else>
            <div class="message warning">
              {{ scoring.message }}
            </div>
          </template>

          <div class="footnote">限制：Phase 5 仍不自动识别副露、宝牌指示牌、里宝牌、立直信息或桌面完整局况，需人工补录必要上下文。</div>
        </div>
      </aside>
    </div>
  </div>
</template>
