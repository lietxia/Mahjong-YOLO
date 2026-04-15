<script setup lang="ts">
import { genFileId } from 'element-plus';
import type { UploadFile, UploadInstance, UploadProps, UploadRawFile } from 'element-plus';
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue';
import DetectionCanvas from './components/DetectionCanvas.vue';
import { clearRegisteredAssetCaches, registerAssetCacheServiceWorker, type AssetCacheStatus } from './lib/cache';
import { compareWithBaseline, findBaselineSample } from './lib/baseline';
import { getBrowserCompatibility, getWebGpuAvailabilityState, isGpuAdapterUnavailableReason } from './lib/compatibility';
import { calculateMahjongScore, type AgariType, type ScoreContext, type Wind } from './lib/mahjong';
import { summarizeModelAssets } from './lib/labels';
import { loadModelAssets, type ModelAssets } from './lib/manifest';
import { countRedDora, isMahjongTileCode, parseTileInput, toOrderedTileLabels, type RecognizedTile } from './lib/tile';
import type { InferenceTimings } from './lib/yolo.messages';
import type { InferenceOutcome, YoloBrowserRunner } from './lib/yolo';

const windOptions: Array<{ label: string; value: Wind }> = [
  { label: '东', value: 'east' },
  { label: '南', value: 'south' },
  { label: '西', value: 'west' },
  { label: '北', value: 'north' },
];

const agariTypeOptions: Array<{ label: string; value: AgariType }> = [
  { label: '荣和', value: 'ron' },
  { label: '自摸', value: 'tsumo' },
];

const modelAssets = ref<ModelAssets | null>(null);
const modelRunner = ref<YoloBrowserRunner | null>(null);
const uploadRef = ref<UploadInstance | null>(null);
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
const editableRecommendedHand = ref('');
const doraIndicatorsInput = ref('');
const uraIndicatorsInput = ref('');
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
const recommendedTiles = computed(() => toOrderedTileLabels(uploadDetections.value));
const recommendedHandText = computed(() => recommendedTiles.value.join(' '));
const scoringTiles = computed(() => parseTileInput(editableRecommendedHand.value));
const recommendedRedDoraCount = computed(() => countRedDora(recommendedTiles.value));
const scoringRedDoraCount = computed(() => countRedDora(scoringTiles.value));
const scoring = computed(() => calculateMahjongScore(scoringTiles.value, scoreContext));
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
  if (scoringTiles.value.length === 0) {
    return '还没有可用于计分的手牌。';
  }

  return scoring.value.message;
});
const modelInfo = computed(() => (modelAssets.value ? summarizeModelAssets(modelAssets.value) : null));
const rawDetectionPayload = computed(() =>
  uploadDetections.value.map((detection) => ({
    [detection.label]: [
      Number(detection.confidence.toFixed(3)),
      detection.bbox.map((value) => Number(value.toFixed(1))),
      [Number(detection.centerX.toFixed(1)), Number(detection.centerY.toFixed(1))],
    ],
  })),
);
const rawDetectionText = computed(() => JSON.stringify(rawDetectionPayload.value, null, 2));
const matchedBaseline = computed(() => findBaselineSample(selectedImageName.value, modelAssets.value));
const baselineComparison = computed(() => {
  if (!uploadInferenceAttempted.value || !matchedBaseline.value) {
    return null;
  }

  return compareWithBaseline(recommendedTiles.value, matchedBaseline.value);
});
const canRunInference = computed(() =>
  Boolean(imageUrl.value) && !runningInference.value && !loadingAssets.value && !preparingModel.value && inferenceSupported.value,
);
const canResetUpload = computed(() => Boolean(imageUrl.value) || uploadDetections.value.length > 0);

function parseIndicatorList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

watch(recommendedHandText, (nextValue) => {
  editableRecommendedHand.value = nextValue;
}, { immediate: true });

watch(scoringTiles, (tiles, previousTiles) => {
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

watch(doraIndicatorsInput, (value) => {
  scoreContext.doraIndicators = parseIndicatorList(value);
}, { immediate: true });

watch(uraIndicatorsInput, (value) => {
  scoreContext.uraIndicators = parseIndicatorList(value);
}, { immediate: true });

function applyInferenceOutcome(outcome: InferenceOutcome) {
  uploadInferenceAttempted.value = true;
  uploadDetections.value = outcome.detections;
  statusMessage.value =
    outcome.backend === 'webgpu'
      ? `上传图片推理已在 Web Worker 中通过 WebGPU 完成，当前模型为 ${activeModelLabel.value}。`
      : `上传图片推理当前来自 Web Worker 内的 WASM 回退路径，当前模型为 ${activeModelLabel.value}。`;
  scoreMessage.value = `识别完成：${outcome.detections.length} 个框；画布与原始检测明细展示全部后处理检测，最可能手牌推荐默认已写入可编辑计分输入。`;
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

async function onModelChange(nextModelId: string) {
  if (!nextModelId || nextModelId === modelAssets.value?.manifest.activeModel.id) {
    return;
  }

  const previousModelId = modelAssets.value?.manifest.activeModel.id ?? selectedModelId.value;
  statusMessage.value = '正在切换模型...';

  const loaded = await loadAssetsMetadata(nextModelId);
  if (!loaded) {
    selectedModelId.value = previousModelId;
    statusMessage.value = modelAssets.value
      ? `模型切换失败，当前仍使用 ${modelAssets.value.manifest.activeModel.label}。`
      : '模型切换失败，请稍后重试。';
    return;
  }

  resetModelSession();
  uploadInferenceAttempted.value = false;
  uploadDetections.value = [];
  editableRecommendedHand.value = '';
  inferenceError.value = null;
  scoreMessage.value = '';
  if (modelAssets.value) {
    statusMessage.value = `已切换到 ${modelAssets.value.manifest.activeModel.label}，请重新运行推理。`;
  }
}

function applySelectedFile(file: File, fileName: string) {
  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }

  imageUrl.value = URL.createObjectURL(file);
  selectedFile.value = file;
  selectedFileName.value = fileName;
  selectedImageName.value = fileName;
  uploadInferenceAttempted.value = false;
  uploadDetections.value = [];
  editableRecommendedHand.value = '';
  inferenceError.value = null;
  outputShape.value = [];
  timingMetrics.value = null;
  scoreMessage.value = '';
}

function onFileChange(uploadFile: UploadFile) {
  const file = uploadFile.raw;
  if (!file) {
    return;
  }

  applySelectedFile(file, uploadFile.name);
}

const onFileExceed: UploadProps['onExceed'] = (files) => {
  const replacementFile = files[0] as UploadRawFile | undefined;
  if (!replacementFile) {
    return;
  }

  uploadRef.value?.clearFiles();
  replacementFile.uid = genFileId();
  uploadRef.value?.handleStart(replacementFile);
  applySelectedFile(replacementFile, replacementFile.name);
};

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
    statusMessage.value = '推理失败，请稍后重试。';
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
  uploadRef.value?.clearFiles();
  editableRecommendedHand.value = '';
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
  <div style="height: 100vh;">
    <el-splitter style="height: 100%;">
      <el-splitter-panel size="68%" min="420">
        <el-scrollbar height="100%">
          <el-space direction="vertical" fill size="large">
            <el-space wrap>
              <el-select
                v-if="modelOptions.length > 0"
                v-model="selectedModelId"
                placeholder="选择模型"
                :disabled="loadingAssets || preparingModel || runningInference"
                @change="onModelChange"
              >
                <el-option v-for="model in modelOptions" :key="model.id" :label="model.label" :value="model.id" />
              </el-select>
              <el-button type="primary" :disabled="!canRunInference" @click="runInference">
                {{ runningInference ? '推理中...' : preparingModel ? '初始化中...' : '运行 YOLO 推理' }}
              </el-button>
              <el-button :disabled="!canResetUpload" @click="resetUpload">清空</el-button>
              <el-tag v-if="selectedFileName" type="info" effect="plain">{{ selectedFileName }}</el-tag>
            </el-space>

            <el-upload
              ref="uploadRef"
              drag
              :auto-upload="false"
              :show-file-list="false"
              :limit="1"
              accept="image/*"
              :on-change="onFileChange"
              :on-exceed="onFileExceed"
            >
              <div>拖拽图片到这里，或点击选择</div>
              <div>支持单张麻将手牌图片；上传后可直接预览并运行推理。</div>
            </el-upload>

            <DetectionCanvas :image-url="imageUrl" :detections="uploadDetections" />
          </el-space>
        </el-scrollbar>
      </el-splitter-panel>

      <el-splitter-panel size="32%" min="320">
        <el-scrollbar height="100%">
          <el-space direction="vertical" fill size="large">
            <el-card header="状态与结果">
              <el-alert v-if="uploadEmptyMessage" :title="uploadEmptyMessage" type="info" :closable="false" show-icon />
              <el-alert v-if="assetError" :title="`初始化失败：${assetError}`" type="warning" :closable="false" show-icon />
              <el-alert v-if="inferenceError" :title="`推理失败：${inferenceError}`" type="warning" :closable="false" show-icon />
              <el-alert v-if="scoreMessage" :title="scoreMessage" type="success" :closable="false" show-icon />

              <el-descriptions :column="2" border>
                <el-descriptions-item label="模型准备">{{ modelPreparationLabel }}</el-descriptions-item>
                <el-descriptions-item label="执行路径">{{ executionPathLabel }}</el-descriptions-item>
                <el-descriptions-item label="WebGPU">{{ webGpuCapabilityLabel }}</el-descriptions-item>
                <el-descriptions-item label="缓存">{{ cacheStatus?.enabled ? 'SW 已启用' : '未启用' }}</el-descriptions-item>
              </el-descriptions>

              <el-collapse>
                <el-collapse-item title="诊断详情" name="diagnostics">
                  <el-alert :title="modelLifecycleMessage" type="info" :closable="false" show-icon />
                  <el-alert :title="cacheStatusMessage" type="info" :closable="false" show-icon />

                  <el-button :disabled="clearingCache || loadingAssets || preparingModel || runningInference" @click="clearAssetCache">
                    {{ clearingCache ? '清理中...' : '清理静态缓存' }}
                  </el-button>

                  <el-alert v-if="browserBlockingMessage" :title="`浏览器不满足推理要求：${browserBlockingMessage}`" type="warning" :closable="false" show-icon />
                  <el-alert v-else-if="compatibilityNotices.length > 0" type="info" :closable="false" show-icon>
                    <template #title>兼容性提示</template>
                    <ul>
                      <li v-for="notice in compatibilityNotices" :key="notice">{{ notice }}</li>
                    </ul>
                  </el-alert>

                  <el-alert v-if="backendFallbackReason" type="warning" :closable="false" show-icon>
                    <template #title>
                      <span v-if="isGpuAdapterUnavailableReason(backendFallbackReason)">
                        WebGPU API 可用但无 GPU adapter，已回退 WASM：{{ backendFallbackReason }}
                      </span>
                      <span v-else>WebGPU 未能启用，已回退 WASM：{{ backendFallbackReason }}</span>
                    </template>
                  </el-alert>

                  <el-descriptions v-if="modelInfo" :column="2" border>
                    <el-descriptions-item label="模型">{{ modelInfo.model }}</el-descriptions-item>
                    <el-descriptions-item label="文件">{{ modelInfo.modelFile }}</el-descriptions-item>
                    <el-descriptions-item label="输入">{{ modelInfo.inputSize }}</el-descriptions-item>
                    <el-descriptions-item label="类别">{{ modelInfo.classCount }}</el-descriptions-item>
                    <el-descriptions-item label="阈值">conf={{ modelInfo.confidence }}, iou={{ modelInfo.iou }}</el-descriptions-item>
                    <el-descriptions-item label="后端">{{ backendUsed ?? '未运行' }}</el-descriptions-item>
                    <el-descriptions-item label="线程">Web Worker</el-descriptions-item>
                    <el-descriptions-item label="输出形状">{{ outputShape.length > 0 ? JSON.stringify(outputShape) : '未返回' }}</el-descriptions-item>
                  </el-descriptions>

                  <el-descriptions v-if="timingMetrics" :column="2" border>
                    <el-descriptions-item label="预处理">{{ formatTiming(timingMetrics.preprocessMs) }}</el-descriptions-item>
                    <el-descriptions-item label="推理">{{ formatTiming(timingMetrics.inferenceMs) }}</el-descriptions-item>
                    <el-descriptions-item label="后处理">{{ formatTiming(timingMetrics.postprocessMs) }}</el-descriptions-item>
                    <el-descriptions-item label="总耗时">{{ formatTiming(timingMetrics.totalMs) }}</el-descriptions-item>
                  </el-descriptions>
                </el-collapse-item>
              </el-collapse>
            </el-card>

            <el-card header="手牌推荐">
              <el-alert
                v-if="baselineComparison"
                :title="baselineComparison.exactMatch
                  ? `基线比对通过：${baselineComparison.sample.imageName} 与内置基线一致。`
                  : `基线未通过：当前 ${baselineComparison.actualCount} 张，基线 ${baselineComparison.expectedCount} 张，差异 ${baselineComparison.mismatches.length} 处。`"
                :type="baselineComparison.exactMatch ? 'success' : 'warning'"
                :closable="false"
                show-icon
              />

              <el-empty v-if="recommendedTiles.length === 0" description="当前还没有可推荐的手牌序列。" />
              <template v-else>
                <el-space wrap>
                  <el-tag
                    v-for="(tile, index) in recommendedTiles"
                    :key="`recommended-${tile}-${index}`"
                    :type="isMahjongTileCode(tile) ? 'info' : 'danger'"
                    effect="plain"
                  >
                    {{ tile }}
                  </el-tag>
                </el-space>
                <el-text type="info" size="small">赤宝牌：{{ recommendedRedDoraCount }}</el-text>
              </template>

              <el-input
                v-model="editableRecommendedHand"
                type="textarea"
                :rows="3"
                placeholder="用于计分的手牌（默认取自推荐，可编辑；空格/逗号/换行分隔）"
              />

              <el-space v-if="scoringTiles.length > 0" wrap>
                <el-tag
                  v-for="(tile, index) in scoringTiles"
                  :key="`scoring-${tile}-${index}`"
                  :type="!isMahjongTileCode(tile) ? 'danger' : index === scoreContext.agariIndex ? 'warning' : 'info'"
                  effect="plain"
                >
                  {{ tile }}
                </el-tag>
              </el-space>

              <el-text type="info" size="small">计分输入 {{ scoringTiles.length }} 张；赤宝牌：{{ scoringRedDoraCount }}</el-text>
              <el-text v-if="baselineComparison && baselineComparison.mismatches.length > 0" type="warning" size="small">
                差异：{{ baselineComparison.mismatches.slice(0, 5).map((item) => `#${item.index}: ${item.expected ?? '∅'} → ${item.actual ?? '∅'}`).join(' / ') }}
              </el-text>
            </el-card>

            <el-card header="和牌上下文">
              <el-form label-position="top">
                <el-row :gutter="16">
                  <el-col :span="12">
                    <el-form-item label="场风">
                      <el-radio-group v-model="scoreContext.fieldWind">
                        <el-radio-button v-for="option in windOptions" :key="`field-${option.value}`" :value="option.value">{{ option.label }}</el-radio-button>
                      </el-radio-group>
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="自风">
                      <el-radio-group v-model="scoreContext.seatWind">
                        <el-radio-button v-for="option in windOptions" :key="`seat-${option.value}`" :value="option.value">{{ option.label }}</el-radio-button>
                      </el-radio-group>
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-row :gutter="16">
                  <el-col :span="12">
                    <el-form-item label="和牌方式">
                      <el-radio-group v-model="scoreContext.agariType">
                        <el-radio-button v-for="option in agariTypeOptions" :key="option.value" :value="option.value">{{ option.label }}</el-radio-button>
                      </el-radio-group>
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="和牌索引">
                      <el-input-number v-model="scoreContext.agariIndex" :min="0" :max="Math.max(scoringTiles.length - 1, 0)" placeholder="0-based" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-row :gutter="16">
                  <el-col :span="12">
                    <el-form-item label="宝牌指示牌">
                      <el-input v-model="doraIndicatorsInput" placeholder="逗号分隔，如 3m,7p" clearable />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="里宝牌指示牌">
                      <el-input v-model="uraIndicatorsInput" placeholder="逗号分隔" clearable />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-form-item label="额外状态">
                  <el-checkbox-button v-model="scoreContext.riichi">立直</el-checkbox-button>
                  <el-checkbox-button v-model="scoreContext.ippatsu">一发</el-checkbox-button>
                </el-form-item>
              </el-form>

              <el-alert :title="readinessMessage" :type="scoring.status === 'incomplete' ? 'warning' : 'success'" :closable="false" show-icon />
            </el-card>

            <el-card header="计算结果">
              <template v-if="scoring.status === 'ready' && scoring.result">
                <el-descriptions :column="2" border>
                  <el-descriptions-item label="番数">{{ scoring.result.han }}</el-descriptions-item>
                  <el-descriptions-item label="符数">{{ scoring.result.fu }}</el-descriptions-item>
                  <el-descriptions-item label="点数 1">{{ scoring.result.point1 }}</el-descriptions-item>
                  <el-descriptions-item label="点数 2">{{ scoring.result.point2 }}</el-descriptions-item>
                </el-descriptions>

                <el-divider content-position="left">役种</el-divider>
                <el-space v-if="scoring.result.yaku.length > 0" wrap>
                  <el-tag v-for="(yaku, index) in scoring.result.yaku" :key="`${yaku}-${index}`" effect="plain">{{ yaku }}</el-tag>
                </el-space>
                <el-empty v-else description="没有役种输出。" />

                <el-alert
                  v-if="scoring.result.fuMessages.length > 0"
                  :title="`符说明：${scoring.result.fuMessages.join(' / ')}`"
                  type="info"
                  :closable="false"
                  show-icon
                />
                <el-alert
                  v-if="scoring.warnings.length > 0"
                  :title="scoring.warnings.join('；')"
                  type="warning"
                  :closable="false"
                  show-icon
                />
              </template>

              <el-alert v-else :title="scoring.message" type="warning" :closable="false" show-icon />

              <el-alert
                title="限制：不自动识别副露、宝牌指示牌、里宝牌、立直或局况，需人工补录。"
                type="info"
                :closable="false"
                show-icon
              />
            </el-card>

            <el-collapse>
              <el-collapse-item title="原始检测明细" name="raw-detections">
                <el-empty v-if="rawDetectionPayload.length === 0" description="运行推理后，这里会列出每个检测框的分类、置信度、坐标和中心点信息。" />
                <el-input v-else :model-value="rawDetectionText" type="textarea" :rows="10" readonly />
              </el-collapse-item>
            </el-collapse>
          </el-space>
        </el-scrollbar>
      </el-splitter-panel>
    </el-splitter>
  </div>
</template>
