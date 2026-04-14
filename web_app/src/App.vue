<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue';
import DetectionCanvas from './components/DetectionCanvas.vue';
import { calculateMahjongScore, type AgariType, type ScoreContext, type Wind } from './lib/mahjong';
import { loadModelAssets, type ModelAssets } from './lib/model';
import { countRedDora, sortTilesLeftToRight, toOrderedTileLabels, type RecognizedTile } from './lib/tile';
import { YoloBrowserRunner, type InferenceOutcome } from './lib/yolo';

const modelAssets = ref<ModelAssets | null>(null);
const modelRunner = ref<YoloBrowserRunner | null>(null);
const imageUrl = ref<string | null>(null);
const selectedFileName = ref('');
const loadingAssets = ref(true);
const runningInference = ref(false);
const assetError = ref<string | null>(null);
const inferenceError = ref<string | null>(null);
const statusMessage = ref<string>('');
const backendUsed = ref<'webgpu' | 'wasm' | null>(null);
const outputShape = ref<number[]>([]);
const detections = ref<RecognizedTile[]>([]);
const scoreMessage = ref('');

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

const orderedDetections = computed(() => sortTilesLeftToRight(detections.value));
const orderedTiles = computed(() => toOrderedTileLabels(orderedDetections.value));
const redDoraCount = computed(() => countRedDora(orderedTiles.value));

const scoring = computed(() => calculateMahjongScore(orderedTiles.value, scoreContext));

const readinessMessage = computed(() => {
  if (orderedTiles.value.length === 0) {
    return '还没有识别到可计算的牌。';
  }
  return scoring.value.message;
});

const modelInfo = computed(() => {
  if (!modelAssets.value) {
    return null;
  }
  return {
    model: modelAssets.value.manifest.modelFile,
    inputSize: modelAssets.value.manifest.inputSize,
    classCount: modelAssets.value.classes.length,
    confidence: modelAssets.value.manifest.confidenceThreshold,
    iou: modelAssets.value.manifest.iouThreshold,
  };
});

function parseIndicatorList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

async function bootstrap() {
  loadingAssets.value = true;
  assetError.value = null;
  try {
    const assets = await loadModelAssets();
    modelAssets.value = assets;
    modelRunner.value = new YoloBrowserRunner(assets);
    statusMessage.value = navigator.gpu
      ? '检测到浏览器支持 WebGPU，将优先尝试 GPU 推理。'
      : '当前浏览器看起来不支持 WebGPU，将回退到 WASM 推理。';
  } catch (error) {
    assetError.value = error instanceof Error ? error.message : '初始化模型资源失败';
  } finally {
    loadingAssets.value = false;
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
  selectedFileName.value = file.name;
  detections.value = [];
  inferenceError.value = null;
  backendUsed.value = null;
  outputShape.value = [];
  scoreMessage.value = '';
}

async function runInference() {
  if (!imageUrl.value || !modelRunner.value) {
    return;
  }

  runningInference.value = true;
  inferenceError.value = null;
  statusMessage.value = '正在浏览器中运行 YOLO 推理...';

  try {
    const image = await loadImage(imageUrl.value);
    const outcome: InferenceOutcome = await modelRunner.value.infer(image);
    detections.value = outcome.detections;
    backendUsed.value = outcome.backend;
    outputShape.value = outcome.rawShape;

    if (orderedTiles.value.length > 0) {
      scoreContext.agariIndex = Math.min(orderedTiles.value.length - 1, 13);
    }

    scoreMessage.value = `识别完成：${outcome.detections.length} 个框，已按从左到右提取 ${orderedTiles.value.length} 张牌。`;
    statusMessage.value =
      outcome.backend === 'webgpu'
        ? '推理已通过 WebGPU 完成。'
        : 'WebGPU 未能启用，当前结果来自 WASM 回退路径。';
  } catch (error) {
    inferenceError.value = error instanceof Error ? error.message : '推理失败';
  } finally {
    runningInference.value = false;
  }
}

function resetAll() {
  if (imageUrl.value) {
    URL.revokeObjectURL(imageUrl.value);
  }
  imageUrl.value = null;
  selectedFileName.value = '';
  detections.value = [];
  inferenceError.value = null;
  statusMessage.value = '已清空当前图片与识别结果。';
  backendUsed.value = null;
  outputShape.value = [];
  scoreMessage.value = '';
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

onMounted(bootstrap);

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('无法读取上传图片。'));
    image.src = source;
  });
}
</script>

<template>
  <div class="page">
    <header class="page-header">
      <h1>Mahjong YOLO Phase 1 Demo</h1>
      <p>
        这个阶段只做一条最小浏览器闭环：上传图片 → YOLO 识别麻将牌 → 左到右排序 →
        用最少手动上下文调用 <code>mahjong-vue</code> 的算番核心。
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
            <button :disabled="!imageUrl || runningInference || loadingAssets" @click="runInference">
              {{ runningInference ? '推理中...' : '运行 YOLO 推理' }}
            </button>
            <button class="secondary" :disabled="!imageUrl && detections.length === 0" @click="resetAll">
              清空
            </button>
          </div>

          <div v-if="assetError" class="message warning">模型资源初始化失败：{{ assetError }}</div>
          <div v-if="inferenceError" class="message warning">推理失败：{{ inferenceError }}</div>
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
              <span class="meta-label">输出形状</span>
              {{ outputShape.length > 0 ? JSON.stringify(outputShape) : '未返回' }}
            </div>
          </div>

          <div v-if="orderedTiles.length > 0" class="message">
            Phase 1 排序规则：单排手牌假设，按检测框中心点从左到右排序。
          </div>
        </div>

        <DetectionCanvas :image-url="imageUrl" :detections="orderedDetections" />

        <div class="panel">
          <h2>识别到的牌</h2>
          <p>
            当前共识别 {{ orderedTiles.length }} 张牌。Phase 1 只支持 <strong>14 张闭门手牌</strong>
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
            限制：Phase 1 不自动识别副露、宝牌指示牌、里宝牌、立直信息或桌面完整局况，需人工补录必要上下文。
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>
