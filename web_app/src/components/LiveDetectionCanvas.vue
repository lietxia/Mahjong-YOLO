<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import type { RecognizedTile } from '../lib/tile';
import { createLabelPalette, getLabelColor } from '../lib/labels';

const props = defineProps<{
  stream: MediaStream | null;
  detections: RecognizedTile[];
  placeholderMessage: string;
}>();

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const palette = computed(() => createLabelPalette(props.detections.map((detection) => detection.label)));

let animationFrameId = 0;

function getCanvasColors() {
  return {
    placeholder: '#909399',
    labelText: '#ffffff',
  };
}

function drawPlaceholder(message: string) {
  const canvas = canvasRef.value;
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  canvas.width = 960;
  canvas.height = 540;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = getCanvasColors().placeholder;
  context.font = '16px sans-serif';
  context.fillText(message, 24, 40);
}

function drawFrame() {
  const canvas = canvasRef.value;
  const video = videoRef.value;
  if (!canvas || !video) {
    animationFrameId = window.requestAnimationFrame(drawFrame);
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    animationFrameId = window.requestAnimationFrame(drawFrame);
    return;
  }

  if (!props.stream) {
    drawPlaceholder(props.placeholderMessage);
    animationFrameId = window.requestAnimationFrame(drawFrame);
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0 || video.videoHeight === 0) {
    drawPlaceholder(props.placeholderMessage);
    animationFrameId = window.requestAnimationFrame(drawFrame);
    return;
  }

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const { labelText } = getCanvasColors();

  props.detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.bbox;
    const color = getLabelColor(detection.label, palette.value);
    const text = `${detection.label} ${detection.confidence.toFixed(2)}`;

    context.strokeStyle = color;
    context.lineWidth = 2;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);

    context.font = '14px sans-serif';
    const textWidth = context.measureText(text).width + 12;
    const textHeight = 22;
    const labelY = Math.max(0, y1 - textHeight);
    context.fillStyle = color;
    context.fillRect(x1, labelY, textWidth, textHeight);
    context.fillStyle = labelText;
    context.fillText(text, x1 + 6, labelY + 15);
  });

  animationFrameId = window.requestAnimationFrame(drawFrame);
}

async function attachStream(stream: MediaStream | null) {
  const video = videoRef.value;
  if (!video) {
    return;
  }

  if (!stream) {
    video.pause();
    video.srcObject = null;
    return;
  }

  video.srcObject = stream;
  try {
    await video.play();
  } catch {
    // 浏览器自动播放策略可能暂时阻止播放，下一帧会继续尝试绘制。
  }
}

function hasReadyFrame(): boolean {
  const video = videoRef.value;
  return Boolean(video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0);
}

async function captureFrameBitmap(): Promise<ImageBitmap | null> {
  const video = videoRef.value;
  if (!video || !hasReadyFrame()) {
    return null;
  }

  return createImageBitmap(video);
}

function captureScreenshot(): string | null {
  const canvas = canvasRef.value;
  if (!canvas || canvas.width === 0 || canvas.height === 0) {
    return null;
  }

  return canvas.toDataURL('image/png');
}

defineExpose({
  captureFrameBitmap,
  captureScreenshot,
  hasReadyFrame,
});

onMounted(() => {
  void attachStream(props.stream);
  animationFrameId = window.requestAnimationFrame(drawFrame);
});

watch(() => props.stream, (stream) => {
  void attachStream(stream);
});

onUnmounted(() => {
  window.cancelAnimationFrame(animationFrameId);
});
</script>

<template>
  <el-card>
    <template #header>实时检测画面</template>
    <el-space direction="vertical" fill>
      <p v-if="!stream">{{ placeholderMessage }}</p>
      <video ref="videoRef" hidden autoplay muted playsinline></video>
      <el-scrollbar>
        <canvas ref="canvasRef"></canvas>
      </el-scrollbar>
    </el-space>
  </el-card>
</template>
