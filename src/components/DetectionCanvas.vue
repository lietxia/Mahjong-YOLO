<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import type { RecognizedTile } from '../lib/tile';
import { createLabelPalette, getLabelColor } from '../lib/labels';

const props = defineProps<{
  imageUrl: string | null;
  detections: RecognizedTile[];
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);

const hasContent = computed(() => Boolean(props.imageUrl));
const palette = computed(() => createLabelPalette(props.detections.map((detection) => detection.label)));

function getCanvasColors() {
  return {
    placeholder: '#909399',
    labelText: '#ffffff',
  };
}

function draw() {
  const canvas = canvasRef.value;
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);

  if (!props.imageUrl) {
    context.fillStyle = getCanvasColors().placeholder;
    context.font = '16px sans-serif';
    context.fillText('请先上传一张麻将牌图片。', 24, 40);
    return;
  }

  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    context.drawImage(image, 0, 0);
    const { labelText } = getCanvasColors();

    props.detections.forEach((detection) => {
      const [x1, y1, x2, y2] = detection.bbox;
      const color = getLabelColor(detection.label, palette.value);
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const text = `${detection.label} ${detection.confidence.toFixed(2)}`;
      context.font = '14px sans-serif';
      const textWidth = context.measureText(text).width + 12;
      const textHeight = 22;
      const labelY = Math.max(0, y1 - textHeight);
      context.fillStyle = color;
      context.fillRect(x1, labelY, textWidth, textHeight);
      context.fillStyle = labelText;
      context.fillText(text, x1 + 6, labelY + 15);
    });
  };
  image.src = props.imageUrl;
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/png');
  });
}

async function openPreview() {
  const canvas = canvasRef.value;
  if (!props.imageUrl || !canvas || canvas.width === 0 || canvas.height === 0) {
    return;
  }

  const previewBlob = await canvasToBlob(canvas);
  if (!previewBlob) {
    return;
  }

  const previewUrl = URL.createObjectURL(previewBlob);
  const win = window.open('', '_blank');
  if (!win) {
    URL.revokeObjectURL(previewUrl);
    return;
  }

  win.document.write(`<!DOCTYPE html><html><head><title>检测结果预览</title><style>*{margin:0;padding:0}body{background:#000;display:flex;justify-content:center;align-items:center;min-height:100vh}img{max-width:100vw;max-height:100vh;object-fit:contain}</style></head><body><img src="${previewUrl}" alt="检测结果预览"></body></html>`);
  win.document.close();
}

onMounted(draw);
watch(() => [props.imageUrl, props.detections], draw, { deep: true, flush: 'post' });
</script>

<template>
  <el-empty
    v-if="!hasContent"
    description="拖入或选择一张麻将手牌图片后，这里会显示自适应预览与检测框。"
  />
  <el-scrollbar v-else>
    <canvas ref="canvasRef" @click="openPreview" style="display: block; width: 100%; height: auto; cursor: zoom-in;"></canvas>
  </el-scrollbar>
</template>
