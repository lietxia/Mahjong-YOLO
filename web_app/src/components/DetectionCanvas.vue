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
  const styles = getComputedStyle(document.documentElement);
  return {
    placeholder: styles.getPropertyValue('--color-text-muted').trim() || '#6b7280',
    labelText: styles.getPropertyValue('--color-white').trim() || '#ffffff',
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

onMounted(draw);
watch(() => [props.imageUrl, props.detections], draw, { deep: true });
</script>

<template>
  <div class="panel">
    <h2>检测结果</h2>
    <p v-if="!hasContent">选择图片并触发推理后，会在这里叠加框、类别和置信度。</p>
    <div class="canvas-wrapper">
      <canvas ref="canvasRef"></canvas>
    </div>
  </div>
</template>
