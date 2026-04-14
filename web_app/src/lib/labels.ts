import type { ModelAssets } from './manifest';

const SUIT_COLORS: Record<string, string> = {
  m: '#b91c1c',
  p: '#1d4ed8',
  s: '#15803d',
  z: '#7c3aed',
};

export function createLabelPalette(labels: string[]): Record<string, string> {
  return labels.reduce<Record<string, string>>((palette, label) => {
    const suit = label.slice(-1);
    palette[label] = SUIT_COLORS[suit] ?? '#374151';
    return palette;
  }, {});
}

export function getLabelColor(label: string, palette: Record<string, string>): string {
  return palette[label] ?? '#111827';
}

export function summarizeModelAssets(assets: ModelAssets) {
  return {
    model: assets.manifest.modelFile,
    inputSize: assets.manifest.inputSize,
    classCount: assets.classes.length,
    confidence: assets.manifest.confidenceThreshold,
    iou: assets.manifest.iouThreshold,
    classesSource: assets.manifest.classesSource,
    baselineSource: assets.manifest.baselineSource,
    notes: assets.manifest.notes,
  };
}
