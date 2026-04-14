export type ModelVariant = {
  id: string;
  label: string;
  modelFile: string;
  notes?: string[];
};

export type ModelManifest = {
  defaultModelId: string;
  models: ModelVariant[];
  classesFile: string;
  baselinesFile: string;
  inputSize: number;
  confidenceThreshold: number;
  iouThreshold: number;
  ordering: string;
  classesSource: string;
  baselineSource: string;
  notes: string[];
};

export type BaselineSample = {
  imageName: string;
  expectedOrderedTiles: string[];
  notes: string[];
  source?: string;
};

export type ResolvedModelManifest = ModelManifest & {
  activeModel: ModelVariant;
  modelFile: string;
};

export type ModelAssets = {
  manifest: ResolvedModelManifest;
  classes: string[];
  baselines: BaselineSample[];
};

async function loadJson<T>(path: string, failureMessage: string): Promise<T> {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(failureMessage);
  }

  return (await response.json()) as T;
}

export async function loadModelAssets(selectedModelId?: string): Promise<ModelAssets> {
  const manifest = await loadJson<ModelManifest>('/model/model_manifest.json', '无法读取 model_manifest.json');
  const activeModelId = selectedModelId ?? manifest.defaultModelId;
  const activeModel = manifest.models.find((model) => model.id === activeModelId);

  if (!activeModel) {
    throw new Error(`模型 ${activeModelId} 不存在于 model_manifest.json`);
  }

  const [classes, baselines] = await Promise.all([
    loadJson<string[]>(`/model/${manifest.classesFile}`, '无法读取 classes.json'),
    loadJson<BaselineSample[]>(`/model/${manifest.baselinesFile}`, `无法读取 ${manifest.baselinesFile}`),
  ]);

  return {
    manifest: {
      ...manifest,
      activeModel,
      modelFile: activeModel.modelFile,
    },
    classes,
    baselines,
  };
}
