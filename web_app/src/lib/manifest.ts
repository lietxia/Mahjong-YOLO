export type ModelManifest = {
  modelFile: string;
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

export type ModelAssets = {
  manifest: ModelManifest;
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

export async function loadModelAssets(): Promise<ModelAssets> {
  const manifest = await loadJson<ModelManifest>('/model/model_manifest.json', '无法读取 model_manifest.json');
  const [classes, baselines] = await Promise.all([
    loadJson<string[]>(`/model/${manifest.classesFile}`, '无法读取 classes.json'),
    loadJson<BaselineSample[]>(`/model/${manifest.baselinesFile}`, `无法读取 ${manifest.baselinesFile}`),
  ]);

  return { manifest, classes, baselines };
}
