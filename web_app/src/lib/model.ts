export type ModelManifest = {
  modelFile: string;
  classesFile: string;
  inputSize: number;
  confidenceThreshold: number;
  iouThreshold: number;
  ordering: string;
  notes: string[];
};

export type ModelAssets = {
  manifest: ModelManifest;
  classes: string[];
};

export async function loadModelAssets(): Promise<ModelAssets> {
  const [manifestResponse, classesResponse] = await Promise.all([
    fetch('/model/model_manifest.json'),
    fetch('/model/classes.json'),
  ]);

  if (!manifestResponse.ok) {
    throw new Error('无法读取 model_manifest.json');
  }

  if (!classesResponse.ok) {
    throw new Error('无法读取 classes.json');
  }

  const manifest = (await manifestResponse.json()) as ModelManifest;
  const classes = (await classesResponse.json()) as string[];
  return { manifest, classes };
}
