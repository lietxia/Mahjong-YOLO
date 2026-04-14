import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadModelAssets } from './manifest';

const mockManifest = {
  defaultModelId: 'default-model',
  models: [
    {
      id: 'default-model',
      label: 'Default',
      modelFile: 'default.onnx',
    },
    {
      id: 'large-model',
      label: 'Large',
      modelFile: 'large.onnx',
    },
  ],
  classesFile: 'classes.json',
  baselinesFile: 'python-baselines.json',
  inputSize: 640,
  confidenceThreshold: 0.3,
  iouThreshold: 0.5,
  ordering: 'x-center-ascending',
  classesSource: 'classes-source',
  baselineSource: 'baseline-source',
  notes: [],
};

function mockFetchResponses() {
  return vi.fn(async (input: string | URL | Request) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;

    if (url.endsWith('/model/model_manifest.json')) {
      return new Response(JSON.stringify(mockManifest), { status: 200 });
    }

    if (url.endsWith('/model/classes.json')) {
      return new Response(JSON.stringify(['1m', '2m']), { status: 200 });
    }

    if (url.endsWith('/model/python-baselines.json')) {
      return new Response(JSON.stringify([{ imageName: 'sample.png', expectedOrderedTiles: ['1m'], notes: [] }]), {
        status: 200,
      });
    }

    return new Response(null, { status: 404 });
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('model manifest loader', () => {
  it('resolves the default model when no explicit model id is provided', async () => {
    vi.stubGlobal('fetch', mockFetchResponses());

    const assets = await loadModelAssets();

    expect(assets.manifest.activeModel.id).toBe('default-model');
    expect(assets.manifest.modelFile).toBe('default.onnx');
    expect(assets.classes).toEqual(['1m', '2m']);
  });

  it('resolves the requested model when a model id is provided', async () => {
    vi.stubGlobal('fetch', mockFetchResponses());

    const assets = await loadModelAssets('large-model');

    expect(assets.manifest.activeModel.id).toBe('large-model');
    expect(assets.manifest.modelFile).toBe('large.onnx');
  });
});
