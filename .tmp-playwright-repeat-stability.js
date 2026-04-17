async (page) => {
  const baseUrl = 'http://127.0.0.1:5173';
  const imagePath = '/Users/lietxia/Documents/code/Mahjong-YOLO/test.jpg';

  const baseManifest = {
    defaultModelId: 'mahjong-yolon-best',
    models: [
      {
        id: 'mahjong-yolon-best',
        label: 'Nano 模型（mahjong-yolon-best）',
        modelFile: 'https://pub-e3b5792ae4f24700b2b0f0a495d5256b.r2.dev/mahjong-yolon-best.onnx',
        notes: ['轻量 ONNX 模型，初始化快。'],
      },
      {
        id: 'mahjong-yolol-best',
        label: 'Large 模型（mahjong-yolol-best）',
        modelFile: 'https://pub-e3b5792ae4f24700b2b0f0a495d5256b.r2.dev/mahjong-yolol-best.onnx',
        notes: ['更大的 ONNX 模型，通常精度更高但初始化更慢。'],
      },
    ],
    classesFile: 'classes.json',
    baselinesFile: 'python-baselines.json',
    inputSize: 640,
    confidenceThreshold: 0.3,
    iouThreshold: 0.3,
    sahiEnabled: true,
    sahiSliceSize: 320,
    sahiOverlapRatio: 0.2,
    sahiIncludeFullImage: true,
    sizeFilterEnabled: true,
    sizeRatioThreshold: 0.5,
    ordering: 'x-center-ascending',
    classesSource: 'notebooks/data.yaml#names (37 classes)',
    baselineSource: 'scripts/export_web_python_references.py -> models/nano/mahjong-yolon-best.onnx (deterministic web-route reference)',
    notes: [
      'Phase 5 assumes a single-row hand image and sorts detections left-to-right.',
      'Scoring only supports recognized closed hands plus minimal manual context.',
      'Frontend class mapping is frozen here to avoid reusing inconsistent old repo metadata.',
    ],
  };

  const experiments = [
    { name: 'baseline', patch: {} },
    { name: 'iou_0_4', patch: { iouThreshold: 0.4 } },
    { name: 'iou_0_5', patch: { iouThreshold: 0.5 } },
  ];

  const allResults = [];

  for (const exp of experiments) {
    for (let run = 1; run <= 3; run += 1) {
      const manifest = JSON.parse(JSON.stringify(baseManifest));
      Object.assign(manifest, exp.patch);

      await page.unroute('**/model/model_manifest.json').catch(() => {});
      await page.route('**/model/model_manifest.json', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(manifest),
        });
      });

      await page.goto(baseUrl);
      await page.locator('input[type=file]').setInputFiles([imagePath]);
      await page.getByRole('button', { name: '运行 YOLO 推理' }).click();
      await page.waitForFunction(
        () => document.body.innerText.includes('识别完成：') || document.body.innerText.includes('推理失败：'),
        null,
        { timeout: 120000 },
      );
      await page.waitForTimeout(1000);

      await page.getByRole('button', { name: '原始检测明细' }).click();
      await page.waitForTimeout(300);

      const textareaValues = await page.locator('textarea').evaluateAll((nodes) => nodes.map((node) => node.value));
      const rawText = textareaValues.find((value) => typeof value === 'string' && value.trim().startsWith('[')) || '[]';
      const recommendedText = textareaValues.find((value) => typeof value === 'string' && !value.trim().startsWith('[')) || '';

      let detections = [];
      try {
        detections = JSON.parse(rawText);
      } catch {
        detections = [];
      }

      allResults.push({
        experiment: exp.name,
        run,
        detectionCount: detections.length,
        recommendedText: recommendedText.trim(),
        rawText,
      });
    }
  }

  return allResults;
}
