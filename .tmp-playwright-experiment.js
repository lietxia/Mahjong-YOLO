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
    { name: 'sahi_off', patch: { sahiEnabled: false } },
    { name: 'iou_0_5', patch: { iouThreshold: 0.5 } },
    { name: 'sahi_512_overlap_035', patch: { sahiSliceSize: 512, sahiOverlapRatio: 0.35 } },
    { name: 'no_full_image', patch: { sahiIncludeFullImage: false } },
    { name: 'no_size_filter', patch: { sizeFilterEnabled: false } },
    { name: 'conf_0_2_iou_0_5', patch: { confidenceThreshold: 0.2, iouThreshold: 0.5 } },
    { name: 'class_aware_iou_0_5', patch: { iouThreshold: 0.5 }, classAwareNms: true },
  ];

  const context = page.context();
  const results = [];

  for (const exp of experiments) {
    const p = await context.newPage();
    const manifest = JSON.parse(JSON.stringify(baseManifest));
    Object.assign(manifest, exp.patch);

    await p.route('**/model/model_manifest.json', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(manifest),
      });
    });

    if (exp.classAwareNms) {
      await p.route('**/src/lib/postprocess.ts*', async (route) => {
        const response = await route.fetch();
        let body = await response.text();
        body = body.replace(
          'if (computeIou(sorted[index].bbox, current.bbox) > iouThreshold) {',
          'if (sorted[index].classId === current.classId && computeIou(sorted[index].bbox, current.bbox) > iouThreshold) {',
        );
        await route.fulfill({
          response,
          body,
          contentType: 'application/javascript',
        });
      });
    }

    await p.goto(baseUrl);
    await p.locator('input[type=file]').setInputFiles([imagePath]);
    await p.getByRole('button', { name: '运行 YOLO 推理' }).click();
    await p.waitForFunction(
      () => document.body.innerText.includes('识别完成：') || document.body.innerText.includes('推理失败：'),
      null,
      { timeout: 120000 },
    );
    await p.waitForTimeout(1000);

    const bodyText = await p.locator('body').innerText();
    const statusLine = bodyText.split('\n').find((line) => line.includes('识别完成：') || line.includes('推理失败：')) || '';

    await p.getByRole('button', { name: '原始检测明细' }).click();
    await p.waitForTimeout(300);

    const textareaValues = await p.locator('textarea').evaluateAll((nodes) => nodes.map((node) => node.value));
    const rawText = textareaValues.find((value) => typeof value === 'string' && value.trim().startsWith('[')) || '[]';
    const recommendedText = textareaValues.find((value) => typeof value === 'string' && !value.trim().startsWith('[')) || '';

    let detections = [];
    try {
      detections = JSON.parse(rawText);
    } catch {
      detections = [];
    }

    const normalized = detections.map((entry) => {
      const first = Object.entries(entry)[0] || ['UNKNOWN', [0, [0, 0, 0, 0], [0, 0]]];
      const label = first[0];
      const payload = first[1];
      return {
        label,
        confidence: payload[0],
        bbox: payload[1],
        center: payload[2],
      };
    });

    const upperCount = normalized.filter((item) => Array.isArray(item.center) && item.center[1] < 200).length;
    const lowerCount = normalized.filter((item) => Array.isArray(item.center) && item.center[1] >= 200).length;
    const lowConfidenceCount = normalized.filter((item) => typeof item.confidence === 'number' && item.confidence < 0.4).length;
    const labels = normalized.map((item) => item.label);

    results.push({
      name: exp.name,
      patch: exp.patch,
      classAwareNms: Boolean(exp.classAwareNms),
      statusLine,
      detectionCount: normalized.length,
      upperCount,
      lowerCount,
      lowConfidenceCount,
      recommendedCount: recommendedText.trim() ? recommendedText.trim().split(/\s+/).length : 0,
      recommendedText: recommendedText.trim(),
      labels,
    });

    await p.close();
  }

  return results;
}
