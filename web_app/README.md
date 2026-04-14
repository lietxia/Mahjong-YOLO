# web_app 使用与部署说明

这个目录是 Mahjong YOLO 的浏览器端推理应用。Phase 5 保持了 Phase 4 的全部能力：

- 上传图片推理
- 摄像头实时检测
- Python 基线比对
- 手牌排序与麻将计分
- Web Worker 后台推理

同时补上了产品化交付所需的几项能力：

- **按需初始化模型会话**：首屏只读取 `model_manifest.json`、类别和基线数据，不会在挂载时立刻初始化大模型会话。
- **WebGPU / WASM 回退说明**：浏览器支持 WebGPU 时优先使用 WebGPU；否则自动回退到 WASM，并在界面里显示当前执行路径和限制说明。
- **最小静态缓存策略**：生产构建会注册一个小型 Service Worker，缓存应用壳、模型清单和已经访问过的静态模型资源。
- **更完整的状态反馈**：页面会解释空状态、初始化中、回退中、失败场景与摄像头兼容性。

## 1. 本地运行

```bash
cd web_app
npm install
npm run dev
```

默认开发地址：`http://localhost:5173`

开发模式下不会注册 Service Worker，方便你调试资源和缓存。

## 2. 测试与构建

```bash
cd web_app
npm test
npm run build
```

构建产物输出到 `web_app/dist/`。

## 3. 部署方式

这是一个纯静态前端应用，可直接部署到任何能托管静态文件的平台，例如：

- Nginx
- GitHub Pages / Cloudflare Pages / Netlify / Vercel 静态站点
- 任何可提供 HTTPS 的对象存储静态网站

### 当前部署前提

当前代码默认按 **站点根路径** 提供资源，例如：

- `/model/model_manifest.json`
- `/model/classes.json`
- `/model/python-baselines.json`
- `/model/mahjong-yolon-best.onnx`
- `/sw.js`

因此最简单、最稳妥的方式是把 `dist/` 部署到域名根目录。

### 推荐部署检查项

1. 确认 `dist/model/` 中的 JSON 和 ONNX 文件都能被静态服务器直接访问。
2. 为生产环境启用 **HTTPS**。
3. 如果需要摄像头功能，必须使用 **HTTPS 或 localhost**。
4. 不要让静态服务器拦截或重写 `model/*.json` 和 `model/*.onnx` 请求。

## 4. 浏览器兼容性与回退

### 推理基本要求

页面依赖以下浏览器能力：

- Web Worker
- fetch
- createImageBitmap
- WebAssembly

如果这些能力缺失，页面会在 UI 中明确提示当前环境不支持推理。

### WebGPU / WASM 行为

- **优先路径**：`Web Worker + WebGPU`
- **回退路径**：`Web Worker + WASM`

当浏览器没有 WebGPU、或 WebGPU 会话初始化失败时，应用会自动回退到 WASM。回退后：

- 图片上传推理仍可继续使用
- 摄像头实时检测仍可继续使用
- 实时采样频率会自动降低，避免堆帧
- 页面会显示当前是回退路径，而不是静默降级

### 摄像头要求

摄像头功能除了 `getUserMedia` 之外，还要求当前页面处于：

- `https://...`
- 或 `http://localhost`

如果浏览器支持摄像头 API 但当前不是安全上下文，页面会提示你切到 HTTPS/localhost，而不是只显示模糊错误。

## 5. 静态缓存策略

生产模式下，应用会注册 `public/sw.js` 生成的 Service Worker。当前策略保持极简：

- 预缓存应用壳与模型元数据
- 对访问过的 `assets/` 与 `model/` 资源做缓存
- 导航请求优先走网络，失败时回退缓存
- 静态资源使用 `stale-while-revalidate` 风格，兼顾复用与更新

这不是完整离线模式，也不会引入额外路由或后台逻辑；它只是给当前 Vite 静态应用提供一个可维护的缓存落点。

### 清理缓存

Phase 5 页面里提供了“**清理静态缓存**”按钮，用于：

- 删除当前应用缓存
- 注销当前 Service Worker

清理后请刷新页面，让浏览器重新注册新的缓存策略。

## 6. 维护建议

如果后续更新模型或元数据，建议一起检查这几项：

1. `public/model/model_manifest.json` 是否仍与实际文件名一致。
2. `public/model/classes.json` 和前端类别顺序是否一致。
3. `public/model/python-baselines.json` 是否仍对应当前前端基线图片命名。
4. 构建后 `dist/model/mahjong-yolon-best.onnx` 是否被正确复制。
5. 生产站点是否仍允许浏览器缓存 `model/` 与 `assets/` 资源。
