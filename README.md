# Mahjong-YOLO

麻将牌识别 Web 应用 — 纯前端浏览器推理，基于 ONNX Runtime Web。

## 功能

- **图片上传推理**：拖拽或点击上传麻将手牌图片，浏览器端实时检测
- **多模型切换**：支持 Nano（5MB，快速）和 Large（97MB，高精度）两个 ONNX 模型
- **WebGPU / WASM 双路径**：优先使用 WebGPU 加速，自动回退到 WASM
- **Web Worker 后台推理**：推理在 Worker 线程中执行，不阻塞 UI
- **检测框画布展示**：画布自适应大小，点击可大图预览
- **日麻计分**：基于检测结果自动推荐手牌，支持手动编辑后计算番数/符数/点数
- **Python 基线比对**：与内置 Python 推理基线结果对比
- **按需加载模型**：首屏只加载元数据，首次推理时才下载 ONNX 模型
- **Service Worker 缓存**：生产环境缓存应用壳和模型资源

## 技术栈

- Vue 3 + TypeScript
- Vite 6
- Element Plus（零自定义 CSS）
- ONNX Runtime Web（WebGPU + WASM）
- Vitest

## 快速开始

```bash
npm install
npm run dev
```

默认开发地址：`http://localhost:5173`

## 构建与测试

```bash
npm test        # 运行测试
npm run build   # 生产构建，输出到 dist/
npm run preview # 预览生产构建
```

## 项目结构

```
├── src/
│   ├── App.vue                    # 主页面（el-splitter 布局）
│   ├── main.ts                    # 入口，注册 Element Plus
│   ├── components/
│   │   └── DetectionCanvas.vue    # 检测框画布组件
│   ├── workers/
│   │   └── yolo.worker.ts         # ONNX 推理 Web Worker
│   └── lib/                       # 推理、计分、工具库
│       ├── yolo.ts                # Worker 通信封装
│       ├── preprocess.ts          # 图像预处理（letterbox + tensor）
│       ├── postprocess.ts         # 推理后处理（decode / NMS）
│       ├── mahjong.ts             # 日麻计分逻辑
│       ├── tile.ts                # 牌型解析与排序
│       ├── manifest.ts            # 模型清单加载
│       ├── baseline.ts            # Python 基线比对
│       ├── compatibility.ts       # 浏览器兼容性检测
│       └── cache.ts               # Service Worker 缓存管理
├── public/model/                  # ONNX 模型 + 元数据
│   ├── mahjong-yolon-best.onnx    # Nano 模型
│   ├── mahjong-yolol-best.onnx    # Large 模型
│   ├── model_manifest.json        # 模型清单
│   ├── classes.json               # 类别标签（38 类）
│   └── python-baselines.json      # Python 基线数据
├── vendor/mahjong-vue/src/store/  # 日麻计分逻辑（vendored）
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## 部署

纯静态前端应用，可部署到任何静态文件托管平台：

- GitHub Pages / Cloudflare Pages / Netlify / Vercel
- Nginx / 对象存储静态网站

构建产物在 `dist/` 目录，需部署到域名根路径。

### 部署检查项

1. `dist/model/` 中的 JSON 和 ONNX 文件可被直接访问
2. 启用 HTTPS（WebGPU 需要安全上下文）
3. 不要拦截或重写 `model/*.json` 和 `model/*.onnx` 请求

## 浏览器兼容性

### 推理基本要求

- Web Worker
- fetch
- createImageBitmap
- WebAssembly

### WebGPU / WASM 行为

- **优先路径**：Web Worker + WebGPU
- **回退路径**：Web Worker + WASM

浏览器没有 WebGPU 或 WebGPU 初始化失败时自动回退到 WASM，页面会显示当前执行路径。

## 麻将牌类别

模型识别 38 种麻将牌：

- **万子**：1m–9m, 0m（赤五万）
- **饼子**：1p–9p, 0p（赤五饼）
- **索子**：1s–9s, 0s（赤五索）
- **风牌**：1z（东）、2z（南）、3z（西）、4z（北）
- **三元牌**：5z（白）、6z（发）、7z（中）
- **特殊**：UNKNOWN（模糊/损坏牌）

## License

MIT
