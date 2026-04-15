# Mahjong-YOLO Web App 重构进度

## 概述

将项目从完整的训练+推理仓库改造为纯前端浏览器推理项目。UI 从自定义 CSS 全面迁移到 Element Plus 原生组件。

## 已完成工作

### Round 1：Element Plus 全面迁移

- 安装 `element-plus` 依赖
- 删除 `src/styles.css` 全局自定义样式文件
- `App.vue` 全部改用 Element Plus 原生组件（`el-card`、`el-form`、`el-button`、`el-select`、`el-descriptions`、`el-alert`、`el-tag`、`el-space`、`el-empty`、`el-divider` 等）
- 清空 `vendor/mahjong-vue` 遗留 UI 目录下的自定义 CSS 视图文件（`App.vue`、`CalculatorView.vue`、`PracticeView.vue`、`HomeView.vue`、`BlockSelect.vue`、`PaiSelect.vue`、`HelloWorld.vue`、`main.ts`、`router/index.ts`）
- 保留 `vendor/src/store/*` 逻辑层（供 `src/lib/mahjong.ts` 使用）
- 验证：npm test 29/29 通过、npm run build 成功、tsc --noEmit 无错误、浏览器截图与交互验证通过

### Round 2：响应式布局

- 改用 `el-row` / `el-col` 带 `:xs/:sm/:md/:lg` 响应式断点
- "图片上传与推理"操作区独立为卡片，放到"检测结果"上方
- "原始检测明细"下沉为全宽卡片
- 验证：在 390px / 768px / 1280px 三个宽度下浏览器验证布局正确

### Round 3：Splitter 布局 + 拖拽上传 + MessageBox 预览

- 主布局改为 `el-splitter` 水平分割（左面板 = 操作 + 检测结果，右面板 = 计分）
- 删除顶部 3 段说明文案及其 `el-card` 容器
- 检测区集成 `el-upload` 拖拽上传（`drag` 模式，`limit=1`，支持替换）
- 画布自适应大小，点击用 `ElMessageBox` 弹大图预览
- 压平 HTML 嵌套，移除不必要层级
- 验证：测试 29/29 通过、构建成功、浏览器 QA 通过

### Round 4：纯前端项目改造

- 删除所有训练相关文件：
  - Python 脚本（check_classes.py、create_labeled_demo.py、generate_inference_examples.py、inference_validation.py、validate_models.py）
  - 训练 notebooks（notebooks/）
  - PyTorch 模型权重（trained_models_v2/、models/）
  - 模型转换脚本（scripts/）
  - 推理示例图片（inference_examples/、inference_validation/）
  - 测试照片（mahjong_photos/）
  - 训练结果数据（model_validation_results.csv）
- 将 `web_app/` 内容提升到项目根目录
- 清理遗留代码：
  - 删除 LiveDetectionCanvas.vue、live-camera.ts 及其测试
  - 移除 vue-router 依赖
  - 清理 vendor/mahjong-vue 中未使用的文件（仅保留 src/store/）
- 重写 README.md 为纯前端推理项目说明
- 更新 .gitignore 为纯前端项目配置
- 验证：测试 25/25 通过、构建成功

## 当前技术栈

- **框架**: Vue 3.5 + TypeScript 5.7
- **构建**: Vite 6
- **UI 组件库**: Element Plus 2.13（零自定义 CSS）
- **推理引擎**: ONNX Runtime Web 1.22（WebGPU 优先，WASM 回退）
- **测试**: Vitest（25 个测试用例）

## 当前功能

- 上传图片推理（拖拽或点击上传）
- 多 ONNX 模型切换（Nano / Large）
- WebGPU / WASM 双路径自动回退
- Web Worker 后台推理
- 检测框画布展示 + 点击大图预览
- Python 基线比对
- 手牌排序与日麻计分
- Service Worker 静态资源缓存
- 按需初始化模型会话（首屏不下载 ONNX）

## 项目结构

```
├── src/
│   ├── App.vue                    # 主页面（el-splitter 布局）
│   ├── main.ts                    # 入口，注册 Element Plus
│   ├── components/
│   │   └── DetectionCanvas.vue    # 检测框画布组件
│   ├── workers/
│   │   └── yolo.worker.ts         # ONNX 推理 Web Worker
│   └── lib/                       # 推理/计分/工具库
├── public/model/                  # ONNX 模型 + 元数据
├── vendor/mahjong-vue/src/store/  # 日麻计分逻辑（vendored）
└── package.json
```

## 下一步计划

- 项目已完成纯前端改造，可直接部署到静态托管平台
