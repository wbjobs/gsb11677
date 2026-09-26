# WebNN 神经网络推理与性能对照 Demo

在浏览器中构建 **全连接 (Gemm) + 卷积 (Conv2D) + 池化 (MaxPool2D)** 网络，
使用 **WebNN / WebGPU / 纯 JS** 三种后端做推理与性能对照，推理运行在 **Web Worker** 中，
主线程通过 **PerformanceObserver** 监测长任务，结果用 **Canvas** 可视化。

## 网络结构

```
Input [1,1,28,28]
  → Conv2D(1→4, 3×3, pad=1) + ReLU   → [1,4,28,28]
  → MaxPool2D(2×2, stride=2)         → [1,4,14,14]
  → Flatten                          → [1,784]
  → Gemm(784→10) + Softmax           → [1,10]
```

权重与输入由固定种子的 PRNG 生成，三个后端计算结果可逐元素比对（误差 < 1e-3 判定一致）。

## 运行

```bash
cd B
python3 -m http.server 8000
# 打开 http://localhost:8000
```

WebNN 需要 Chromium 系浏览器（Edge / Chrome 新版本，部分平台需开启
`chrome://flags#webnn-api`）。不支持时会自动降级，不影响使用。

## 验收标准对照

| 验收项 | 实现 |
| --- | --- |
| WebNN 可用时推理正确 | `js/backends/webnn-backend.js` 构图推理，结果与 JS 参考比对 |
| 性能对照可量化 | Worker 内逐次计时，输出 avg/min/p95，Canvas 柱状图 |
| 不支持时降级且结果一致 | 降级链 WebNN→WebGPU→JS，一致性自动校验 |
| 算子不支持有提示 | 逐算子 try/catch 定位 + "探测 WebNN 算子支持" 按钮 |
| 形状不匹配被捕获 | `assertInputShape` 前置校验 + "测试形状不匹配" 按钮 |
| 后端不可用有提示 | `createContext`/`requestAdapter` 失败分类提示并降级 |
| 推理失败可重试 | 每后端最多 2 次重试（重建上下文），勾选框可注入模拟失败 |
| 主线程不卡 | 推理在 Worker；FPS 计数 + longtask 观察 + 动画指示 |
| 可视化准确 | 输入热力图 / 4 通道池化特征图 / 输出概率 / 性能柱状图 |

## 文件结构

```
index.html              页面与控制项
css/style.css           样式
js/main.js              主线程编排、PerformanceObserver、FPS
js/worker.js            Worker：调度、重试、降级链、计时
js/model.js             网络结构、确定性权重/输入、错误类型
js/backends/webnn-backend.js   WebNN 后端 + 算子探测
js/backends/webgpu-backend.js  WebGPU 后端 (WGSL compute shader)
js/backends/js-backend.js      纯 JS 参考实现
js/viz.js               Canvas 可视化
```
