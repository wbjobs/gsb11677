# WebNN 神经网络推理演示

围绕 WebNN 的端到端推理 Demo：构建 **全连接（GEMM）/ 卷积（conv2d）/ 池化（maxPool2d、averagePool2d）**
的小型 CNN，在 **Web Worker** 中执行推理，并与 **纯 JS**、**WebGPU** 做性能对照。

## 运行

```bash
cd A
python3 -m http.server 8000
# 打开 http://localhost:8000 （推荐 Edge / Chrome 113+，WebNN 需浏览器支持）
```

## 网络结构

```
input [1,1,28,28]
  → conv2d 3x3 pad=1 (1→4) + relu   → [1,4,28,28]
  → maxPool2d 2x2/2                 → [1,4,14,14]
  → conv2d 3x3 pad=1 (4→8) + relu   → [1,8,14,14]
  → averagePool2d 2x2/2             → [1,8,7,7]
  → reshape + gemm 392→10           → [1,10]
```

权重与输入由固定种子的 PRNG 生成（`js/prng.js`），三个后端共享同一权重，结果可互相校验。

## 技术栈

- **WebNN**：`navigator.ml.createContext` + `MLGraphBuilder`（`js/backends/webnn-backend.js`）
- **WebGPU**：WGSL compute shader 实现 conv/pool/gemm（`js/backends/webgpu-backend.js`）
- **纯 JS**：CPU 参考实现，同时作为结果正确性的 ground truth（`js/backends/js-backend.js`）
- **Web Worker**：所有后端推理都在 Worker 中执行（`js/worker.js`），主线程只做渲染
- **PerformanceObserver**：Worker 内观测 `measure`（init/inference 耗时），主线程观测 `longtask`（验证主线程不卡）
- **Canvas**：输入、中间特征图、FC 输出柱状图、性能对比柱状图

## 验收标准对照

| 验收项 | 实现 |
| --- | --- |
| WebNN 可用时推理结果正确 | 每次推理后与纯 JS 参考比对（容差 1e-3），UI 显示 maxDiff |
| 性能对照可量化 | “性能对比”按钮：各后端 warmup 5 次 + N 次迭代，输出 avg/p50/min/max 并绘制柱状图 |
| 不支持时有降级且结果一致 | 降级链 WebNN → WebGPU → JS，降级后仍与 JS 参考比对一致性 |
| 算子不支持有提示 | “检测算子支持”探测 conv2d/gemm/relu/pool/gelu/lstm；构图时缺算子抛 `UnsupportedOpError` |
| 形状不匹配被捕获 | 所有后端入口 `assertShape` 校验，抛 `ShapeMismatchError`，不触发降级；“测试形状不匹配”按钮可复现 |
| 后端不可用有提示 | `navigator.ml` 缺失 / `createContext` 失败 / WebGPU adapter 为空 → `BackendUnavailableError`，日志+状态栏提示 |
| 推理失败可重试 | WebNN dispatch 失败指数退避重试 3 次；勾选“模拟推理失败”可演示重试耗尽后自动降级 |
| 主线程不卡 | 推理全部在 Worker；主线程 `longtask` PerformanceObserver 实时显示长任务次数 |
| 可视化准确 | 特征图直接取自各后端真实中间输出（WebNN 图多输出 / WebGPU readback / JS 中间张量） |

## 说明

- WebGPU 计时包含结果 readback（`mapAsync`）开销，属于端到端耗时。
- WebNN 描述符同时兼容新版 `shape` 与旧版 `dimensions` 写法。
- 在无 WebNN 的环境打开页面即自然演示“浏览器不支持 → 降级 WebGPU/JS”路径。
