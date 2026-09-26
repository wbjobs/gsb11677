import {
  generateWeights,
  generateInput,
  assertInputShape,
  INPUT_SHAPE,
  ShapeMismatchError,
  BackendUnavailableError,
  OpUnsupportedError,
} from './model.js';
import * as jsBackend from './backends/js-backend.js';
import { createWebGPUBackend } from './backends/webgpu-backend.js';
import { createWebNNBackend, probeOps } from './backends/webnn-backend.js';

const weights = generateWeights();
const input = generateInput();

const backendCache = new Map();
let injectFailuresLeft = 0;

// PerformanceObserver inside the worker: forwards performance.measure entries
// to the main thread for display.
const measureObserver = new PerformanceObserver((list) => {
  const entries = list.getEntries().map((e) => ({
    name: e.name,
    duration: e.duration,
  }));
  postMessage({ type: 'measures', entries });
});
measureObserver.observe({ entryTypes: ['measure'] });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function maybeInjectFailure(backend) {
  if (injectFailuresLeft > 0) {
    injectFailuresLeft--;
    throw new Error(`模拟推理失败 (backend=${backend}, 剩余注入次数=${injectFailuresLeft})`);
  }
}

async function getBackend(name, deviceType, forceRecreate = false) {
  if (name === 'js') {
    return {
      name: 'js',
      infer: (w, x) => Promise.resolve(jsBackend.infer(w, x)),
    };
  }
  const key = `${name}:${deviceType || ''}`;
  if (!forceRecreate && backendCache.has(key)) return backendCache.get(key);
  backendCache.delete(key);
  const backend =
    name === 'webnn'
      ? await createWebNNBackend(deviceType)
      : await createWebGPUBackend((info) => {
          backendCache.delete(key);
          postMessage({
            type: 'backend-lost',
            backend: 'webgpu',
            reason: `GPU device lost: ${info.reason || 'unknown'} ${info.message || ''}`,
          });
        });
  backendCache.set(key, backend);
  return backend;
}

function summarize(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
  return { avg, min: sorted[0], max: sorted[sorted.length - 1], p95, iterations: times.length };
}

async function runBackendWithRetry(name, deviceType, iterations, maxRetries = 2) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const backend = await getBackend(name, deviceType, attempt > 0);
      const times = [];
      let out = null;
      for (let i = 0; i < iterations; i++) {
        maybeInjectFailure(name);
        performance.mark(`${name}-start`);
        const t0 = performance.now();
        out = await backend.infer(weights, input);
        const dt = performance.now() - t0;
        performance.mark(`${name}-end`);
        performance.measure(`infer:${name}`, `${name}-start`, `${name}-end`);
        times.push(dt);
      }
      return { ...out, stats: summarize(times) };
    } catch (e) {
      lastError = e;
      backendCache.delete(`${name}:${deviceType || ''}`);
      // Only transient inference failures are worth retrying; capability
      // errors (unavailable backend / unsupported op / bad shape) fall
      // through to the next backend immediately.
      const retriable =
        !(e instanceof BackendUnavailableError) &&
        !(e instanceof OpUnsupportedError) &&
        !(e instanceof ShapeMismatchError);
      if (!retriable) break;
      if (attempt < maxRetries) {
        postMessage({
          type: 'retry',
          backend: name,
          attempt: attempt + 1,
          maxRetries,
          error: String((e && e.message) || e),
        });
        await sleep(120 * (attempt + 1));
      }
    }
  }
  throw lastError;
}

function fallbackChain(requested) {
  switch (requested) {
    case 'webnn':
      return ['webnn', 'webgpu', 'js'];
    case 'webgpu':
      return ['webgpu', 'js'];
    default:
      return ['webnn', 'webgpu', 'js']; // auto
  }
}

function classifyError(e) {
  if (e instanceof ShapeMismatchError) return '形状不匹配';
  if (e instanceof OpUnsupportedError) return `算子不支持 (${e.op})`;
  if (e instanceof BackendUnavailableError) return '后端不可用';
  return '推理失败';
}

async function handleRun(msg) {
  const { backend, deviceType, iterations, injectFailures } = msg;
  injectFailuresLeft = injectFailures || 0;

  // Shape validation happens before touching any backend.
  try {
    assertInputShape(input, INPUT_SHAPE);
  } catch (e) {
    postMessage({ type: 'run-error', kind: classifyError(e), error: e.message });
    return;
  }

  const chain = backend === 'js' ? ['js'] : fallbackChain(backend);
  let used = null;
  let result = null;
  for (const name of chain) {
    try {
      result = await runBackendWithRetry(name, deviceType, iterations);
      used = name;
      break;
    } catch (e) {
      postMessage({
        type: 'fallback',
        from: name,
        kind: classifyError(e),
        reason: String((e && e.message) || e),
      });
    }
  }
  if (!result) {
    postMessage({ type: 'run-error', kind: '推理失败', error: '所有后端均不可用' });
    return;
  }
  postMessage({
    type: 'result',
    backend: used,
    requested: backend,
    probs: result.probs,
    pooled: result.pooled,
    stats: result.stats,
  });
}

async function handleBenchmark(msg) {
  const { iterations, deviceType } = msg;
  const candidates = ['webnn', 'webgpu', 'js'];
  const results = [];
  let reference = null;
  for (const name of candidates) {
    postMessage({ type: 'benchmark-progress', backend: name, state: 'start' });
    try {
      const r = await runBackendWithRetry(name, deviceType, iterations);
      results.push({ backend: name, stats: r.stats, probs: r.probs, pooled: r.pooled });
      if (name === 'js') reference = r.probs;
      postMessage({ type: 'benchmark-progress', backend: name, state: 'done' });
    } catch (e) {
      results.push({
        backend: name,
        error: String((e && e.message) || e),
        kind: classifyError(e),
      });
      postMessage({
        type: 'benchmark-progress',
        backend: name,
        state: 'error',
        error: String((e && e.message) || e),
      });
    }
  }
  // Consistency check against the pure-JS reference.
  if (!reference) {
    const ref = jsBackend.infer(weights, input);
    reference = ref.probs;
  }
  for (const r of results) {
    if (!r.probs) continue;
    let maxDiff = 0;
    for (let i = 0; i < reference.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(r.probs[i] - reference[i]));
    }
    r.maxDiffVsJs = maxDiff;
    r.consistent = maxDiff < 1e-3;
  }
  postMessage({ type: 'benchmark-result', results });
}

function handleShapeTest(msg) {
  const { backend, deviceType } = msg;
  const badShape = [1, 1, 14, 14];
  const badInput = new Float32Array(14 * 14);
  try {
    assertInputShape(badInput, badShape);
    postMessage({ type: 'shape-test-result', ok: false, error: '校验未触发(异常)' });
  } catch (e) {
    postMessage({
      type: 'shape-test-result',
      ok: true,
      kind: classifyError(e),
      error: e.message,
      note: `已在进入 ${backend} 后端前捕获，未执行推理`,
    });
  }
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'capabilities':
        postMessage({
          type: 'capabilities',
          webnn: typeof navigator !== 'undefined' && 'ml' in navigator,
          webgpu: typeof navigator !== 'undefined' && 'gpu' in navigator,
          worker: true,
        });
        break;
      case 'getInput':
        postMessage({ type: 'input', input });
        break;
      case 'run':
        await handleRun(msg);
        break;
      case 'benchmark':
        await handleBenchmark(msg);
        break;
      case 'probeOps':
        try {
          const results = await probeOps(msg.deviceType);
          postMessage({ type: 'probe-result', results });
        } catch (err) {
          postMessage({ type: 'probe-error', error: String((err && err.message) || err) });
        }
        break;
      case 'shapeTest':
        handleShapeTest(msg);
        break;
    }
  } catch (err) {
    postMessage({ type: 'run-error', kind: '内部错误', error: String((err && err.message) || err) });
  }
};
