import { createModel } from './model.js';
import { maxAbsDiff } from './model.js';
import { JsBackend } from './backends/js-backend.js';
import { WebGpuBackend } from './backends/webgpu-backend.js';
import { WebNnBackend } from './backends/webnn-backend.js';
import {
  BackendUnavailableError,
  UnsupportedOpError,
  ShapeMismatchError,
  InferenceError,
} from './validate.js';

const TOLERANCE = 1e-3;
const model = createModel(42);
const backends = new Map();

// PerformanceObserver: forward all measure entries to the main thread.
let perfObserver = null;
try {
  perfObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      postMessage({
        type: 'perf',
        entry: { name: entry.name, duration: entry.duration, startTime: entry.startTime },
      });
    }
  });
  perfObserver.observe({ entryTypes: ['measure'] });
} catch {
  perfObserver = null;
}

function log(level, message) {
  postMessage({ type: 'log', level, message });
}

function timed(label, fn) {
  const startMark = `${label}:start:${performance.now()}`;
  performance.mark(startMark);
  return Promise.resolve()
    .then(fn)
    .then((value) => {
      performance.measure(label, { start: startMark });
      return value;
    });
}

async function getBackend(name) {
  if (backends.has(name)) return backends.get(name);
  let backend;
  if (name === 'webnn') backend = new WebNnBackend('gpu');
  else if (name === 'webgpu') backend = new WebGpuBackend();
  else backend = new JsBackend();
  await timed(`init:${name}`, () => backend.init());
  backends.set(name, backend);
  log('info', `后端已初始化: ${backend.label}`);
  return backend;
}

// Fallback chain: webnn -> webgpu -> js
const CHAIN = { webnn: 'webgpu', webgpu: 'js', js: null };

async function runWithFallback(requested, input, inputShape, options) {
  const fallbacks = [];
  let name = requested;
  while (name) {
    try {
      const backend = await getBackend(name);
      if (name === 'webnn' && options.simulateFailure) {
        backend.failNextDispatches = 3; // exhaust retries to demo fallback
      }
      const result = await timed(`inference:${name}`, () => backend.run(model, input, inputShape));
      return { result, backendName: name, backendLabel: backend.label, fallbacks };
    } catch (err) {
      if (err instanceof ShapeMismatchError) {
        // Shape errors are caller bugs: do not fall back, surface them.
        throw err;
      }
      const reason = describeError(err);
      fallbacks.push({ from: name, reason });
      log('warn', `[${name}] ${reason} → 尝试降级`);
      name = CHAIN[name];
    }
  }
  throw new InferenceError('所有后端均不可用');
}

function describeError(err) {
  if (err instanceof BackendUnavailableError) return `后端不可用: ${err.message}`;
  if (err instanceof UnsupportedOpError) return err.message;
  if (err instanceof InferenceError) return err.message;
  return `未知错误: ${err.message}`;
}

async function handleRun(id, msg) {
  const inputShape = msg.badShape ? [1, 1, 32, 32] : [...model.spec.inputShape];
  const input = msg.badShape ? new Float32Array(32 * 32).fill(0.5) : model.input;
  try {
    const { result, backendName, backendLabel, fallbacks } =
      await runWithFallback(msg.backend, input, inputShape, msg);

    // Consistency check against the pure-JS reference.
    let consistency = null;
    if (backendName !== 'js') {
      const js = await getBackend('js');
      const ref = await js.run(model, model.input, model.spec.inputShape);
      const diff = maxAbsDiff(result.output, ref.output);
      consistency = { ok: diff <= TOLERANCE, maxDiff: diff };
      if (!consistency.ok) {
        log('error', `结果不一致! ${backendLabel} vs JS 最大误差 ${diff}`);
      }
    }

    postMessage({
      id,
      type: 'result',
      payload: {
        backend: backendName,
        backendLabel,
        requested: msg.backend,
        fallbacks,
        output: result.output,
        outputShape: result.outputShape,
        intermediates: result.intermediates,
        shapes: result.shapes,
        attempts: result.attempts ?? 1,
        consistency,
      },
    });
  } catch (err) {
    postMessage({
      id,
      type: 'error',
      error: {
        name: err.name,
        message: err.message,
        isShapeMismatch: err instanceof ShapeMismatchError,
      },
    });
  }
}

async function handleBenchmark(id, msg) {
  const iterations = msg.iterations ?? 50;
  const warmup = 5;
  const order = ['webnn', 'webgpu', 'js'];
  const rows = [];
  for (const name of order) {
    let backend;
    try {
      backend = await getBackend(name);
    } catch (err) {
      rows.push({ backend: name, label: name, unavailable: describeError(err) });
      continue;
    }
    try {
      for (let i = 0; i < warmup; i++) {
        await backend.run(model, model.input, model.spec.inputShape);
      }
      const times = [];
      for (let i = 0; i < iterations; i++) {
        const t0 = performance.now();
        await backend.run(model, model.input, model.spec.inputShape);
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      rows.push({
        backend: name,
        label: backend.label,
        avg,
        min: times[0],
        max: times[times.length - 1],
        p50: times[Math.floor(times.length / 2)],
        iterations,
      });
      log('info', `基准 ${backend.label}: avg=${avg.toFixed(3)}ms`);
    } catch (err) {
      rows.push({ backend: name, label: backend.label, unavailable: describeError(err) });
    }
  }
  postMessage({ id, type: 'benchmarkResult', rows });
}

async function handleProbe(id) {
  try {
    const webnn = await getBackend('webnn');
    const results = await webnn.probeOps();
    postMessage({ id, type: 'probeResult', results });
  } catch (err) {
    postMessage({ id, type: 'error', error: { name: err.name, message: describeError(err) } });
  }
}

onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'run') handleRun(msg.id, msg);
  else if (msg.type === 'benchmark') handleBenchmark(msg.id, msg);
  else if (msg.type === 'probe') handleProbe(msg.id);
};

log('info', 'Worker 已就绪，模型权重已用固定种子生成（三个后端共享同一权重）');
