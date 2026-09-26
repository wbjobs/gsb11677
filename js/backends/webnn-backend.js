import {
  CONV_OUT_CHANNELS,
  KERNEL,
  POOLED_H,
  POOLED_W,
  FLAT_SIZE,
  NUM_CLASSES,
  BackendUnavailableError,
  OpUnsupportedError,
} from '../model.js';

export const backendName = 'webnn';

function isWebNNSupported() {
  return typeof navigator !== 'undefined' && 'ml' in navigator;
}

export async function createWebNNBackend(deviceType = 'gpu') {
  if (!isWebNNSupported()) {
    throw new BackendUnavailableError(
      '当前浏览器不支持 WebNN (navigator.ml 不存在)，请使用 Chromium 系浏览器并启用 WebNN'
    );
  }
  let context;
  try {
    context = await navigator.ml.createContext({ deviceType });
  } catch (e) {
    throw new BackendUnavailableError(
      `WebNN 后端不可用 (deviceType=${deviceType}): ${e.message || e}`
    );
  }

  let graph = null;
  return {
    name: backendName,
    deviceType,
    async infer(weights, input) {
      // Graph constants are baked at build time; (re)build when weights change.
      if (!graph || graph.weightsRef !== weights) {
        graph = await buildGraph(context, weights);
      }
      return graph.compute(input);
    },
  };
}

async function buildGraph(context, weights) {
  const builder = new MLGraphBuilder(context);

  // Wrap every op so failures identify the offending operator.
  const track = (op, fn) => {
    try {
      return fn();
    } catch (e) {
      throw new OpUnsupportedError(op, `算子 ${op} 构建失败(可能不被当前后端支持): ${e.message || e}`);
    }
  };

  // Modern spec: {dataType, shape}; legacy spec: {type, dimensions}. Try modern first.
  let useLegacy = false;
  const desc = (shape) =>
    useLegacy
      ? { type: 'float32', dimensions: shape }
      : { dataType: 'float32', shape };

  const build = async () => {
    const input = track('input', () => builder.input('input', desc([1, 1, 28, 28])));
    const constant = (data, shape) =>
      track('constant', () => builder.constant(desc(shape), data));

    const w = weights;
    const convW = constant(w.convW, [CONV_OUT_CHANNELS, 1, KERNEL, KERNEL]);
    const convB = constant(w.convB, [CONV_OUT_CHANNELS]);
    const conv = track('conv2d', () =>
      builder.conv2d(input, convW, {
        bias: convB,
        padding: [1, 1, 1, 1],
        strides: [1, 1],
        dilations: [1, 1],
        groups: 1,
        inputLayout: 'nchw',
        filterLayout: 'oihw',
      })
    );
    const relu = track('relu', () => builder.relu(conv));
    const pooled = track('maxPool2d', () =>
      builder.maxPool2d(relu, {
        windowDimensions: [2, 2],
        strides: [2, 2],
        layout: 'nchw',
      })
    );
    const flat = track('reshape', () => builder.reshape(pooled, [1, FLAT_SIZE]));
    const fcW = constant(w.fcW, [FLAT_SIZE, NUM_CLASSES]);
    const fcB = constant(w.fcB, [NUM_CLASSES]);
    const logits = track('gemm', () => builder.gemm(flat, fcW, { c: fcB }));
    const probs = track('softmax', () => {
      try {
        return builder.softmax(logits, 1);
      } catch (_) {
        return builder.softmax(logits);
      }
    });

    try {
      return await builder.build({ probs, pooled });
    } catch (e) {
      const msg = String((e && e.message) || e);
      if (!useLegacy && /shape|dataType|operand/i.test(msg)) {
        useLegacy = true;
        return build();
      }
      throw new OpUnsupportedError(
        'build',
        `WebNN 图构建失败(可能包含不支持的算子): ${msg}`
      );
    }
  };

  const mlGraph = await build();

  const compute = async (inputData) => {
    const probOut = new Float32Array(NUM_CLASSES);
    const pooledOut = new Float32Array(FLAT_SIZE);
    const inputs = { input: inputData };
    const outputs = { probs: probOut, pooled: pooledOut };
    if (typeof context.compute === 'function') {
      await context.compute(mlGraph, inputs, outputs);
    } else if (typeof mlGraph.compute === 'function') {
      await mlGraph.compute(inputs, outputs);
    } else {
      throw new BackendUnavailableError('WebNN 后端不可用: 找不到 compute 接口');
    }
    return { probs: probOut, pooled: pooledOut, logits: probOut };
  };

  return { compute, weightsRef: weights };
}

// Probe a list of operators by building minimal graphs; report unsupported ones.
export async function probeOps(deviceType = 'gpu') {
  if (!isWebNNSupported()) {
    throw new BackendUnavailableError('当前浏览器不支持 WebNN，无法探测算子');
  }
  let context;
  try {
    context = await navigator.ml.createContext({ deviceType });
  } catch (e) {
    throw new BackendUnavailableError(`WebNN 后端不可用: ${e.message || e}`);
  }
  const f32 = (shape) => ({ dataType: 'float32', shape });
  const probes = {
    conv2d: (b) => {
      const x = b.input('x', f32([1, 1, 4, 4]));
      const w = b.constant(f32([1, 1, 2, 2]), new Float32Array(4));
      return b.build({ y: b.conv2d(x, w) });
    },
    maxPool2d: (b) => {
      const x = b.input('x', f32([1, 1, 4, 4]));
      return b.build({ y: b.maxPool2d(x, { windowDimensions: [2, 2], strides: [2, 2] }) });
    },
    averagePool2d: (b) => {
      const x = b.input('x', f32([1, 1, 4, 4]));
      return b.build({ y: b.averagePool2d(x, { windowDimensions: [2, 2], strides: [2, 2] }) });
    },
    gemm: (b) => {
      const x = b.input('x', f32([1, 4]));
      const w = b.constant(f32([4, 2]), new Float32Array(8));
      return b.build({ y: b.gemm(x, w) });
    },
    relu: (b) => b.build({ y: b.relu(b.input('x', f32([1, 4]))) }),
    sigmoid: (b) => b.build({ y: b.sigmoid(b.input('x', f32([1, 4]))) }),
    tanh: (b) => b.build({ y: b.tanh(b.input('x', f32([1, 4]))) }),
    softmax: (b) => {
      const x = b.input('x', f32([1, 4]));
      let y;
      try {
        y = b.softmax(x, 1);
      } catch (_) {
        y = b.softmax(x);
      }
      return b.build({ y });
    },
    batchNormalization: (b) => {
      const x = b.input('x', f32([1, 2, 2, 2]));
      const mean = b.constant(f32([2]), new Float32Array(2));
      const variance = b.constant(f32([2]), new Float32Array([1, 1]));
      return b.build({ y: b.batchNormalization(x, mean, variance) });
    },
    convTranspose2d: (b) => {
      const x = b.input('x', f32([1, 1, 4, 4]));
      const w = b.constant(f32([1, 1, 2, 2]), new Float32Array(4));
      return b.build({ y: b.convTranspose2d(x, w) });
    },
    gelu: (b) => b.build({ y: b.gelu(b.input('x', f32([1, 4]))) }),
    lstm: (b) => {
      const x = b.input('x', f32([1, 1, 4]));
      const w = b.constant(f32([1, 16, 4]), new Float32Array(64));
      const r = b.constant(f32([1, 16, 4]), new Float32Array(64));
      return b.build({ y: b.lstm(x, w, r, { returnSequence: true }) });
    },
  };
  const results = [];
  for (const [op, fn] of Object.entries(probes)) {
    try {
      await fn(new MLGraphBuilder(context));
      results.push({ op, supported: true });
    } catch (e) {
      results.push({ op, supported: false, error: String((e && e.message) || e) });
    }
  }
  return results;
}
