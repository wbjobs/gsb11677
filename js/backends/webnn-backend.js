import {
  assertShape,
  BackendUnavailableError,
  UnsupportedOpError,
  InferenceError,
} from '../validate.js';
import { intermediateShapes } from '../model.js';

const MAX_RETRY = 3;

export class WebNnBackend {
  constructor(deviceType = 'gpu') {
    this.name = 'webnn';
    this.label = `WebNN (${deviceType.toUpperCase()})`;
    this.deviceType = deviceType;
    this.context = null;
    this.graph = null;
    this.useLegacyDimensions = false;
    // Demo hook: when > 0, the next N dispatches throw to exercise retry/fallback.
    this.failNextDispatches = 0;
  }

  async init() {
    if (!('ml' in navigator) || !navigator.ml) {
      throw new BackendUnavailableError(
        '当前浏览器不支持 WebNN (navigator.ml 不存在)，需要支持 WebNN 的浏览器（如 Edge/Chrome 113+）');
    }
    try {
      this.context = await navigator.ml.createContext({ deviceType: this.deviceType });
    } catch (err) {
      throw new BackendUnavailableError(
        `WebNN ${this.deviceType} 后端不可用: ${err.message}`);
    }
    if (!this.context) {
      throw new BackendUnavailableError(`WebNN ${this.deviceType} 后端不可用: createContext 返回空`);
    }
    return true;
  }

  _desc(shape) {
    return this.useLegacyDimensions
      ? { dataType: 'float32', dimensions: shape }
      : { dataType: 'float32', shape };
  }

  // Probe which ops this WebNN implementation actually supports.
  async probeOps() {
    const builder = new MLGraphBuilder(this.context);
    const f32 = this._desc([1, 1, 4, 4]);
    const results = {};
    const tryOp = (name, fn) => {
      try {
        if (typeof builder[name] !== 'function') {
          results[name] = { supported: false, reason: 'MLGraphBuilder 上不存在该方法' };
          return;
        }
        fn();
        results[name] = { supported: true };
      } catch (err) {
        results[name] = { supported: false, reason: err.message };
      }
    };
    tryOp('conv2d', () => {
      const x = builder.input('probe_x', f32);
      const w = builder.constant(this._desc([1, 1, 3, 3]), new Float32Array(9));
      builder.conv2d(x, w, { padding: [1, 1, 1, 1] });
    });
    tryOp('gemm', () => {
      const x = builder.input('probe_g', this._desc([1, 4]));
      const w = builder.constant(this._desc([4, 4]), new Float32Array(16));
      builder.gemm(x, w);
    });
    tryOp('relu', () => builder.relu(builder.input('probe_r', f32)));
    tryOp('maxPool2d', () => {
      builder.maxPool2d(builder.input('probe_mp', f32), { windowDimensions: [2, 2] });
    });
    tryOp('averagePool2d', () => {
      builder.averagePool2d(builder.input('probe_ap', f32), { windowDimensions: [2, 2] });
    });
    // Intentionally exotic ops to demonstrate the "unsupported op" path.
    tryOp('lstm', () => {
      throw new UnsupportedOpError('lstm', '探测未执行，按不受支持处理');
    });
    tryOp('gelu', () => builder.gelu(builder.input('probe_ge', f32)));
    return results;
  }

  _buildGraph(model) {
    const { spec } = model;
    const shapes = intermediateShapes(spec);
    const builder = new MLGraphBuilder(this.context);

    const required = ['conv2d', 'gemm', 'relu', 'maxPool2d', 'averagePool2d', 'reshape'];
    for (const op of required) {
      if (typeof builder[op] !== 'function') {
        throw new UnsupportedOpError(op, 'MLGraphBuilder 缺少该方法');
      }
    }

    const input = builder.input('input', this._desc(spec.inputShape));
    const w1 = builder.constant(this._desc(spec.conv1.weightShape), model.conv1Weight);
    const b1 = builder.constant(this._desc(spec.conv1.biasShape), model.conv1Bias);
    let conv1;
    try {
      conv1 = builder.conv2d(input, w1, {
        padding: spec.conv1.padding,
        strides: spec.conv1.strides,
        bias: b1,
        inputLayout: 'nchw',
        filterLayout: 'oihw',
      });
    } catch (err) {
      throw new UnsupportedOpError('conv2d', err.message);
    }
    const conv1Relu = builder.relu(conv1);

    const pool1 = builder.maxPool2d(conv1Relu, {
      windowDimensions: spec.pool1.window,
      strides: spec.pool1.strides,
      layout: 'nchw',
    });

    const w2 = builder.constant(this._desc(spec.conv2.weightShape), model.conv2Weight);
    const b2 = builder.constant(this._desc(spec.conv2.biasShape), model.conv2Bias);
    const conv2 = builder.conv2d(pool1, w2, {
      padding: spec.conv2.padding,
      strides: spec.conv2.strides,
      bias: b2,
      inputLayout: 'nchw',
      filterLayout: 'oihw',
    });
    const conv2Relu = builder.relu(conv2);

    const pool2 = builder.averagePool2d(conv2Relu, {
      windowDimensions: spec.pool2.window,
      strides: spec.pool2.strides,
      layout: 'nchw',
    });

    const flat = builder.reshape(pool2, [1, spec.fc.inFeatures]);
    const fcW = builder.constant(this._desc([spec.fc.inFeatures, spec.fc.outFeatures]), model.fcWeight);
    const fcB = builder.constant(this._desc([spec.fc.outFeatures]), model.fcBias);
    const fcOut = builder.gemm(flat, fcW, { c: fcB });

    return builder.build({
      fcOut,
      conv1: conv1Relu,
      pool1,
      conv2: conv2Relu,
      pool2,
    });
  }

  async _ensureGraph(model) {
    if (this.graph) return this.graph;
    try {
      this.graph = await this._buildGraph(model);
    } catch (err) {
      if (!this.useLegacyDimensions &&
          (err instanceof TypeError || /shape|dimensions/i.test(err.message))) {
        // Older WebNN builds use `dimensions` instead of `shape`.
        this.useLegacyDimensions = true;
        this.graph = await this._buildGraph(model);
      } else {
        throw err;
      }
    }
    return this.graph;
  }

  async run(model, input, inputShape) {
    if (!this.context) {
      throw new BackendUnavailableError('WebNN context 未初始化');
    }
    const { spec } = model;
    assertShape(inputShape, spec.inputShape);
    const shapes = intermediateShapes(spec);
    const graph = await this._ensureGraph(model);

    const outputs = {
      fcOut: new Float32Array(spec.fc.outFeatures),
      conv1: new Float32Array(shapes.conv1Out.reduce((a, b) => a * b, 1)),
      pool1: new Float32Array(shapes.pool1Out.reduce((a, b) => a * b, 1)),
      conv2: new Float32Array(shapes.conv2Out.reduce((a, b) => a * b, 1)),
      pool2: new Float32Array(shapes.pool2Out.reduce((a, b) => a * b, 1)),
    };

    let lastError = null;
    for (let attempt = 1; attempt <= MAX_RETRY; attempt++) {
      try {
        if (this.failNextDispatches > 0) {
          this.failNextDispatches--;
          throw new Error('模拟的 dispatch 失败（演示重试/降级）');
        }
        this.context.dispatch(graph, { input }, outputs);
        return {
          output: outputs.fcOut,
          outputShape: shapes.fcOut,
          intermediates: {
            conv1: outputs.conv1,
            pool1: outputs.pool1,
            conv2: outputs.conv2,
            pool2: outputs.pool2,
          },
          shapes,
          attempts: attempt,
        };
      } catch (err) {
        lastError = err;
        if (attempt < MAX_RETRY) {
          await new Promise((resolve) => setTimeout(resolve, 100 * 2 ** (attempt - 1)));
        }
      }
    }
    throw new InferenceError(
      `WebNN 推理失败，已重试 ${MAX_RETRY} 次: ${lastError?.message}`, lastError);
  }

  async dispose() {
    this.graph = null;
    this.context = null;
  }
}
