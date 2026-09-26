import { randomTensor } from './prng.js';

// A small deterministic CNN:
//   input  [1,1,28,28]
//   conv1  3x3 pad=1 -> [1,4,28,28]  + relu
//   pool1  max 2x2/2 -> [1,4,14,14]
//   conv2  3x3 pad=1 -> [1,8,14,14]  + relu
//   pool2  avg 2x2/2 -> [1,8,7,7]
//   fc     gemm 392->10 -> [1,10]
export const MODEL_SPEC = {
  inputShape: [1, 1, 28, 28],
  conv1: { weightShape: [4, 1, 3, 3], biasShape: [4], padding: [1, 1, 1, 1], strides: [1, 1] },
  pool1: { kind: 'max', window: [2, 2], strides: [2, 2] },
  conv2: { weightShape: [8, 4, 3, 3], biasShape: [8], padding: [1, 1, 1, 1], strides: [1, 1] },
  pool2: { kind: 'average', window: [2, 2], strides: [2, 2] },
  fc: { inFeatures: 8 * 7 * 7, outFeatures: 10 },
};

export function createModel(seed = 42) {
  const spec = MODEL_SPEC;
  return {
    spec,
    input: randomTensor(spec.inputShape, seed, 1.0),
    conv1Weight: randomTensor(spec.conv1.weightShape, seed + 1, 0.4),
    conv1Bias: randomTensor(spec.conv1.biasShape, seed + 2, 0.1),
    conv2Weight: randomTensor(spec.conv2.weightShape, seed + 3, 0.4),
    conv2Bias: randomTensor(spec.conv2.biasShape, seed + 4, 0.1),
    fcWeight: randomTensor([spec.fc.inFeatures, spec.fc.outFeatures], seed + 5, 0.2),
    fcBias: randomTensor([spec.fc.outFeatures], seed + 6, 0.1),
  };
}

// Intermediate shapes, useful for validation and visualization.
export function intermediateShapes(spec = MODEL_SPEC) {
  const [, , h, w] = spec.inputShape;
  const conv1Out = [1, spec.conv1.weightShape[0], h, w];
  const pool1Out = [1, conv1Out[1], h / 2, w / 2];
  const conv2Out = [1, spec.conv2.weightShape[0], h / 2, w / 2];
  const pool2Out = [1, conv2Out[1], h / 4, w / 4];
  return { conv1Out, pool1Out, conv2Out, pool2Out, fcOut: [1, spec.fc.outFeatures] };
}

export function maxAbsDiff(a, b) {
  if (a.length !== b.length) return Infinity;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}
