import { mulberry32 } from './prng.js';

// Network topology: Conv2D(1->4, 3x3, pad 1) + ReLU -> MaxPool2D(2x2, stride 2)
// -> Flatten -> Gemm(784->10) -> Softmax
export const INPUT_SHAPE = [1, 1, 28, 28];
export const CONV_OUT_CHANNELS = 4;
export const KERNEL = 3;
export const POOL = 2;
export const CONV_H = 28;
export const CONV_W = 28;
export const POOLED_H = 14;
export const POOLED_W = 14;
export const FLAT_SIZE = CONV_OUT_CHANNELS * POOLED_H * POOLED_W; // 784
export const NUM_CLASSES = 10;

export function generateWeights(seed = 11677) {
  const rand = mulberry32(seed);
  const heScale = (fanIn) => Math.sqrt(2 / fanIn);
  const randSym = (scale) => (rand() * 2 - 1) * scale;
  return {
    convW: Float32Array.from({ length: CONV_OUT_CHANNELS * 1 * KERNEL * KERNEL }, () =>
      randSym(heScale(KERNEL * KERNEL))
    ),
    convB: Float32Array.from({ length: CONV_OUT_CHANNELS }, () => randSym(0.1)),
    fcW: Float32Array.from({ length: FLAT_SIZE * NUM_CLASSES }, () => randSym(heScale(FLAT_SIZE))),
    fcB: Float32Array.from({ length: NUM_CLASSES }, () => randSym(0.1)),
  };
}

// Deterministic "digit-like" 28x28 grayscale pattern in [0, 1].
export function generateInput(seed = 42) {
  const rand = mulberry32(seed);
  const data = new Float32Array(28 * 28);
  for (let x = 5; x < 23; x++) data[4 * 28 + x] = 1; // top bar
  for (let i = 0; i < 19; i++) {
    const y = 4 + i;
    const x = 22 - Math.round(i * 0.7);
    data[y * 28 + x] = 1;
    if (x + 1 < 28) data[y * 28 + x + 1] = 0.8;
  }
  for (let i = 0; i < data.length; i++) {
    const n = rand();
    if (n > 0.97) data[i] = Math.max(data[i], n - 0.2);
  }
  return data;
}

export function assertInputShape(input, shape) {
  const expected = INPUT_SHAPE;
  const sameDims =
    shape.length === expected.length && shape.every((d, i) => d === expected[i]);
  const expectedLen = expected.reduce((a, b) => a * b, 1);
  if (!sameDims || input.length !== expectedLen) {
    throw new ShapeMismatchError(
      `输入形状不匹配: 期望 [${expected.join(',')}] (长度 ${expectedLen}), ` +
        `实际 [${shape.join(',')}] (长度 ${input.length})`
    );
  }
}

export class ShapeMismatchError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ShapeMismatchError';
  }
}

export class BackendUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BackendUnavailableError';
  }
}

export class OpUnsupportedError extends Error {
  constructor(op, message) {
    super(message);
    this.name = 'OpUnsupportedError';
    this.op = op;
  }
}
