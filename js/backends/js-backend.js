import {
  CONV_OUT_CHANNELS,
  KERNEL,
  CONV_H,
  CONV_W,
  POOLED_H,
  POOLED_W,
  FLAT_SIZE,
  NUM_CLASSES,
} from '../model.js';

export const backendName = 'js';

function conv2dRelu(input, convW, convB) {
  const out = new Float32Array(CONV_OUT_CHANNELS * CONV_H * CONV_W);
  const pad = (KERNEL - 1) / 2;
  for (let oc = 0; oc < CONV_OUT_CHANNELS; oc++) {
    for (let oy = 0; oy < CONV_H; oy++) {
      for (let ox = 0; ox < CONV_W; ox++) {
        let sum = convB[oc];
        for (let ky = 0; ky < KERNEL; ky++) {
          const iy = oy + ky - pad;
          if (iy < 0 || iy >= CONV_H) continue;
          for (let kx = 0; kx < KERNEL; kx++) {
            const ix = ox + kx - pad;
            if (ix < 0 || ix >= CONV_W) continue;
            sum += input[iy * CONV_W + ix] * convW[oc * KERNEL * KERNEL + ky * KERNEL + kx];
          }
        }
        out[oc * CONV_H * CONV_W + oy * CONV_W + ox] = Math.max(sum, 0);
      }
    }
  }
  return out;
}

function maxPool2d(conv) {
  const out = new Float32Array(CONV_OUT_CHANNELS * POOLED_H * POOLED_W);
  for (let c = 0; c < CONV_OUT_CHANNELS; c++) {
    const base = c * CONV_H * CONV_W;
    for (let oy = 0; oy < POOLED_H; oy++) {
      for (let ox = 0; ox < POOLED_W; ox++) {
        const y = oy * 2;
        const x = ox * 2;
        const m = Math.max(
          conv[base + y * CONV_W + x],
          conv[base + y * CONV_W + x + 1],
          conv[base + (y + 1) * CONV_W + x],
          conv[base + (y + 1) * CONV_W + x + 1]
        );
        out[c * POOLED_H * POOLED_W + oy * POOLED_W + ox] = m;
      }
    }
  }
  return out;
}

function gemm(flat, fcW, fcB) {
  const logits = new Float32Array(NUM_CLASSES);
  for (let o = 0; o < NUM_CLASSES; o++) {
    let sum = fcB[o];
    for (let i = 0; i < FLAT_SIZE; i++) sum += flat[i] * fcW[i * NUM_CLASSES + o];
    logits[o] = sum;
  }
  return logits;
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let total = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp(logits[i] - max);
    total += probs[i];
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= total;
  return probs;
}

export function infer(weights, input) {
  const conv = conv2dRelu(input, weights.convW, weights.convB);
  const pooled = maxPool2d(conv);
  const logits = gemm(pooled, weights.fcW, weights.fcB);
  return { probs: softmax(logits), pooled, logits };
}
