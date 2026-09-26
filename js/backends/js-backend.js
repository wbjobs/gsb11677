import { assertShape } from '../validate.js';
import { intermediateShapes } from '../model.js';

// Pure-JS reference implementation. Also serves as the ground truth
// that WebNN / WebGPU outputs are validated against.
export class JsBackend {
  constructor() {
    this.name = 'js';
    this.label = 'Pure JS (CPU)';
  }

  async init() {
    return true; // always available
  }

  async run(model, input, inputShape) {
    const { spec } = model;
    assertShape(inputShape, spec.inputShape);

    const shapes = intermediateShapes(spec);
    const conv1 = conv2dNchw(input, inputShape, model.conv1Weight, spec.conv1.weightShape,
      model.conv1Bias, spec.conv1.padding, spec.conv1.strides);
    reluInPlace(conv1);

    const pool1 = pool2dNchw(conv1, shapes.conv1Out, spec.pool1.window, spec.pool1.strides, 'max');

    const conv2 = conv2dNchw(pool1, shapes.pool1Out, model.conv2Weight, spec.conv2.weightShape,
      model.conv2Bias, spec.conv2.padding, spec.conv2.strides);
    reluInPlace(conv2);

    const pool2 = pool2dNchw(conv2, shapes.conv2Out, spec.pool2.window, spec.pool2.strides, 'average');

    const fcOut = gemm(pool2, [1, spec.fc.inFeatures], model.fcWeight,
      [spec.fc.inFeatures, spec.fc.outFeatures], model.fcBias);

    return {
      output: fcOut,
      outputShape: shapes.fcOut,
      intermediates: { conv1, pool1, conv2, pool2 },
      shapes,
    };
  }

  async dispose() {}
}

export function conv2dNchw(input, inputShape, weight, weightShape, bias, padding, strides) {
  const [n, cIn, hIn, wIn] = inputShape;
  const [cOut, cInW, kH, kW] = weightShape;
  if (cIn !== cInW) {
    throw new Error(`conv2d 通道不匹配: input C=${cIn}, weight C=${cInW}`);
  }
  const [padT, padB, padL, padR] = padding;
  const [sH, sW] = strides;
  const hOut = Math.floor((hIn + padT + padB - kH) / sH) + 1;
  const wOut = Math.floor((wIn + padL + padR - kW) / sW) + 1;
  const out = new Float32Array(n * cOut * hOut * wOut);

  for (let b = 0; b < n; b++) {
    for (let oc = 0; oc < cOut; oc++) {
      for (let oh = 0; oh < hOut; oh++) {
        for (let ow = 0; ow < wOut; ow++) {
          let acc = bias ? bias[oc] : 0;
          for (let ic = 0; ic < cIn; ic++) {
            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * sH + kh - padT;
              if (ih < 0 || ih >= hIn) continue;
              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * sW + kw - padL;
                if (iw < 0 || iw >= wIn) continue;
                const inIdx = ((b * cIn + ic) * hIn + ih) * wIn + iw;
                const wIdx = ((oc * cIn + ic) * kH + kh) * kW + kw;
                acc += input[inIdx] * weight[wIdx];
              }
            }
          }
          out[((b * cOut + oc) * hOut + oh) * wOut + ow] = acc;
        }
      }
    }
  }
  return out;
}

export function pool2dNchw(input, inputShape, window, strides, kind) {
  const [n, c, hIn, wIn] = inputShape;
  const [wH, wW] = window;
  const [sH, sW] = strides;
  const hOut = Math.floor((hIn - wH) / sH) + 1;
  const wOut = Math.floor((wIn - wW) / sW) + 1;
  const out = new Float32Array(n * c * hOut * wOut);

  for (let b = 0; b < n; b++) {
    for (let ch = 0; ch < c; ch++) {
      for (let oh = 0; oh < hOut; oh++) {
        for (let ow = 0; ow < wOut; ow++) {
          let acc = kind === 'max' ? -Infinity : 0;
          for (let kh = 0; kh < wH; kh++) {
            for (let kw = 0; kw < wW; kw++) {
              const ih = oh * sH + kh;
              const iw = ow * sW + kw;
              const v = input[((b * c + ch) * hIn + ih) * wIn + iw];
              if (kind === 'max') {
                if (v > acc) acc = v;
              } else {
                acc += v;
              }
            }
          }
          if (kind === 'average') acc /= wH * wW;
          out[((b * c + ch) * hOut + oh) * wOut + ow] = acc;
        }
      }
    }
  }
  return out;
}

export function gemm(a, aShape, b, bShape, c) {
  const [m, k] = aShape;
  const [k2, n] = bShape;
  if (k !== k2) {
    throw new Error(`gemm 形状不匹配: [${aShape}] x [${bShape}]`);
  }
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = c ? c[j] : 0;
      for (let p = 0; p < k; p++) {
        acc += a[i * k + p] * b[p * n + j];
      }
      out[i * n + j] = acc;
    }
  }
  return out;
}

export function reluInPlace(data) {
  for (let i = 0; i < data.length; i++) {
    if (data[i] < 0) data[i] = 0;
  }
  return data;
}
