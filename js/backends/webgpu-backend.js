import { assertShape, BackendUnavailableError, InferenceError } from '../validate.js';
import { intermediateShapes } from '../model.js';

const CONV_SHADER = /* wgsl */`
struct Params {
  n: u32, cIn: u32, hIn: u32, wIn: u32,
  cOut: u32, kH: u32, kW: u32,
  padT: u32, padL: u32, sH: u32, sW: u32,
  hOut: u32, wOut: u32, useRelu: u32, _pad: u32,
};
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read> wBuf: array<f32>;
@group(0) @binding(2) var<storage, read> bBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.n * p.cOut * p.hOut * p.wOut;
  if (idx >= total) { return; }
  let ow = idx % p.wOut;
  let oh = (idx / p.wOut) % p.hOut;
  let oc = (idx / (p.wOut * p.hOut)) % p.cOut;
  let b  = idx / (p.wOut * p.hOut * p.cOut);
  var acc = bBuf[oc];
  for (var ic = 0u; ic < p.cIn; ic++) {
    for (var kh = 0u; kh < p.kH; kh++) {
      let ih = oh * p.sH + kh;
      if (ih < p.padT || ih >= p.hIn + p.padT) { continue; }
      for (var kw = 0u; kw < p.kW; kw++) {
        let iw = ow * p.sW + kw;
        if (iw < p.padL || iw >= p.wIn + p.padL) { continue; }
        let inIdx = ((b * p.cIn + ic) * p.hIn + (ih - p.padT)) * p.wIn + (iw - p.padL);
        let wIdx = ((oc * p.cIn + ic) * p.kH + kh) * p.kW + kw;
        acc += inBuf[inIdx] * wBuf[wIdx];
      }
    }
  }
  if (p.useRelu == 1u && acc < 0.0) { acc = 0.0; }
  outBuf[idx] = acc;
}`;

const POOL_SHADER = /* wgsl */`
struct Params {
  n: u32, c: u32, hIn: u32, wIn: u32,
  wH: u32, wW: u32, sH: u32, sW: u32,
  hOut: u32, wOut: u32, kind: u32, _p0: u32,
  _p1: u32, _p2: u32, _p3: u32, _p4: u32,
};
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.n * p.c * p.hOut * p.wOut;
  if (idx >= total) { return; }
  let ow = idx % p.wOut;
  let oh = (idx / p.wOut) % p.hOut;
  let ch = (idx / (p.wOut * p.hOut)) % p.c;
  let b  = idx / (p.wOut * p.hOut * p.c);
  var acc = 0.0;
  if (p.kind == 0u) { acc = -3.4e38; }
  for (var kh = 0u; kh < p.wH; kh++) {
    for (var kw = 0u; kw < p.wW; kw++) {
      let ih = oh * p.sH + kh;
      let iw = ow * p.sW + kw;
      let v = inBuf[((b * p.c + ch) * p.hIn + ih) * p.wIn + iw];
      if (p.kind == 0u) { acc = max(acc, v); } else { acc += v; }
    }
  }
  if (p.kind == 1u) { acc = acc / f32(p.wH * p.wW); }
  outBuf[idx] = acc;
}`;

const GEMM_SHADER = /* wgsl */`
struct Params { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> aBuf: array<f32>;
@group(0) @binding(1) var<storage, read> bBuf: array<f32>;
@group(0) @binding(2) var<storage, read> cBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.m * p.n) { return; }
  let j = idx % p.n;
  let i = idx / p.n;
  var acc = cBuf[j];
  for (var t = 0u; t < p.k; t++) {
    acc += aBuf[i * p.k + t] * bBuf[t * p.n + j];
  }
  outBuf[idx] = acc;
}`;

export class WebGpuBackend {
  constructor() {
    this.name = 'webgpu';
    this.label = 'WebGPU';
    this.device = null;
  }

  async init() {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new BackendUnavailableError('当前浏览器不支持 WebGPU (navigator.gpu 不存在)');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new BackendUnavailableError('WebGPU 适配器不可用 (requestAdapter 返回 null)');
    }
    this.device = await adapter.requestDevice();
    this.device.lost.then((info) => {
      if (info.reason !== 'destroyed') {
        this.device = null;
      }
    });
    this.convPipeline = this._pipeline(CONV_SHADER);
    this.poolPipeline = this._pipeline(POOL_SHADER);
    this.gemmPipeline = this._pipeline(GEMM_SHADER);
    return true;
  }

  _pipeline(code) {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  _buffer(dataOrSize, usage) {
    const size = typeof dataOrSize === 'number' ? dataOrSize : dataOrSize.byteLength;
    const buffer = this.device.createBuffer({
      size: Math.max(16, (size + 15) & ~15),
      usage,
      mappedAtCreation: typeof dataOrSize !== 'number',
    });
    if (typeof dataOrSize !== 'number') {
      new Float32Array(buffer.getMappedRange()).set(dataOrSize);
      buffer.unmap();
    }
    return buffer;
  }

  _runPass(pipeline, buffers, params, outElementCount) {
    const entries = buffers.map((buffer, i) => ({
      binding: i,
      resource: { buffer },
    }));
    const paramBuffer = this._buffer(new Uint32Array(params),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    entries.push({ binding: buffers.length, resource: { buffer: paramBuffer } });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outElementCount / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    paramBuffer.destroy();
  }

  async _readback(buffer, elementCount) {
    const size = elementCount * 4;
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  async run(model, input, inputShape) {
    if (!this.device) {
      throw new BackendUnavailableError('WebGPU 设备已丢失或未初始化');
    }
    const { spec } = model;
    assertShape(inputShape, spec.inputShape);
    const shapes = intermediateShapes(spec);
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    try {
      const inBuf = this._buffer(input, S);
      const w1 = this._buffer(model.conv1Weight, S);
      const b1 = this._buffer(model.conv1Bias, S);
      const conv1Count = shapes.conv1Out.reduce((a, b) => a * b, 1);
      const conv1Buf = this._buffer(conv1Count * 4, S);
      this._runPass(this.convPipeline, [inBuf, w1, b1, conv1Buf],
        [1, 1, 28, 28, 4, 3, 3, 1, 1, 1, 1, 28, 28, 1, 0], conv1Count);

      const pool1Count = shapes.pool1Out.reduce((a, b) => a * b, 1);
      const pool1Buf = this._buffer(pool1Count * 4, S);
      this._runPass(this.poolPipeline, [conv1Buf, pool1Buf],
        [1, 4, 28, 28, 2, 2, 2, 2, 14, 14, 0, 0, 0, 0, 0, 0], pool1Count);

      const w2 = this._buffer(model.conv2Weight, S);
      const b2 = this._buffer(model.conv2Bias, S);
      const conv2Count = shapes.conv2Out.reduce((a, b) => a * b, 1);
      const conv2Buf = this._buffer(conv2Count * 4, S);
      this._runPass(this.convPipeline, [pool1Buf, w2, b2, conv2Buf],
        [1, 4, 14, 14, 8, 3, 3, 1, 1, 1, 1, 14, 14, 1, 0], conv2Count);

      const pool2Count = shapes.pool2Out.reduce((a, b) => a * b, 1);
      const pool2Buf = this._buffer(pool2Count * 4, S);
      this._runPass(this.poolPipeline, [conv2Buf, pool2Buf],
        [1, 8, 14, 14, 2, 2, 2, 2, 7, 7, 1, 0, 0, 0, 0, 0], pool2Count);

      const fcW = this._buffer(model.fcWeight, S);
      const fcB = this._buffer(model.fcBias, S);
      const fcCount = spec.fc.outFeatures;
      const fcBuf = this._buffer(fcCount * 4, S);
      this._runPass(this.gemmPipeline, [pool2Buf, fcW, fcB, fcBuf],
        [1, spec.fc.inFeatures, spec.fc.outFeatures, 0], fcCount);

      const [output, conv1, pool1, conv2, pool2] = await Promise.all([
        this._readback(fcBuf, fcCount),
        this._readback(conv1Buf, conv1Count),
        this._readback(pool1Buf, pool1Count),
        this._readback(conv2Buf, conv2Count),
        this._readback(pool2Buf, pool2Count),
      ]);

      for (const buf of [inBuf, w1, b1, conv1Buf, pool1Buf, w2, b2, conv2Buf, pool2Buf, fcW, fcB, fcBuf]) {
        buf.destroy();
      }

      return {
        output,
        outputShape: shapes.fcOut,
        intermediates: { conv1, pool1, conv2, pool2 },
        shapes,
      };
    } catch (err) {
      if (err.name && err.name.endsWith('Error') && err.name !== 'Error') throw err;
      throw new InferenceError(`WebGPU 推理失败: ${err.message}`, err);
    }
  }

  async dispose() {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
  }
}
