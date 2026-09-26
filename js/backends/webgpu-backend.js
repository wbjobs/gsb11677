import {
  CONV_OUT_CHANNELS,
  KERNEL,
  CONV_H,
  CONV_W,
  POOLED_H,
  POOLED_W,
  FLAT_SIZE,
  NUM_CLASSES,
  BackendUnavailableError,
} from '../model.js';
import { softmax } from './js-backend.js';

export const backendName = 'webgpu';

const CONV_SHADER = `
const H: u32 = ${CONV_H}u;
const W: u32 = ${CONV_W}u;
const CO: u32 = ${CONV_OUT_CHANNELS}u;
const K: u32 = ${KERNEL}u;
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read> wBuf: array<f32>;
@group(0) @binding(2) var<storage, read> bBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x; let oy = gid.y; let oc = gid.z;
  if (ox >= W || oy >= H || oc >= CO) { return; }
  var sum = bBuf[oc];
  for (var ky = 0u; ky < K; ky = ky + 1u) {
    for (var kx = 0u; kx < K; kx = kx + 1u) {
      let iy = i32(oy) + i32(ky) - 1;
      let ix = i32(ox) + i32(kx) - 1;
      if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        sum = sum + inBuf[u32(iy) * W + u32(ix)] * wBuf[oc * K * K + ky * K + kx];
      }
    }
  }
  outBuf[oc * H * W + oy * W + ox] = max(sum, 0.0);
}`;

const POOL_SHADER = `
const H: u32 = ${CONV_H}u;
const W: u32 = ${CONV_W}u;
const CO: u32 = ${CONV_OUT_CHANNELS}u;
const OH: u32 = ${POOLED_H}u;
const OW: u32 = ${POOLED_W}u;
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read_write> outBuf: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x; let oy = gid.y; let oc = gid.z;
  if (ox >= OW || oy >= OH || oc >= CO) { return; }
  let base = oc * H * W;
  let y = oy * 2u; let x = ox * 2u;
  let m = max(
    max(inBuf[base + y * W + x], inBuf[base + y * W + x + 1u]),
    max(inBuf[base + (y + 1u) * W + x], inBuf[base + (y + 1u) * W + x + 1u])
  );
  outBuf[oc * OH * OW + oy * OW + ox] = m;
}`;

const GEMM_SHADER = `
const N: u32 = ${FLAT_SIZE}u;
const M: u32 = ${NUM_CLASSES}u;
@group(0) @binding(0) var<storage, read> inBuf: array<f32>;
@group(0) @binding(1) var<storage, read> wBuf: array<f32>;
@group(0) @binding(2) var<storage, read> bBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuf: array<f32>;
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= M) { return; }
  var sum = bBuf[o];
  for (var i = 0u; i < N; i = i + 1u) {
    sum = sum + inBuf[i] * wBuf[i * M + o];
  }
  outBuf[o] = sum;
}`;

export async function createWebGPUBackend(onDeviceLost) {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new BackendUnavailableError('当前环境不支持 WebGPU (navigator.gpu 不存在)');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new BackendUnavailableError('WebGPU 后端不可用: 无法获取 GPUAdapter');
  }
  let device;
  try {
    device = await adapter.requestDevice();
  } catch (e) {
    throw new BackendUnavailableError(`WebGPU 后端不可用: requestDevice 失败 (${e.message})`);
  }
  device.lost.then((info) => {
    if (onDeviceLost) onDeviceLost(info);
  });

  const makePipeline = (code, entryCount) => {
    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    return pipeline;
  };
  const convPipeline = makePipeline(CONV_SHADER);
  const poolPipeline = makePipeline(POOL_SHADER);
  const gemmPipeline = makePipeline(GEMM_SHADER);

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const makeBuffer = (size, usage) => device.createBuffer({ size, usage });

  const inBuf = makeBuffer(28 * 28 * 4, storage);
  const convWStorage = makeBuffer(4 * KERNEL * KERNEL * 4, storage);
  const convBStorage = makeBuffer(4 * 4, storage);
  const convOutBuf = makeBuffer(CONV_OUT_CHANNELS * CONV_H * CONV_W * 4, storage);
  const pooledBuf = makeBuffer(FLAT_SIZE * 4, storage);
  const fcWStorage = makeBuffer(FLAT_SIZE * NUM_CLASSES * 4, storage);
  const fcBStorage = makeBuffer(NUM_CLASSES * 4, storage);
  const logitsBuf = makeBuffer(NUM_CLASSES * 4, storage);
  const pooledStage = makeBuffer(FLAT_SIZE * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const logitsStage = makeBuffer(NUM_CLASSES * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

  let weightsUploaded = false;

  const bind = (pipeline, buffers) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });

  const convBind = bind(convPipeline, [inBuf, convWStorage, convBStorage, convOutBuf]);
  const poolBind = bind(poolPipeline, [convOutBuf, pooledBuf]);
  const gemmBind = bind(gemmPipeline, [pooledBuf, fcWStorage, fcBStorage, logitsBuf]);

  return {
    name: backendName,
    async infer(weights, input) {
      const q = device.queue;
      if (!weightsUploaded) {
        q.writeBuffer(convWStorage, 0, weights.convW);
        q.writeBuffer(convBStorage, 0, weights.convB);
        q.writeBuffer(fcWStorage, 0, weights.fcW);
        q.writeBuffer(fcBStorage, 0, weights.fcB);
        weightsUploaded = true;
      }
      q.writeBuffer(inBuf, 0, input);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(convPipeline);
      pass.setBindGroup(0, convBind);
      pass.dispatchWorkgroups(Math.ceil(CONV_W / 8), Math.ceil(CONV_H / 8), CONV_OUT_CHANNELS);
      pass.setPipeline(poolPipeline);
      pass.setBindGroup(0, poolBind);
      pass.dispatchWorkgroups(Math.ceil(POOLED_W / 8), Math.ceil(POOLED_H / 8), CONV_OUT_CHANNELS);
      pass.setPipeline(gemmPipeline);
      pass.setBindGroup(0, gemmBind);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(pooledBuf, 0, pooledStage, 0, FLAT_SIZE * 4);
      encoder.copyBufferToBuffer(logitsBuf, 0, logitsStage, 0, NUM_CLASSES * 4);
      q.submit([encoder.finish()]);

      await pooledStage.mapAsync(GPUMapMode.READ);
      const pooled = new Float32Array(FLAT_SIZE);
      pooled.set(new Float32Array(pooledStage.getMappedRange()));
      pooledStage.unmap();
      await logitsStage.mapAsync(GPUMapMode.READ);
      const logits = new Float32Array(NUM_CLASSES);
      logits.set(new Float32Array(logitsStage.getMappedRange()));
      logitsStage.unmap();

      return { probs: softmax(logits), pooled, logits };
    },
  };
}
