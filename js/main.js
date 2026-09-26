import { createModel } from './model.js';

const worker = new Worker('./js/worker.js', { type: 'module' });
const model = createModel(42); // same seed as worker, used to draw the input

const els = {
  backendSelect: document.getElementById('backend-select'),
  runBtn: document.getElementById('run-btn'),
  benchBtn: document.getElementById('bench-btn'),
  probeBtn: document.getElementById('probe-btn'),
  badShapeBtn: document.getElementById('badshape-btn'),
  failToggle: document.getElementById('fail-toggle'),
  iterations: document.getElementById('iterations'),
  status: document.getElementById('status'),
  log: document.getElementById('log'),
  longtask: document.getElementById('longtask-status'),
  perfTable: document.getElementById('perf-table'),
  inputCanvas: document.getElementById('input-canvas'),
  featureCanvas: document.getElementById('feature-canvas'),
  outputCanvas: document.getElementById('output-canvas'),
  benchCanvas: document.getElementById('bench-canvas'),
  resultInfo: document.getElementById('result-info'),
};

let msgId = 0;
const pending = new Map();

function send(msg) {
  return new Promise((resolve, reject) => {
    const id = ++msgId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id });
  });
}

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'log') {
    appendLog(msg.level, msg.message);
    return;
  }
  if (msg.type === 'perf') {
    recordPerf(msg.entry);
    return;
  }
  const handler = pending.get(msg.id);
  if (!handler) return;
  pending.delete(msg.id);
  if (msg.type === 'error') {
    handler.reject(msg.error);
  } else {
    handler.resolve(msg);
  }
};

worker.onerror = (err) => {
  appendLog('error', `Worker 错误: ${err.message}`);
};

// ---------- main-thread responsiveness monitoring ----------
let longtaskCount = 0;
try {
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      longtaskCount++;
      els.longtask.textContent =
        `主线程长任务: ${longtaskCount} 次 (最近 ${entry.duration.toFixed(1)}ms)`;
      els.longtask.className = 'warn';
    }
  }).observe({ entryTypes: ['longtask'] });
} catch {
  els.longtask.textContent = '主线程长任务监控: 此浏览器不支持 longtask';
}

// ---------- perf entries from worker ----------
const perfRows = [];
function recordPerf(entry) {
  perfRows.unshift(entry);
  if (perfRows.length > 12) perfRows.pop();
  els.perfTable.innerHTML = perfRows
    .map((e) => `<tr><td>${e.name}</td><td>${e.duration.toFixed(3)} ms</td></tr>`)
    .join('');
}

// ---------- logging ----------
function appendLog(level, message) {
  const line = document.createElement('div');
  line.className = `log-${level}`;
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  line.textContent = `[${time}] [${level.toUpperCase()}] ${message}`;
  els.log.prepend(line);
}

function setStatus(text, cls = '') {
  els.status.textContent = text;
  els.status.className = cls;
}

// ---------- canvas helpers ----------
function drawHeatmap(ctx, data, shape, x0, y0, scale, label) {
  const [, , h, w] = shape.length === 4 ? shape : [1, 1, shape[0], shape[1]];
  let min = Infinity, max = -Infinity;
  for (const v of data) { if (v < min) min = v; if (v > max) max = v; }
  const range = max - min || 1;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const t = (data[i] - min) / range;
    img.data[i * 4] = Math.round(30 + 225 * t);
    img.data[i * 4 + 1] = Math.round(40 + 160 * t);
    img.data[i * 4 + 2] = Math.round(120 + 135 * (1 - t));
    img.data[i * 4 + 3] = 255;
  }
  offCtx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, w * scale, h * scale);
  if (label) {
    ctx.fillStyle = '#9aa4b2';
    ctx.font = '10px monospace';
    ctx.fillText(label, x0, y0 + h * scale + 11);
  }
}

function drawFeatureMaps(canvas, intermediates, shapes) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#11151c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const groups = [
    { key: 'conv1', shape: shapes.conv1Out, scale: 3, title: 'conv1+relu' },
    { key: 'pool1', shape: shapes.pool1Out, scale: 3, title: 'maxPool' },
    { key: 'conv2', shape: shapes.conv2Out, scale: 3, title: 'conv2+relu' },
    { key: 'pool2', shape: shapes.pool2Out, scale: 6, title: 'avgPool' },
  ];
  let y = 8;
  ctx.font = '12px sans-serif';
  for (const g of groups) {
    const [n, c, h, w] = g.shape;
    const channelSize = h * w;
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(`${g.title}  [${g.shape}]`, 8, y + 10);
    let x = 8;
    const rowY = y + 16;
    for (let ch = 0; ch < c; ch++) {
      const slice = intermediates[g.key].subarray(ch * channelSize, (ch + 1) * channelSize);
      drawHeatmap(ctx, slice, [1, 1, h, w], x, rowY, g.scale, `ch${ch}`);
      x += w * g.scale + 10;
    }
    y = rowY + h * g.scale + 22;
  }
}

function drawOutputBars(canvas, output) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#11151c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const max = Math.max(...output.map(Math.abs), 1e-6);
  const barW = canvas.width / output.length;
  const midY = canvas.height / 2;
  const pred = output.indexOf(Math.max(...output));
  for (let i = 0; i < output.length; i++) {
    const h = (output[i] / max) * (canvas.height / 2 - 16);
    ctx.fillStyle = i === pred ? '#4cc2ff' : '#3a7ca5';
    ctx.fillRect(i * barW + 4, h >= 0 ? midY - h : midY, barW - 8, Math.abs(h) || 1);
    ctx.fillStyle = '#9aa4b2';
    ctx.font = '11px monospace';
    ctx.fillText(String(i), i * barW + barW / 2 - 3, canvas.height - 4);
  }
  ctx.strokeStyle = '#333c4a';
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(canvas.width, midY);
  ctx.stroke();
  return pred;
}

function drawBenchmarkChart(canvas, rows) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#11151c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const valid = rows.filter((r) => !r.unavailable);
  if (!valid.length) return;
  const maxAvg = Math.max(...valid.map((r) => r.avg));
  const barH = 34;
  const gap = 16;
  const labelW = 130;
  ctx.font = '12px sans-serif';
  rows.forEach((row, i) => {
    const y = 14 + i * (barH + gap);
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(row.label, 8, y + barH / 2 + 4);
    if (row.unavailable) {
      ctx.fillStyle = '#b06a6a';
      ctx.fillText(`不可用: ${row.unavailable}`, labelW, y + barH / 2 + 4);
      return;
    }
    const w = (row.avg / maxAvg) * (canvas.width - labelW - 130);
    const colors = { webnn: '#7c5cff', webgpu: '#4cc2ff', js: '#ffb454' };
    ctx.fillStyle = colors[row.backend] ?? '#888';
    ctx.fillRect(labelW, y, Math.max(w, 2), barH);
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(
      `avg ${row.avg.toFixed(3)}ms | p50 ${row.p50.toFixed(3)} | min ${row.min.toFixed(3)} | max ${row.max.toFixed(3)}`,
      labelW + Math.max(w, 2) + 8, y + barH / 2 + 4);
  });
}

// ---------- actions ----------
async function runInference(options = {}) {
  const backend = els.backendSelect.value;
  setStatus('推理中…');
  els.runBtn.disabled = true;
  try {
    const msg = await send({
      type: 'run',
      backend,
      simulateFailure: els.failToggle.checked,
      badShape: options.badShape ?? false,
    });
    const p = msg.payload;
    for (const f of p.fallbacks) {
      appendLog('warn', `降级: ${f.from} → ${f.reason}`);
    }
    const consistencyText = p.consistency
      ? (p.consistency.ok
        ? `与 JS 参考一致 (maxDiff=${p.consistency.maxDiff.toExponential(2)})`
        : `与 JS 参考不一致! maxDiff=${p.consistency.maxDiff}`)
      : '本后端即 JS 参考';
    els.resultInfo.textContent =
      `后端: ${p.backendLabel} | 尝试次数: ${p.attempts} | ${consistencyText}`;
    setStatus(`推理完成 (${p.backendLabel})`, 'ok');
    drawFeatureMaps(els.featureCanvas, p.intermediates, p.shapes);
    const pred = drawOutputBars(els.outputCanvas, p.output);
    appendLog('info', `输出 [${p.outputShape}]，argmax=${pred}，${consistencyText}`);
  } catch (err) {
    if (err.isShapeMismatch) {
      setStatus('形状不匹配（已捕获）', 'error');
      appendLog('error', `ShapeMismatch 已被捕获: ${err.message}`);
    } else {
      setStatus('推理失败', 'error');
      appendLog('error', `${err.name}: ${err.message}`);
    }
  } finally {
    els.runBtn.disabled = false;
  }
}

async function runBenchmark() {
  setStatus('基准测试中…');
  els.benchBtn.disabled = true;
  try {
    const msg = await send({ type: 'benchmark', iterations: Number(els.iterations.value) || 50 });
    drawBenchmarkChart(els.benchCanvas, msg.rows);
    setStatus('基准测试完成', 'ok');
  } catch (err) {
    setStatus('基准测试失败', 'error');
    appendLog('error', `${err.name}: ${err.message}`);
  } finally {
    els.benchBtn.disabled = false;
  }
}

async function probeOps() {
  setStatus('探测 WebNN 算子支持…');
  try {
    const msg = await send({ type: 'probe' });
    for (const [op, info] of Object.entries(msg.results)) {
      if (info.supported) {
        appendLog('info', `算子 ${op}: 支持`);
      } else {
        appendLog('warn', `算子 ${op}: 不支持 — ${info.reason}`);
      }
    }
    setStatus('算子探测完成', 'ok');
  } catch (err) {
    setStatus('算子探测失败', 'error');
    appendLog('error', `${err.name}: ${err.message}`);
  }
}

// ---------- init ----------
function drawInput() {
  const ctx = els.inputCanvas.getContext('2d');
  drawHeatmap(ctx, model.input, model.spec.inputShape, 0, 0,
    els.inputCanvas.width / model.spec.inputShape[3]);
}

els.runBtn.addEventListener('click', () => runInference());
els.benchBtn.addEventListener('click', runBenchmark);
els.probeBtn.addEventListener('click', probeOps);
els.badShapeBtn.addEventListener('click', () => runInference({ badShape: true }));

drawInput();
setStatus('就绪');
appendLog('info', '页面已加载，推理在 Web Worker 中执行，主线程保持响应');
