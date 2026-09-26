// Canvas visualization helpers (all rendering happens on the main thread).

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

// Inferno-like color map: t in [0,1] -> css rgb string.
function heatColor(t) {
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, c * 2.2));
  const g = Math.round(255 * Math.max(0, Math.min(1, (c - 0.25) * 1.8)));
  const b = Math.round(255 * Math.max(0, Math.min(1, (c - 0.6) * 2.2)));
  return `rgb(${r},${g},${b})`;
}

export function drawHeatmap(canvas, data, rows, cols, label) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  let max = 1e-8;
  for (const v of data) max = Math.max(max, Math.abs(v));
  const cw = w / cols;
  const ch = h / rows;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      ctx.fillStyle = heatColor(data[y * cols + x] / max);
      ctx.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
    }
  }
  if (label) {
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '11px system-ui';
    ctx.fillText(label, 6, 14);
  }
}

export function drawFeatureMaps(canvas, pooled, channels, rows, cols) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  let max = 1e-8;
  for (const v of pooled) max = Math.max(max, Math.abs(v));
  const gap = 6;
  const cellW = (w - gap * (channels + 1)) / channels;
  const cellH = h - gap * 2 - 16;
  const size = Math.min(cellW, cellH);
  for (let c = 0; c < channels; c++) {
    const ox = gap + c * (cellW + gap);
    const oy = gap + 16;
    const cw = size / cols;
    const ch = size / rows;
    const base = c * rows * cols;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        ctx.fillStyle = heatColor(pooled[base + y * cols + x] / max);
        ctx.fillRect(ox + x * cw, oy + y * ch, cw + 0.5, ch + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.strokeRect(ox, oy, size, size);
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = '11px system-ui';
    ctx.fillText(`ch${c}`, ox, 12);
  }
}

export function drawProbBars(canvas, probs, highlight = true) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const n = probs.length;
  const gap = 8;
  const barW = (w - gap * (n + 1)) / n;
  const labelH = 18;
  let argmax = 0;
  for (let i = 1; i < n; i++) if (probs[i] > probs[argmax]) argmax = i;
  for (let i = 0; i < n; i++) {
    const p = probs[i];
    const bh = (h - labelH - 8) * p;
    const x = gap + i * (barW + gap);
    const y = h - labelH - bh;
    ctx.fillStyle = highlight && i === argmax ? '#4ade80' : '#60a5fa';
    ctx.fillRect(x, y, barW, bh);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(String(i), x + barW / 2, h - 5);
    ctx.fillText(p.toFixed(2), x + barW / 2, Math.max(10, y - 4));
  }
  ctx.textAlign = 'left';
}

export function drawPerfChart(canvas, results) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const ok = results.filter((r) => r.stats);
  const labelH = 20;
  const topPad = 26;
  if (ok.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '12px system-ui';
    ctx.fillText('暂无可量化的性能数据', 10, h / 2);
    return;
  }
  const maxAvg = Math.max(...ok.map((r) => r.stats.avg));
  const gap = 14;
  const barW = Math.min(90, (w - gap * (ok.length + 1)) / ok.length);
  const colors = { webnn: '#c084fc', webgpu: '#4ade80', js: '#fbbf24' };
  ok.forEach((r, i) => {
    const x = gap + i * (barW + gap);
    const bh = ((h - labelH - topPad) * r.stats.avg) / maxAvg;
    const y = h - labelH - bh;
    ctx.fillStyle = colors[r.backend] || '#60a5fa';
    ctx.fillRect(x, y, barW, bh);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(r.backend, x + barW / 2, h - 6);
    ctx.fillText(`${r.stats.avg.toFixed(3)} ms`, x + barW / 2, y - 5);
  });
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '11px system-ui';
  ctx.fillText('平均单次推理耗时 (越低越好)', 4, 14);
}

// Small always-on animation proving the main thread stays responsive.
export function startSpinner(canvas) {
  const tick = () => {
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const t = performance.now() / 1000;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) / 2 - 4;
    for (let i = 0; i < 8; i++) {
      const a = t * 2 + (i * Math.PI) / 4;
      ctx.beginPath();
      ctx.arc(cx + r * Math.cos(a) * 0.7, cy + r * Math.sin(a) * 0.7, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(96,165,250,${0.25 + 0.75 * ((i / 8 + t) % 1)})`;
      ctx.fill();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
