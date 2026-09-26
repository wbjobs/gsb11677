import {
  drawHeatmap,
  drawFeatureMaps,
  drawProbBars,
  drawPerfChart,
  startSpinner,
} from './viz.js';

const $ = (id) => document.getElementById(id);
const worker = new Worker('./js/worker.js', { type: 'module' });

const logPanel = $('log');
function log(level, text) {
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  const line = document.createElement('div');
  line.className = `log-line log-${level}`;
  line.textContent = `[${time}] [${level}] ${text}`;
  logPanel.appendChild(line);
  logPanel.scrollTop = logPanel.scrollHeight;
}

// ---- Main-thread responsiveness monitors ----
let longTaskCount = 0;
let longTaskMax = 0;
try {
  const longTaskObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      longTaskCount++;
      longTaskMax = Math.max(longTaskMax, entry.duration);
      $('longtask').textContent = `${longTaskCount} 个 (最长 ${longTaskMax.toFixed(0)} ms)`;
      log('warn', `检测到长任务: ${entry.duration.toFixed(1)} ms`);
    }
  });
  longTaskObserver.observe({ entryTypes: ['longtask'] });
} catch (_) {
  $('longtask').textContent = '不支持 longtask 观察';
}

let frames = 0;
setInterval(() => {
  $('fps').textContent = `${frames} FPS`;
  frames = 0;
}, 1000);
(function countFrames() {
  frames++;
  requestAnimationFrame(countFrames);
})();
startSpinner($('spinner'));

// ---- Worker messaging ----
worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'capabilities': {
      setBadge('cap-webnn', msg.webnn);
      setBadge('cap-webgpu', msg.webgpu);
      setBadge('cap-worker', msg.worker);
      log(
        'info',
        `环境检测: WebNN=${msg.webnn ? '支持' : '不支持'}, WebGPU=${msg.webgpu ? '支持' : '不支持'}`
      );
      if (!msg.webnn) log('warn', 'WebNN 不可用，运行时将自动降级到 WebGPU/JS');
      break;
    }
    case 'input':
      drawHeatmap($('inputCanvas'), msg.input, 28, 28, 'input 28x28');
      break;
    case 'retry':
      log('warn', `后端 ${msg.backend} 第 ${msg.attempt}/${msg.maxRetries} 次重试: ${msg.error}`);
      break;
    case 'fallback':
      log('warn', `后端 ${msg.from} 失败 [${msg.kind}]，降级到下一个后端。原因: ${msg.reason}`);
      break;
    case 'backend-lost':
      log('error', `WebGPU 设备丢失: ${msg.reason}，后续将重建或降级`);
      break;
    case 'result': {
      const s = msg.stats;
      log(
        'success',
        `推理完成 (后端=${msg.backend}${msg.backend !== msg.requested ? ', 由 ' + msg.requested + ' 降级' : ''}): ` +
          `avg=${s.avg.toFixed(3)}ms min=${s.min.toFixed(3)}ms p95=${s.p95.toFixed(3)}ms (${s.iterations} 次)`
      );
      const argmax = msg.probs.indexOf(Math.max(...msg.probs));
      $('prediction').textContent = `预测类别: ${argmax} (p=${msg.probs[argmax].toFixed(3)}, 后端=${msg.backend})`;
      drawFeatureMaps($('mapsCanvas'), msg.pooled, 4, 14, 14);
      drawProbBars($('probsCanvas'), msg.probs);
      break;
    }
    case 'run-error':
      log('error', `[${msg.kind}] ${msg.error}`);
      break;
    case 'benchmark-progress':
      log('info', `性能对照: ${msg.backend} ${msg.state === 'start' ? '开始...' : msg.state === 'done' ? '完成' : '失败: ' + msg.error}`);
      break;
    case 'benchmark-result': {
      drawPerfChart($('perfCanvas'), msg.results);
      for (const r of msg.results) {
        if (r.stats) {
          log(
            'success',
            `${r.backend}: avg=${r.stats.avg.toFixed(3)}ms min=${r.stats.min.toFixed(3)}ms ` +
              `p95=${r.stats.p95.toFixed(3)}ms | 与 JS 参考最大误差=${r.maxDiffVsJs.toExponential(2)} ` +
              `${r.consistent ? '(结果一致 ✓)' : '(结果不一致 ✗)'}`
          );
        } else {
          log('warn', `${r.backend}: 不可用 [${r.kind}] ${r.error}`);
        }
      }
      break;
    }
    case 'probe-result': {
      const unsupported = msg.results.filter((r) => !r.supported);
      for (const r of msg.results) {
        log(r.supported ? 'success' : 'warn', `算子 ${r.op}: ${r.supported ? '支持' : '不支持 - ' + r.error}`);
      }
      log(
        'info',
        `算子探测完成: ${msg.results.length - unsupported.length}/${msg.results.length} 支持` +
          (unsupported.length ? `，不支持: ${unsupported.map((r) => r.op).join(', ')}` : '')
      );
      break;
    }
    case 'probe-error':
      log('error', `算子探测失败: ${msg.error}`);
      break;
    case 'shape-test-result':
      if (msg.ok) log('success', `形状不匹配已被捕获 [${msg.kind}]: ${msg.error} (${msg.note})`);
      else log('error', `形状测试异常: ${msg.error}`);
      break;
    case 'measures':
      // performance.measure entries forwarded from the worker's PerformanceObserver
      if (msg.entries.length) {
        const last = msg.entries[msg.entries.length - 1];
        $('lastMeasure').textContent = `${last.name}: ${last.duration.toFixed(3)} ms`;
      }
      break;
  }
};

function setBadge(id, ok) {
  const el = $(id);
  el.textContent = ok ? '可用' : '不可用';
  el.className = `badge ${ok ? 'badge-ok' : 'badge-bad'}`;
}

function currentOptions() {
  return {
    backend: $('backendSelect').value,
    deviceType: $('deviceSelect').value,
    iterations: Math.max(1, parseInt($('itersInput').value, 10) || 50),
    injectFailures: $('injectChk').checked ? 2 : 0,
  };
}

$('runBtn').onclick = () => {
  const o = currentOptions();
  log('info', `请求推理: backend=${o.backend}, device=${o.deviceType}, iterations=${o.iterations}` +
    (o.injectFailures ? ', 已注入 2 次模拟失败' : ''));
  worker.postMessage({ type: 'run', ...o });
};

$('benchBtn').onclick = () => {
  const o = currentOptions();
  log('info', `开始性能对照 (每后端 ${o.iterations} 次迭代)...`);
  worker.postMessage({ type: 'benchmark', iterations: o.iterations, deviceType: o.deviceType });
};

$('probeBtn').onclick = () => {
  log('info', '探测 WebNN 算子支持情况...');
  worker.postMessage({ type: 'probeOps', deviceType: $('deviceSelect').value });
};

$('shapeBtn').onclick = () => {
  const o = currentOptions();
  log('info', '构造错误形状输入 [1,1,14,14] 以验证形状校验...');
  worker.postMessage({ type: 'shapeTest', backend: o.backend, deviceType: o.deviceType });
};

// Boot
worker.postMessage({ type: 'capabilities' });
worker.postMessage({ type: 'getInput' });
log('info', 'Demo 已启动，推理在 Web Worker 中执行，主线程保持响应');
