import json
from pathlib import Path

# Usage:
#   python make_visualizer.py mobilenet_timeline.json -o mobilenet_visualizer.html

MAX_TOOLTIP_ITEMS = 80  # store at most N items per timestamp per kind in TL_DATA

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MobileNet Weight-Timeline Visualizer</title>
  <style>
    :root{
      --bg:#0b0f14; --panel:#121924; --panel2:#0f1620; --text:#e7eef9; --muted:#9fb0c3;
      --accent:#5dd6ff; --dw:#7CFF6B; --pw:#FFB86B; --border:rgba(255,255,255,.10);
    }
    *{ box-sizing:border-box; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    body{ margin:0; background:var(--bg); color:var(--text); }
    header{
      position:sticky; top:0; z-index:10;
      background:linear-gradient(180deg, rgba(18,25,36,.95), rgba(18,25,36,.75));
      backdrop-filter: blur(10px);
      border-bottom:1px solid var(--border);
      padding:12px 16px;
      display:flex; gap:12px; align-items:center; flex-wrap:wrap;
    }
    header h1{ font-size:16px; margin:0; color:var(--text); letter-spacing:.2px; }
    .pill{ padding:6px 10px; border:1px solid var(--border); border-radius:999px; color:var(--muted); font-size:12px; }
    .pill b{ color:var(--text); font-weight:600; }
    .search{
      flex:1; min-width:220px;
      display:flex; align-items:center; gap:8px;
      padding:8px 10px; border:1px solid var(--border); border-radius:10px;
      background:rgba(0,0,0,.15);
    }
    .search input{
      width:100%; border:0; outline:none; background:transparent; color:var(--text);
      font-size:13px;
    }
    main{ display:grid; grid-template-columns: 380px 1fr; gap:12px; padding:12px; }
    @media (max-width: 980px){ main{ grid-template-columns: 1fr; } }

    .panel{
      background:var(--panel); border:1px solid var(--border); border-radius:14px;
      overflow:hidden;
    }
    .panel .hd{
      padding:10px 12px; border-bottom:1px solid var(--border);
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0));
    }
    .panel .hd .title{ font-size:13px; color:var(--text); font-weight:600; }
    .panel .bd{ padding:10px 12px; }

    .layers{
      display:grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
      gap:10px;
    }
    .layer-card{
      cursor:pointer; user-select:none;
      background:var(--panel2); border:1px solid var(--border); border-radius:12px;
      padding:10px;
      transition: transform .08s ease, border-color .08s ease;
      min-height:74px;
      display:flex; flex-direction:column; justify-content:space-between;
    }
    .layer-card:hover{ transform: translateY(-1px); border-color: rgba(93,214,255,.45); }
    .layer-top{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
    .layer-idx{ font-size:12px; color:var(--muted); }
    .tag{
      font-size:11px; padding:3px 7px; border-radius:999px; border:1px solid var(--border);
      color:var(--text);
    }
    .tag.dw{ border-color: rgba(124,255,107,.35); color: var(--dw); }
    .tag.pw{ border-color: rgba(255,184,107,.35); color: var(--pw); }
    .layer-name{ margin-top:6px; font-size:13px; color:var(--text); font-weight:600; }
    .layer-sub{ font-size:11px; color:var(--muted); margin-top:2px; }

    .active{ outline:2px solid rgba(93,214,255,.4); }

    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    button{
      background:rgba(0,0,0,.18); color:var(--text);
      border:1px solid var(--border); border-radius:10px;
      padding:8px 10px; font-size:13px; cursor:pointer;
    }
    button:hover{ border-color: rgba(93,214,255,.45); }
    .table{
      width:100%; border-collapse:collapse; margin-top:10px;
      font-size:13px;
    }
    .table th, .table td{
      padding:8px 10px; border-bottom:1px solid var(--border);
      text-align:left;
      vertical-align:top;
    }
    .table th{ color:var(--muted); font-weight:600; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .muted{ color:var(--muted); }
    .hint{ font-size:12px; color:var(--muted); margin-top:6px; line-height:1.4; }
    .badge{
      display:inline-block; padding:2px 8px; border-radius:999px;
      border:1px solid var(--border); font-size:12px;
      margin-left:6px;
    }
  </style>
</head>
<body>
<header>
  <h1>MobileNet Weight-Timeline Visualizer</h1>
  <div class="pill"><b id="layerCount"></b> layers</div>
  <div class="pill">Global timeline + per-layer details</div>
  <div class="search">
    <span class="muted" style="font-size:12px;">Filter</span>
    <input id="filter" placeholder="e.g. layer:7  type:dw  pointwise  depthwise" />
  </div>
</header>

<main>
  <section class="panel">
    <div class="hd">
      <div class="title">Layers</div>
      <div class="muted" style="font-size:12px;" id="filterInfo"></div>
    </div>
    <div class="bd">
      <div class="layers" id="layers"></div>
    </div>
  </section>

  <section class="panel">
    <div class="hd">
      <div class="title" id="detailTitle">Global Timeline</div>
      <div class="muted" style="font-size:12px;" id="detailMeta"></div>
    </div>

    <div class="bd" id="detailBody">
      <div class="row" style="gap:8px; margin-bottom:10px;">
        <button id="tabTimeline">Timeline</button>
        <button id="tabDetails">Details</button>
        <span class="muted" id="timelineRange" style="margin-left:auto;"></span>
      </div>

      <div id="timelineView">
        <div style="border:1px solid var(--border); border-radius:12px; overflow:hidden; background:rgba(0,0,0,.10);">
          <canvas id="timelineCanvas" height="240" style="width:100%; display:block;"></canvas>
        </div>
        <div class="hint">
          Wheel = zoom • Drag = pan • Hover = inspect. Top lanes show memory-load activity.
        </div>
        <div id="tooltip" style="
          position:fixed; display:none; pointer-events:none;
          background:rgba(10,14,20,.95); border:1px solid rgba(255,255,255,.15);
          color:var(--text); padding:8px 10px; border-radius:10px; max-width:520px;
          font-size:12px; line-height:1.35; z-index:9999;">
        </div>
      </div>

      <div id="detailsView" style="display:none;">
        <div class="hint">
          Click a layer on the left to view detailed Start/End/Δ tables.
        </div>
      </div>
    </div>
  </section>
</main>

<script>
// Full per-layer data (for Details view)
const DATA = __DATA__;

// Precomputed global timeline groups (fast)
const TL_DATA = __TL_DATA__;

const layersEl = document.getElementById('layers');
const layerCountEl = document.getElementById('layerCount');
const filterEl = document.getElementById('filter');
const filterInfoEl = document.getElementById('filterInfo');

const detailTitleEl = document.getElementById('detailTitle');
const detailMetaEl = document.getElementById('detailMeta');

const tabDetailsBtn = document.getElementById('tabDetails');
const tabTimelineBtn = document.getElementById('tabTimeline');
const detailsViewEl = document.getElementById('detailsView');
const timelineViewEl = document.getElementById('timelineView');
const timelineRangeEl = document.getElementById('timelineRange');

const canvas = document.getElementById('timelineCanvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

layerCountEl.textContent = Array.isArray(DATA) ? DATA.length : 0;

let activeLayer = null;

// ---------- Tabs ----------
function showDetails(){
  detailsViewEl.style.display = '';
  timelineViewEl.style.display = 'none';
  detailTitleEl.textContent = activeLayer ? `Layer ${activeLayer.layer}` : 'Layer Details';
}
function showTimeline(){
  detailsViewEl.style.display = 'none';
  timelineViewEl.style.display = '';
  detailTitleEl.textContent = 'Global Timeline';
  resizeCanvasToCSS();
  drawTimeline();
}
tabDetailsBtn.onclick = showDetails;
tabTimelineBtn.onclick = showTimeline;

// ---------- Layer list ----------
function normType(t){
  if (!t) return '';
  const x = String(t).toLowerCase();
  if (x.startsWith('d')) return 'dw';
  if (x.startsWith('p')) return 'pw';
  return x;
}

function summarizeLayer(layer){
  const r = layer.res ?? null;
  const resTxt = (r !== null && r !== undefined) ? `${r}×${r}` : '';
  let core = '';
  if (layer.type === "Depthwise"){
    const ch = layer.channels?.length ?? 0;
    core = `${ch} output channels`;
  } else if (layer.type === "Pointwise"){
    const och = layer.out_channels?.length ?? 0;
    const ich = layer.out_channels?.[0]?.in_channels?.length ?? 0;
    core = `${och} out × ${ich} in`;
  }
  if (resTxt && core) return `${resTxt} • ${core}`;
  if (resTxt) return resTxt;
  return core;
}

function matchesFilter(layer, q){
  q = q.trim().toLowerCase();
  if (!q) return true;

  const parts = q.split(/\s+/).filter(Boolean);
  for (const p of parts){
    if (p.startsWith("layer:")){
      const n = parseInt(p.slice(6), 10);
      if (Number.isFinite(n) && Number(layer.layer) !== n) return false;
      continue;
    }
    if (p.startsWith("type:")){
      const t = normType(p.slice(5));
      if (t === 'dw' && layer.type !== "Depthwise") return false;
      if (t === 'pw' && layer.type !== "Pointwise") return false;
      continue;
    }
    if (["dw","depthwise"].includes(p)){
      if (layer.type !== "Depthwise") return false;
      continue;
    }
    if (["pw","pointwise"].includes(p)){
      if (layer.type !== "Pointwise") return false;
      continue;
    }
    const hay = `${layer.layer} ${layer.type}`.toLowerCase();
    if (!hay.includes(p)) return false;
  }
  return true;
}

function renderLayers(){
  layersEl.innerHTML = '';
  const q = filterEl.value;
  const filtered = (Array.isArray(DATA) ? DATA : []).filter(l => matchesFilter(l, q));
  filterInfoEl.textContent = q.trim() ? `${filtered.length} shown` : '';

  for (const layer of filtered){
    const card = document.createElement('div');
    card.className = 'layer-card';
    if (activeLayer && Number(activeLayer.layer) === Number(layer.layer)) card.classList.add('active');

    const tagCls = layer.type === "Depthwise" ? 'dw' : (layer.type === "Pointwise" ? 'pw' : '');
    const tagTxt = layer.type === "Depthwise" ? 'Depthwise' : (layer.type === "Pointwise" ? 'Pointwise' : layer.type);

    card.innerHTML = `
      <div class="layer-top">
        <div class="layer-idx">Layer ${layer.layer}</div>
        <div class="tag ${tagCls}">${tagTxt}</div>
      </div>
      <div>
        <div class="layer-name">Layer ${layer.layer}</div>
        <div class="layer-sub">${summarizeLayer(layer)}</div>
      </div>
    `;
    card.onclick = () => { selectLayer(Number(layer.layer)); showDetails(); };
    layersEl.appendChild(card);
  }
}

filterEl.addEventListener('input', renderLayers);
renderLayers();

// ---------- Details view ----------
function fmtDur(s, e){
  const ss = Number(s), ee = Number(e);
  if (Number.isFinite(ss) && Number.isFinite(ee)) return String(ee - ss);
  return '';
}

function selectLayer(layerIdx){
  activeLayer = (Array.isArray(DATA) ? DATA : []).find(l => Number(l.layer) === Number(layerIdx));
  if (!activeLayer){
    detailsViewEl.innerHTML = `<div class="muted">Layer not found.</div>`;
    return;
  }
  detailTitleEl.textContent = `Layer ${activeLayer.layer}`;
  const r = activeLayer.res ?? null;
  detailMetaEl.textContent = r ? `${activeLayer.type} • ${r}×${r}` : activeLayer.type;

  if (activeLayer.type === "Depthwise"){
    const channels = activeLayer.channels || [];
    const flat = [];
    for (const ch of channels){
      const out = Number(ch.out_ch);
      const starts = Array.isArray(ch.starts) ? ch.starts : (ch.starts !== undefined ? [ch.starts] : []);
      const ends   = Array.isArray(ch.ends)   ? ch.ends   : (ch.ends   !== undefined ? [ch.ends]   : []);
      const n = Math.min(starts.length, ends.length);
      for (let k = 0; k < n; k++) flat.push({ out_ch: out, k_idx: k, start: starts[k], end: ends[k] });
    }
    let maxEventsPerChannel = 1;
    for (const ch of channels){
      const s = Array.isArray(ch.starts) ? ch.starts.length : (ch.starts !== undefined ? 1 : 0);
      const e = Array.isArray(ch.ends)   ? ch.ends.length   : (ch.ends   !== undefined ? 1 : 0);
      const n = Math.min(s, e);
      if (n > maxEventsPerChannel) maxEventsPerChannel = n;
    }
    flat.sort((a,b) => Number(a.start) - Number(b.start));
    let rows = "";
    for (const it of flat){
      const globalW = it.out_ch * maxEventsPerChannel + it.k_idx;
      rows += `<tr>
        <td class="mono">Weight ${globalW}</td>
        <td class="mono">${it.start}</td>
        <td class="mono">${it.end}</td>
        <td class="mono">${fmtDur(it.start, it.end)}</td>
      </tr>`;
    }
    detailsViewEl.innerHTML = `
      <div class="row">
        <span class="badge">Depthwise</span>
        <span class="muted">Flattened (out=in). Sorted by start time.</span>
      </div>
      <table class="table">
        <thead><tr><th>Weight</th><th>Start</th><th>End</th><th>Δ</th></tr></thead>
        <tbody>${rows || `<tr><td class="muted" colspan="4">No data.</td></tr>`}</tbody>
      </table>
    `;
    return;
  }

  if (activeLayer.type === "Pointwise"){
    const outs = activeLayer.out_channels || [];
    if (!outs.length){
      detailsViewEl.innerHTML = `<div class="muted">No output channels.</div>`;
      return;
    }
    let html = `
      <div class="row">
        <span class="badge">Pointwise</span>
        <span class="muted">Expand an output channel to see input-channel Start/End/Δ.</span>
      </div>
    `;
    for (const o of outs){
      const outCh = Number(o.out_ch);
      const ins = o.in_channels || [];
      let rows = "";
      for (const ic of ins){
        const inCh = Number(ic.in_ch);
        const starts = Array.isArray(ic.starts) ? ic.starts : (ic.starts !== undefined ? [ic.starts] : []);
        const ends   = Array.isArray(ic.ends)   ? ic.ends   : (ic.ends   !== undefined ? [ic.ends]   : []);
        const n = Math.min(starts.length, ends.length);
        let cell = '<span class="muted">—</span>';
        if (n > 0){
          const parts = [];
          for (let k = 0; k < n; k++){
            parts.push(`${starts[k]}→${ends[k]} (Δ${fmtDur(starts[k], ends[k])})`);
          }
          cell = `<span class="mono">${parts.join(", ")}</span>`;
        }
        rows += `<tr><td class="mono">${inCh}</td><td>${cell}</td></tr>`;
      }
      html += `
        <details style="margin-top:10px; border:1px solid var(--border); border-radius:12px; background:rgba(0,0,0,.10);">
          <summary style="cursor:pointer; padding:10px 12px; color:var(--text);">
            <span class="mono">Output Channel ${outCh}</span>
            <span class="muted"> (${ins.length} inputs)</span>
          </summary>
          <div style="padding:0 12px 10px 12px;">
            <table class="table">
              <thead><tr><th>Prev input channel</th><th>Start→End (Δ)</th></tr></thead>
              <tbody>${rows || `<tr><td class="muted" colspan="2">No inputs.</td></tr>`}</tbody>
            </table>
          </div>
        </details>
      `;
    }
    detailsViewEl.innerHTML = html;
    return;
  }

  detailsViewEl.innerHTML = '<div class="muted">Unknown layer type.</div>';
}

// ---------- GLOBAL timeline (fast) ----------
let TL = {
  groups: TL_DATA,
  times: TL_DATA.map(g => g.t),
  minT: TL_DATA.length ? TL_DATA[0].t : 0,
  maxT: TL_DATA.length ? TL_DATA[TL_DATA.length-1].t : 1,
  viewA: 0,
  viewB: 1,
  dragging: false,
  dragX0: 0,
  dragA0: 0,
};

(function initRange(){
  const pad = Math.max(50, (TL.maxT - TL.minT) * 0.02);
  TL.viewA = TL.minT - pad;
  TL.viewB = TL.maxT + pad;
})();

function resizeCanvasToCSS(){
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(240 * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener('resize', () => {
  if (timelineViewEl.style.display !== 'none'){
    resizeCanvasToCSS();
    drawTimeline();
  }
});

function tToX(t, w){
  return (t - TL.viewA) * (w / (TL.viewB - TL.viewA));
}
function xToT(x, w){
  return TL.viewA + x * ((TL.viewB - TL.viewA) / w);
}

// lower/upper bounds for visible range
function lowerBound(arr, x){
  let lo = 0, hi = arr.length;
  while (lo < hi){
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
function upperBound(arr, x){
  let lo = 0, hi = arr.length;
  while (lo < hi){
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function chooseTick(span){
  const targets = [50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000,10000000];
  const px = canvas.getBoundingClientRect().width;
  const desired = span / Math.max(1, px / 120);
  for (const t of targets) if (t >= desired) return t;
  return targets[targets.length-1];
}

function drawTimeline(){
  const w = canvas.getBoundingClientRect().width;
  const h = 240;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = 'rgba(255,255,255,0.02)';
  ctx.fillRect(0,0,w,h);

  const laneLoadDW_Y = 40;
  const laneLoadPW_Y = 70;
  const laneEvents_Y = 150;

  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.beginPath(); ctx.moveTo(0, laneEvents_Y); ctx.lineTo(w, laneEvents_Y); ctx.stroke();

  if (timelineRangeEl){
    timelineRangeEl.textContent = `${Math.round(TL.viewA)} → ${Math.round(TL.viewB)} (clk)`;
  }

  const i0 = lowerBound(TL.times, TL.viewA);
  const i1 = upperBound(TL.times, TL.viewB);

  // draw loads + points only in visible window
  function drawLoad(g, y, color){
    const x1 = tToX(g.loadStart, w);
    const x2 = tToX(g.loadEnd, w);
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
    ctx.lineWidth = 1;
  }

  for (let i = i0; i < i1; i++){
    const g = TL.groups[i];
    if (g.kind === 'DW') drawLoad(g, laneLoadDW_Y, 'rgba(124,255,107,0.45)');
    else drawLoad(g, laneLoadPW_Y, 'rgba(255,184,107,0.45)');
  }

  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '12px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('LOAD (DW)', 10, laneLoadDW_Y - 10);
  ctx.fillText('LOAD (PW)', 10, laneLoadPW_Y - 10);
  ctx.fillText('WEIGHT STARTS', 10, laneEvents_Y - 10);

  for (let i = i0; i < i1; i++){
    const g = TL.groups[i];
    const x = tToX(g.t, w);
    const r = (g.count >= 32) ? 6 : (g.count >= 8 ? 5 : 4);
    ctx.fillStyle = (g.kind === 'DW') ? 'rgba(124,255,107,0.9)' : 'rgba(255,184,107,0.9)';
    ctx.beginPath(); ctx.arc(x, laneEvents_Y, r, 0, Math.PI*2); ctx.fill();

    if (g.count > 1){
      ctx.fillStyle = 'rgba(0,0,0,0.85)';
      ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
      ctx.fillText(String(g.count), x - 3, laneEvents_Y + 3);
    }
  }

  // ticks
  const span = TL.viewB - TL.viewA;
  const tick = chooseTick(span);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';

  const first = Math.floor(TL.viewA / tick) * tick;
  for (let t = first; t <= TL.viewB; t += tick){
    const x = tToX(t, w);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(String(Math.round(t)), x + 4, h - 8);
  }
}

function hideTip(){ tooltip.style.display = "none"; }
function showTip(x, y, g){
  const lines = [];
  lines.push(`<div class="mono">t = ${g.t}</div>`);
  if (g.kind === 'DW'){
    lines.push(`<div><b>Depthwise</b> (count ${g.count})</div>`);
    for (const it of g.items) lines.push(`<div class="mono">L${it.layer} • ch ${it.ch}</div>`);
  } else {
    lines.push(`<div><b>Pointwise</b> (count ${g.count})</div>`);
    for (const it of g.items) lines.push(`<div class="mono">L${it.layer} • out ${it.outCh} • prev_in ${it.prevInCh}</div>`);
  }
  if (g.more && g.more > 0) lines.push(`<div class="muted">… +${g.more} more</div>`);
  lines.push(`<div class="muted">Load: [${Math.round(g.loadStart)}, ${Math.round(g.loadEnd)})</div>`);
  tooltip.innerHTML = lines.join("");
  tooltip.style.left = (x + 12) + "px";
  tooltip.style.top = (y + 12) + "px";
  tooltip.style.display = "block";
}

function nearestIndex(arr, t){
  let lo = 0, hi = arr.length - 1;
  if (hi < 0) return -1;
  if (t <= arr[0]) return 0;
  if (t >= arr[hi]) return hi;
  while (hi - lo > 1){
    const mid = (lo + hi) >> 1;
    if (arr[mid] < t) lo = mid; else hi = mid;
  }
  return (Math.abs(arr[lo]-t) <= Math.abs(arr[hi]-t)) ? lo : hi;
}

// zoom/pan/hover
canvas.addEventListener('wheel', (ev) => {
  if (timelineViewEl.style.display === 'none') return;
  ev.preventDefault();

  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const w = rect.width;

  const tCenter = xToT(x, w);
  const span = TL.viewB - TL.viewA;

  const z = Math.exp(-ev.deltaY * 0.001);
  let newSpan = span / z;
  newSpan = Math.max(200, Math.min(newSpan, (TL.maxT - TL.minT) * 5 + 1000));

  const frac = (tCenter - TL.viewA) / span;
  TL.viewA = tCenter - frac * newSpan;
  TL.viewB = TL.viewA + newSpan;

  drawTimeline();
}, { passive:false });

canvas.addEventListener('mousedown', (ev) => {
  if (timelineViewEl.style.display === 'none') return;
  TL.dragging = true;
  TL.dragX0 = ev.clientX;
  TL.dragA0 = TL.viewA;
});

window.addEventListener('mouseup', () => { TL.dragging = false; });

window.addEventListener('mousemove', (ev) => {
  if (timelineViewEl.style.display === 'none') return;

  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const w = rect.width;

  if (TL.dragging){
    const dx = ev.clientX - TL.dragX0;
    const dt = dx * ((TL.viewB - TL.viewA) / w);
    TL.viewA = TL.dragA0 - dt;
    TL.viewB = TL.viewA + (TL.viewB - TL.viewA);
    hideTip();
    drawTimeline();
    return;
  }

  const t = xToT(x, w);
  const idx = nearestIndex(TL.times, t);
  if (idx < 0){ hideTip(); return; }
  const g = TL.groups[idx];
  const px = tToX(g.t, w);
  if (Math.abs(px - x) <= 8) showTip(ev.clientX, ev.clientY, g);
  else hideTip();
});

// start on timeline by default
showTimeline();
</script>
</body>
</html>
"""

def build_tl_data(data):
    # group by exact start time, separated by kind
    dw = {}  # t -> {"t":t,"kind":"DW","count":N,"items":[...], "more":M}
    pw = {}

    def add_dw(t, layer, ch):
        if t is None:
            return
        try:
            t = int(t)
        except Exception:
            return
        g = dw.get(t)
        if g is None:
            g = {"t": t, "kind": "DW", "count": 0, "items": [], "more": 0}
            dw[t] = g
        g["count"] += 1
        if len(g["items"]) < MAX_TOOLTIP_ITEMS:
            g["items"].append({"layer": int(layer), "ch": int(ch)})
        else:
            g["more"] += 1

    def add_pw(t, layer, out_ch, prev_in):
        if t is None:
            return
        try:
            t = int(t)
        except Exception:
            return
        g = pw.get(t)
        if g is None:
            g = {"t": t, "kind": "PW", "count": 0, "items": [], "more": 0}
            pw[t] = g
        g["count"] += 1
        if len(g["items"]) < MAX_TOOLTIP_ITEMS:
            g["items"].append({"layer": int(layer), "outCh": int(out_ch), "prevInCh": int(prev_in)})
        else:
            g["more"] += 1

    for layer in data:
        L = layer.get("layer")
        typ = layer.get("type")
        if typ == "Depthwise":
            for ch in layer.get("channels", []):
                out = ch.get("out_ch")
                starts = ch.get("starts", [])
                if not isinstance(starts, list):
                    starts = [starts]
                for t in starts:
                    add_dw(t, L, out)
        elif typ == "Pointwise":
            for o in layer.get("out_channels", []):
                outCh = o.get("out_ch")
                for ic in o.get("in_channels", []):
                    prevIn = ic.get("in_ch")
                    starts = ic.get("starts", [])
                    if not isinstance(starts, list):
                        starts = [starts]
                    for t in starts:
                        add_pw(t, L, outCh, prevIn)

    groups = []

    for t, g in dw.items():
        g["loadStart"] = t - 9
        g["loadEnd"] = t
        groups.append(g)

    for t, g in pw.items():
        # load is 32 cycles per weight in this start-time group
        g["loadStart"] = t - 32 * int(g["count"])
        g["loadEnd"] = t
        groups.append(g)

    groups.sort(key=lambda x: x["t"])
    return groups


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="mobilenet_timeline.json")
    ap.add_argument("-o", "--out", default="mobilenet_visualizer.html")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of layers.")

    tl_data = build_tl_data(data)

    html_out = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    html_out = html_out.replace("__TL_DATA__", json.dumps(tl_data))
    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out} (timeline groups: {len(tl_data)})")

if __name__ == "__main__":
    main()
