import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
SNAPSHOTS_DIR = ROOT / "snapshots"
HOST = "127.0.0.1"
PORT = 8765


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Memory Snapshot Viewer</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d7dce2;
      --line-strong: #aeb7c2;
      --text: #1d2733;
      --muted: #5f6b7a;
      --active: #0f766e;
      --active-soft: #d9f3ef;
      --empty: #f1f3f5;
      --filled: #fff7d6;
      --filled-border: #e0ba43;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.35 "Segoe UI", system-ui, sans-serif;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 3;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    .topbar {
      display: flex;
      align-items: center;
      gap: 16px;
      min-height: 52px;
      padding: 8px 14px;
    }

    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      white-space: nowrap;
    }

    .cycles {
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding: 2px 0;
    }

    button, input {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 6px 10px;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--line-strong);
    }

    button.active {
      border-color: var(--active);
      background: var(--active-soft);
      color: #064e48;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      border-top: 1px solid var(--line);
      padding: 8px 14px;
      overflow-x: auto;
    }

    .snapshotMessage {
      border-top: 1px solid var(--line);
      padding: 8px 14px;
      color: var(--text);
      background: #fbfcfd;
    }

    .snapshotMessage:empty {
      display: none;
    }

    .segmented {
      display: inline-flex;
      border: 1px solid var(--line);
      background: #fff;
    }

    .segmented button {
      border: 0;
      border-right: 1px solid var(--line);
    }

    .segmented button:last-child {
      border-right: 0;
    }

    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      white-space: nowrap;
    }

    input[type="number"] {
      width: 82px;
      border: 1px solid var(--line);
      padding: 5px 7px;
      background: #fff;
      color: var(--text);
    }

    main {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 12px;
      padding: 12px 14px 18px;
    }

    aside {
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--panel);
      max-height: calc(100vh - 134px);
      overflow: auto;
    }

    .sideTitle {
      position: sticky;
      top: 0;
      z-index: 1;
      margin: 0;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      background: #eef2f5;
      font-size: 13px;
      font-weight: 650;
    }

    .startTable {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
    }

    .startTable th,
    .startTable td {
      min-width: 0;
      max-width: none;
      height: auto;
      padding: 5px 6px;
      font-size: 12px;
    }

    .startTable th {
      top: 33px;
    }

    .viewer {
      min-width: 0;
    }

    .status {
      color: var(--muted);
      margin-bottom: 10px;
    }

    .dspPanel {
      display: none;
      border: 1px solid var(--line);
      background: var(--panel);
      margin-bottom: 12px;
    }

    .dspPanel.visible {
      display: block;
    }

    .dspHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      background: #eef2f5;
    }

    .dspStats {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
    }

    .dspStats strong {
      color: var(--text);
      font-weight: 650;
    }

    .dspControls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
    }

    .chartBlock {
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }

    .chartBlock:last-child {
      border-bottom: 0;
    }

    .chartTitle {
      margin-bottom: 6px;
      color: var(--muted);
      font-weight: 650;
    }

    canvas {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
    }

    .tableWrap {
      max-height: calc(100vh - 134px);
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
    }

    table {
      border-collapse: separate;
      border-spacing: 0;
      min-width: 100%;
      table-layout: fixed;
    }

    th, td {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 5px 6px;
      min-width: 260px;
      max-width: 260px;
      height: 66px;
      vertical-align: top;
      overflow: hidden;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #eef2f5;
      text-align: left;
      font-weight: 650;
    }

    th.addr, td.addr {
      position: sticky;
      left: 0;
      z-index: 1;
      min-width: 78px;
      max-width: 78px;
      background: #eef2f5;
      font-weight: 650;
    }

    th.addr {
      z-index: 3;
    }

    td.empty {
      background: var(--empty);
      color: #8b96a3;
    }

    td.filled {
      background: var(--filled);
      border-bottom-color: var(--filled-border);
    }

    .packed {
      display: grid;
      gap: 3px;
    }

    .cellLocation {
      margin-bottom: 4px;
      color: var(--muted);
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 11px;
    }

    .slot {
      white-space: normal;
      overflow: hidden;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 12px;
    }

    .slot.none {
      color: #9aa3ad;
    }

    .error {
      color: #a83a32;
      font-weight: 650;
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>Memory Snapshot Viewer</h1>
      <div id="cycles" class="cycles"></div>
    </div>
    <div id="snapshotMessage" class="snapshotMessage"></div>
    <div class="controls">
      <div class="segmented">
        <button id="sramBtn" class="active" type="button">SRAM</button>
        <button id="lramBtn" type="button">LRAM</button>
      </div>
      <label>Start Addr <input id="startAddr" type="number" min="0" value="0"></label>
      <label>Rows <input id="rowCount" type="number" min="1" value="128"></label>
      <label><input id="nonEmptyOnly" type="checkbox"> Non-empty only</label>
      <button id="reloadBtn" type="button">Reload</button>
    </div>
  </header>

  <main>
    <aside>
      <h2 id="startTitle" class="sideTitle">Layer Starts</h2>
      <table id="startOfLayer" class="startTable"></table>
    </aside>
    <div class="viewer">
      <div id="status" class="status">Loading snapshots...</div>
      <section id="dspPanel" class="dspPanel">
        <div class="dspHeader">
          <div>DSP Activity</div>
          <div id="dspStats" class="dspStats"></div>
        </div>
        <div class="dspControls">
          <label>Cycle Start <input id="dspStart" type="number" min="0" value="0"></label>
          <label>Cycles <input id="dspWindow" type="number" min="1" value="500"></label>
          <label>DSP Start <input id="dspRangeStart" type="number" min="0" value="0"></label>
          <label>DSP Count <input id="dspRangeCount" type="number" min="1" value="292"></label>
          <button id="dspFullRangeBtn" type="button">Full Range</button>
        </div>
        <div class="chartBlock">
          <div class="chartTitle">Active DSPs Per Clock Cycle</div>
          <canvas id="dspCountCanvas" height="180"></canvas>
        </div>
        <div class="chartBlock">
          <div class="chartTitle">DSP Waterfall</div>
          <canvas id="dspWaterfallCanvas" height="340"></canvas>
        </div>
      </section>
      <div class="tableWrap">
        <table id="grid"></table>
      </div>
    </div>
  </main>

  <script>
    const state = {
      snapshots: [],
      cycle: null,
      memoryType: "SRAM",
      memory: null,
      startOfLayer: [],
      message: "",
      dspActivity: null
    };

    const cyclesEl = document.getElementById("cycles");
    const snapshotMessageEl = document.getElementById("snapshotMessage");
    const gridEl = document.getElementById("grid");
    const dspPanelEl = document.getElementById("dspPanel");
    const dspStatsEl = document.getElementById("dspStats");
    const dspStartEl = document.getElementById("dspStart");
    const dspWindowEl = document.getElementById("dspWindow");
    const dspRangeStartEl = document.getElementById("dspRangeStart");
    const dspRangeCountEl = document.getElementById("dspRangeCount");
    const dspFullRangeBtn = document.getElementById("dspFullRangeBtn");
    const dspCountCanvas = document.getElementById("dspCountCanvas");
    const dspWaterfallCanvas = document.getElementById("dspWaterfallCanvas");
    const startTitleEl = document.getElementById("startTitle");
    const startOfLayerEl = document.getElementById("startOfLayer");
    const statusEl = document.getElementById("status");
    const sramBtn = document.getElementById("sramBtn");
    const lramBtn = document.getElementById("lramBtn");
    const startAddrEl = document.getElementById("startAddr");
    const rowCountEl = document.getElementById("rowCount");
    const nonEmptyOnlyEl = document.getElementById("nonEmptyOnly");
    const reloadBtn = document.getElementById("reloadBtn");

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? "status error" : "status";
    }

    async function fetchJSON(url) {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return response.json();
    }

    function renderSnapshotMessage() {
      snapshotMessageEl.textContent = state.message ? `Snapshot note: ${state.message}` : "";
    }

    function renderCycleButtons() {
      cyclesEl.innerHTML = "";
      for (const cycle of state.snapshots) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = cycle;
        button.className = cycle === state.cycle ? "active" : "";
        button.addEventListener("click", () => {
          state.cycle = cycle;
          dspStartEl.value = "";
          dspWindowEl.value = "";
          dspRangeStartEl.value = "";
          dspRangeCountEl.value = "";
          renderCycleButtons();
          loadMemory();
        });
        cyclesEl.appendChild(button);
      }
    }

    function formatItem(item) {
      if (item === null || item === undefined) {
        return "empty";
      }
      if (typeof item !== "object") {
        return String(item);
      }
      const type = item.type || "obj";
      if (type === "depthwise") {
        if (item.part !== undefined) {
          return `Depthwise weight | layer ${item.layer} | channel ${item.channel} | kernel part ${item.part}`;
        }
        if (item.x !== undefined && item.y !== undefined) {
          return `Depthwise activation | layer ${item.layer} | channel ${item.channel} | x ${item.x} | y ${item.y}`;
        }
        return `Depthwise item | layer ${item.layer} | channel ${item.channel}`;
      }
      if (type === "pointwise") {
        if (item.x !== undefined && item.y !== undefined) {
          return `Pointwise activation | layer ${item.layer} | channel ${item.channel} | x ${item.x} | y ${item.y}`;
        }
        if (item.y !== undefined) {
          return `Pointwise weight | layer ${item.layer} | input channel ${item.channel} | output channel ${item.y}`;
        }
        return `Pointwise item | layer ${item.layer} | channel ${item.channel}`;
      }
      if (type === "SEWeight") {
        return `Squeeze-excite weight | ${item.matrix} | layer ${item.layer} | row ${item.row} | column ${item.column}`;
      }
      if (type === "fullyConnected") {
        return `Fully connected weight | layer ${item.layer} | row ${item.row} | column ${item.column}`;
      }
      return JSON.stringify(item);
    }

    function isFilled(data) {
      return Array.isArray(data) && data.some((item) => item !== null && item !== undefined);
    }

    function buildAddressMap(unit) {
      const map = new Map();
      for (const entry of unit.addresses || []) {
        map.set(entry.address, entry.data || []);
      }
      return map;
    }

    function renderGrid() {
      if (!state.memory) {
        gridEl.innerHTML = "";
        return;
      }

      const units = state.memory.units;
      const unitMaps = units.map(buildAddressMap);
      const maxAddress = state.memory.addresses - 1;
      const start = Math.max(0, Number(startAddrEl.value || 0));
      const requestedRows = Math.max(1, Number(rowCountEl.value || 1));
      const nonEmptyOnly = nonEmptyOnlyEl.checked;

      const rows = [];
      for (let address = start; address <= maxAddress && rows.length < requestedRows; address += 1) {
        if (nonEmptyOnly) {
          const hasData = unitMaps.some((unitMap) => isFilled(unitMap.get(address)));
          if (!hasData) {
            continue;
          }
        }
        rows.push(address);
      }

      const header = [`<tr><th class="addr">ADDR</th>`, ...units.map((unit) => `<th>${unit.name}</th>`), `</tr>`].join("");
      const body = rows.map((address) => {
        const cells = units.map((unit, index) => {
          const data = unitMaps[index].get(address) || [];
          const filled = isFilled(data);
          const slots = data.map((item, slotIndex) => {
            const text = formatItem(item);
            const cls = item === null || item === undefined ? "slot none" : "slot";
            return `<div class="${cls}" title="${escapeHTML(JSON.stringify(item))}">packed ${slotIndex}: ${escapeHTML(text)}</div>`;
          }).join("");
          const location = `${unit.name} / ADDR${address}`;
          return `<td class="${filled ? "filled" : "empty"}" title="${escapeHTML(location)}"><div class="cellLocation">${escapeHTML(location)}</div><div class="packed">${slots}</div></td>`;
        }).join("");
        return `<tr><td class="addr">ADDR${address}</td>${cells}</tr>`;
      }).join("");

      gridEl.innerHTML = header + body;
      setStatus(`${state.cycle} / ${state.memoryType}: ${units.length} units, showing ${rows.length} rows`);
    }

    function renderStartOfLayer() {
      startTitleEl.textContent = `${state.memoryType} Layer Starts`;
      if (!state.startOfLayer || state.startOfLayer.length === 0) {
        startOfLayerEl.innerHTML = `<tr><td>No layer start data</td></tr>`;
        return;
      }

      const unitKey = `${state.memoryType}Num`;
      const addrKey = `${state.memoryType}Addr`;
      const packKey = state.memoryType === "SRAM" ? "packedIndex" : "packedPosition";
      const startIndexKey = state.memoryType === "SRAM" ? "SRAMStartIndex" : null;

      const header = startIndexKey ? `
        <tr>
          <th>Layer</th>
          <th>Start</th>
          <th>${state.memoryType}</th>
          <th>Addr</th>
          <th>Pack</th>
        </tr>` : `
        <tr>
          <th>Layer</th>
          <th>${state.memoryType}</th>
          <th>Addr</th>
          <th>Pack</th>
        </tr>`;

      const rows = state.startOfLayer.map((entry) => {
        if (startIndexKey) {
          return `
            <tr>
              <td>${entry.layer}</td>
              <td>${entry[startIndexKey]}</td>
              <td>${entry[unitKey]}</td>
              <td>${entry[addrKey]}</td>
              <td>${entry[packKey]}</td>
            </tr>
          `;
        }
        return `
          <tr>
            <td>${entry.layer}</td>
            <td>${entry[unitKey]}</td>
            <td>${entry[addrKey]}</td>
            <td>${entry[packKey]}</td>
          </tr>
        `;
      }).join("");
      startOfLayerEl.innerHTML = header + rows;
    }

    function resizeCanvas(canvas) {
      const width = Math.max(320, Math.floor(canvas.clientWidth));
      const height = Number(canvas.getAttribute("height"));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function dspClockAtIndex(index) {
      return state.dspActivity.firstClock + index;
    }

    function getDSPVisibleRange() {
      const dsp = state.dspActivity;
      if (!dsp || dsp.numCycles === 0) {
        return null;
      }

      const requestedStartClock = Number(dspStartEl.value || dsp.firstClock);
      const requestedCycles = Math.floor(Math.max(1, Number(dspWindowEl.value || dsp.numCycles)));
      const startIndex = Math.floor(Math.min(Math.max(0, requestedStartClock - dsp.firstClock), dsp.numCycles - 1));
      const endIndex = Math.min(dsp.numCycles, startIndex + requestedCycles);
      return { startIndex, endIndex };
    }

    function calculateVisibleDSPStats(timeRange) {
      const dsp = state.dspActivity;
      const visibleCycles = Math.max(0, timeRange.endIndex - timeRange.startIndex);
      const visibleDSPs = dsp.numDSPsShown;
      const visibleCounts = dsp.activeCounts.slice(timeRange.startIndex, timeRange.endIndex);
      const activeSlots = visibleCounts.reduce((total, count) => total + count, 0);
      const maxActive = visibleCounts.length ? Math.max(...visibleCounts) : 0;

      const totalSlots = visibleCycles * visibleDSPs;
      return {
        visibleCycles,
        visibleDSPs,
        utilizationPercent: totalSlots ? (activeSlots / totalSlots) * 100 : 0,
        maxActive
      };
    }

    function renderDSPActivity() {
      const dsp = state.dspActivity;
      if (!dsp || !dsp.hasData) {
        dspPanelEl.classList.remove("visible");
        return;
      }

      if (!dspStartEl.value) {
        dspStartEl.value = dsp.firstClock;
      }
      if (!dspWindowEl.value) {
        dspWindowEl.value = Math.min(500, dsp.numCycles);
      }
      if (!dspRangeStartEl.value) {
        dspRangeStartEl.value = 0;
      }
      if (!dspRangeCountEl.value) {
        dspRangeCountEl.value = dsp.numDSPs;
      }

      const timeRange = getDSPVisibleRange();
      const visibleStats = calculateVisibleDSPStats(timeRange);

      dspPanelEl.classList.add("visible");
      dspStatsEl.innerHTML = [
        `<span><strong>${visibleStats.utilizationPercent.toFixed(2)}%</strong> active time in view</span>`,
        `<span><strong>${dsp.utilizationPercent.toFixed(2)}%</strong> active time full snapshot</span>`,
        `<span><strong>${visibleStats.visibleCycles}</strong> cycles shown</span>`,
        `<span><strong>${visibleStats.visibleDSPs}</strong> DSPs shown</span>`,
        `<span><strong>${visibleStats.maxActive}</strong> peak active in view</span>`
      ].join("");

      drawDSPCountGraph();
      drawDSPWaterfall();
    }

    async function loadDSPActivity() {
      if (!state.cycle) {
        return;
      }

      const dspStart = Math.max(0, Math.floor(Number(dspRangeStartEl.value || 0)));
      const dspCount = Math.max(1, Math.floor(Number(dspRangeCountEl.value || 292)));
      try {
        state.dspActivity = await fetchJSON(`/api/dsp-activations?cycle=${encodeURIComponent(state.cycle)}&dsp_start=${dspStart}&dsp_count=${dspCount}`);
        renderDSPActivity();
      } catch (error) {
        state.dspActivity = null;
        renderDSPActivity();
        setStatus(`Failed to load DSP activity: ${error.message}`, true);
      }
    }

    function drawDSPCountGraph() {
      const dsp = state.dspActivity;
      const range = getDSPVisibleRange();
      if (!dsp || !range) {
        return;
      }

      resizeCanvas(dspCountCanvas);
      const ctx = dspCountCanvas.getContext("2d");
      const width = dspCountCanvas.width;
      const height = dspCountCanvas.height;
      const padLeft = 44;
      const padRight = 10;
      const padTop = 12;
      const padBottom = 26;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#d7dce2";
      ctx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

      ctx.fillStyle = "#5f6b7a";
      ctx.font = "12px Segoe UI, system-ui, sans-serif";
      const visibleDSPs = dsp.numDSPsShown;
      ctx.fillText(String(visibleDSPs), 8, padTop + 4);
      ctx.fillText("0", 28, padTop + plotHeight);
      ctx.fillText(String(dspClockAtIndex(range.startIndex)), padLeft, height - 7);
      ctx.textAlign = "right";
      ctx.fillText(String(dspClockAtIndex(range.endIndex - 1)), width - padRight, height - 7);
      ctx.textAlign = "left";

      const visibleCount = Math.max(1, range.endIndex - range.startIndex);
      ctx.beginPath();
      for (let index = range.startIndex; index < range.endIndex; index += 1) {
        const x = padLeft + ((index - range.startIndex) / Math.max(1, visibleCount - 1)) * plotWidth;
        const activeCount = dsp.activeCounts[index];
        const y = padTop + plotHeight - (activeCount / Math.max(1, visibleDSPs)) * plotHeight;
        if (index === range.startIndex) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function drawDSPWaterfall() {
      const dsp = state.dspActivity;
      const range = getDSPVisibleRange();
      if (!dsp || !range) {
        return;
      }

      resizeCanvas(dspWaterfallCanvas);
      const ctx = dspWaterfallCanvas.getContext("2d");
      const width = dspWaterfallCanvas.width;
      const height = dspWaterfallCanvas.height;
      const padLeft = 48;
      const padRight = 10;
      const padTop = 12;
      const padBottom = 26;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;
      const visibleCount = Math.max(1, range.endIndex - range.startIndex);

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#d7dce2";
      ctx.strokeRect(padLeft, padTop, plotWidth, plotHeight);

      ctx.fillStyle = "#5f6b7a";
      ctx.font = "12px Segoe UI, system-ui, sans-serif";
      ctx.fillText(`DSP${dsp.dspStart}`, 8, padTop + 10);
      ctx.fillText(`DSP${dsp.dspEnd - 1}`, 8, padTop + plotHeight);
      ctx.fillText(String(dspClockAtIndex(range.startIndex)), padLeft, height - 7);
      ctx.textAlign = "right";
      ctx.fillText(String(dspClockAtIndex(range.endIndex - 1)), width - padRight, height - 7);
      ctx.textAlign = "left";

      ctx.fillStyle = "#d7352a";
      const visibleDSPs = Math.max(1, dsp.numDSPsShown);
      const rowHeight = plotHeight / visibleDSPs;
      for (let dspIndex = 0; dspIndex < dsp.activeRuns.length; dspIndex += 1) {
        const y = padTop + dspIndex * rowHeight;
        const h = Math.max(1, Math.ceil(rowHeight));
        for (const run of dsp.activeRuns[dspIndex]) {
          const runStart = Math.max(run[0], range.startIndex);
          const runEnd = Math.min(run[1], range.endIndex);
          if (runStart >= runEnd) {
            continue;
          }
          const x = padLeft + ((runStart - range.startIndex) / visibleCount) * plotWidth;
          const w = Math.max(1, ((runEnd - runStart) / visibleCount) * plotWidth);
          ctx.fillRect(x, y, w, h);
        }
      }
    }

    function escapeHTML(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function loadMemory() {
      if (!state.cycle) {
        return;
      }
      setStatus(`Loading ${state.memoryType} snapshot ${state.cycle}...`);
      gridEl.innerHTML = "";
      try {
        const messageData = await fetchJSON(`/api/message?cycle=${encodeURIComponent(state.cycle)}`);
        state.message = messageData.message || "";
        renderSnapshotMessage();
        await loadDSPActivity();
        state.startOfLayer = await fetchJSON(`/api/start-of-layer?type=${state.memoryType}`);
        renderStartOfLayer();
        state.memory = await fetchJSON(`/api/memory?cycle=${encodeURIComponent(state.cycle)}&type=${state.memoryType}`);
        renderGrid();
      } catch (error) {
        state.message = "";
        state.dspActivity = null;
        renderSnapshotMessage();
        renderDSPActivity();
        setStatus(`Failed to load memory: ${error.message}`, true);
      }
    }

    async function init() {
      try {
        state.snapshots = await fetchJSON("/api/snapshots");
        if (state.snapshots.length === 0) {
          setStatus("No snapshots found.", true);
          return;
        }
        state.cycle = state.snapshots[state.snapshots.length - 1];
        renderCycleButtons();
        await loadMemory();
      } catch (error) {
        setStatus(`Failed to load snapshots: ${error.message}`, true);
      }
    }

    sramBtn.addEventListener("click", () => {
      state.memoryType = "SRAM";
      sramBtn.classList.add("active");
      lramBtn.classList.remove("active");
      loadMemory();
    });

    lramBtn.addEventListener("click", () => {
      state.memoryType = "LRAM";
      lramBtn.classList.add("active");
      sramBtn.classList.remove("active");
      loadMemory();
    });

    startAddrEl.addEventListener("change", renderGrid);
    rowCountEl.addEventListener("change", renderGrid);
    nonEmptyOnlyEl.addEventListener("change", renderGrid);
    reloadBtn.addEventListener("click", loadMemory);
    dspStartEl.addEventListener("change", renderDSPActivity);
    dspWindowEl.addEventListener("change", renderDSPActivity);
    dspRangeStartEl.addEventListener("change", loadDSPActivity);
    dspRangeCountEl.addEventListener("change", loadDSPActivity);
    dspFullRangeBtn.addEventListener("click", () => {
      if (!state.dspActivity || !state.dspActivity.hasData) {
        return;
      }
      dspStartEl.value = state.dspActivity.firstClock;
      dspWindowEl.value = state.dspActivity.numCycles;
      dspRangeStartEl.value = 0;
      dspRangeCountEl.value = state.dspActivity.numDSPs;
      renderDSPActivity();
    });
    window.addEventListener("resize", renderDSPActivity);

    init();
  </script>
</body>
</html>
"""


def numeric_sort_key(path):
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def read_memory_snapshot(cycle, memory_type):
    if memory_type not in {"SRAM", "LRAM"}:
        raise ValueError("memory type must be SRAM or LRAM")

    memory_dir = SNAPSHOTS_DIR / cycle / memory_type
    if not memory_dir.is_dir():
        raise FileNotFoundError(f"Missing snapshot directory: {memory_dir}")

    files = sorted(memory_dir.glob(f"{memory_type}*.json"), key=lambda path: int(path.stem.replace(memory_type, "")))
    units = []
    address_count = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        addresses = data.get("addresses", [])
        address_count = max(address_count, len(addresses))
        units.append({
            "name": file_path.stem,
            "addresses": addresses
        })

    return {
        "cycle": cycle,
        "type": memory_type,
        "addresses": address_count,
        "units": units
    }


def read_start_of_layer(memory_type):
    if memory_type == "SRAM":
        file_path = SNAPSHOTS_DIR / "SRAMStartOfLayer.json"
    else:
        file_path = SNAPSHOTS_DIR / "LRAMStartOfLayer.json"

    if not file_path.is_file():
        return []
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def read_snapshot_message(cycle):
    file_path = SNAPSHOTS_DIR / cycle / "message.json"
    if not file_path.is_file():
        return {"message": ""}
    with file_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if isinstance(data, dict):
        return {"message": str(data.get("message", ""))}
    return {"message": str(data)}


def read_dsp_activations(cycle, dsp_start=0, dsp_count=None):
    file_path = SNAPSHOTS_DIR / cycle / "DSPActivations.json"
    if not file_path.is_file():
        return {
            "hasData": False,
            "message": "No DSP activation tracking found for this snapshot."
        }

    try:
        with file_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError:
        return {
            "hasData": False,
            "message": "DSP activation tracking file is empty or incomplete."
        }

    num_dsps = int(data.get("numDSPs", 0))
    if num_dsps <= 0:
        return {
            "hasData": False,
            "message": "DSP activation tracking file is empty."
        }

    dsp_start = max(0, min(int(dsp_start), num_dsps - 1))
    if dsp_count is None:
        dsp_count = num_dsps
    dsp_count = max(1, int(dsp_count))
    dsp_end = min(num_dsps, dsp_start + dsp_count)
    selected_dsps = dsp_end - dsp_start

    if data.get("format") == "active-runs-v1":
        num_cycles = int(data.get("numCycles", 0))
        if num_cycles <= 0:
            return {
                "hasData": False,
                "message": "DSP activation tracking file is empty."
            }

        first_clock = int(data.get("firstClock") or 0)
        last_clock = int(data.get("lastClock") or (first_clock + num_cycles - 1))
        all_active_runs = data.get("activeRuns", [])
        selected_active_runs = []
        count_deltas = [0] * (num_cycles + 1)
        total_active_slots = 0

        for dsp_index in range(dsp_start, dsp_end):
            runs = all_active_runs[dsp_index] if dsp_index < len(all_active_runs) else []
            clipped_runs = []
            for run in runs:
                if len(run) < 2:
                    continue
                run_start = max(0, min(num_cycles, int(run[0])))
                run_end = max(0, min(num_cycles, int(run[1])))
                if run_start >= run_end:
                    continue
                clipped_runs.append([run_start, run_end])
                count_deltas[run_start] += 1
                count_deltas[run_end] -= 1
                total_active_slots += run_end - run_start
            selected_active_runs.append(clipped_runs)

        active_counts = []
        active_count = 0
        for cycle_index in range(num_cycles):
            active_count += count_deltas[cycle_index]
            active_counts.append(active_count)

        total_slots = num_cycles * selected_dsps
        utilization_percent = (total_active_slots / total_slots) * 100 if total_slots else 0

        return {
            "hasData": True,
            "numDSPs": num_dsps,
            "dspStart": dsp_start,
            "dspEnd": dsp_end,
            "numDSPsShown": selected_dsps,
            "numCycles": num_cycles,
            "firstClock": first_clock,
            "lastClock": last_clock,
            "totalActiveSlots": total_active_slots,
            "utilizationPercent": utilization_percent,
            "maxActiveDSPs": max(active_counts) if active_counts else 0,
            "activeCounts": active_counts,
            "activeRuns": selected_active_runs
        }

    raw_cycles = data.get("cycles", [])
    if not raw_cycles:
        return {
            "hasData": False,
            "message": "DSP activation tracking file is empty."
        }

    active_counts = []
    active_runs = [[] for _ in range(selected_dsps)]
    open_runs = [None] * selected_dsps
    total_active_slots = 0
    first_clock = int(raw_cycles[0].get("clockCycle", 0))
    last_clock = first_clock

    for cycle_index, entry in enumerate(raw_cycles):
        last_clock = int(entry.get("clockCycle", first_clock + cycle_index))
        active = entry.get("active", [])
        active_count = 0

        for local_dsp_index, dsp_index in enumerate(range(dsp_start, dsp_end)):
            is_active = bool(active[dsp_index]) if dsp_index < len(active) else False
            if is_active:
                active_count += 1
                if open_runs[local_dsp_index] is None:
                    open_runs[local_dsp_index] = cycle_index
            elif open_runs[local_dsp_index] is not None:
                active_runs[local_dsp_index].append([open_runs[local_dsp_index], cycle_index])
                open_runs[local_dsp_index] = None

        active_counts.append(active_count)
        total_active_slots += active_count

    num_cycles = len(raw_cycles)
    for local_dsp_index, run_start in enumerate(open_runs):
        if run_start is not None:
            active_runs[local_dsp_index].append([run_start, num_cycles])

    total_slots = num_cycles * selected_dsps
    utilization_percent = (total_active_slots / total_slots) * 100 if total_slots else 0

    return {
        "hasData": True,
        "numDSPs": num_dsps,
        "dspStart": dsp_start,
        "dspEnd": dsp_end,
        "numDSPsShown": selected_dsps,
        "numCycles": num_cycles,
        "firstClock": first_clock,
        "lastClock": last_clock,
        "totalActiveSlots": total_active_slots,
        "utilizationPercent": utilization_percent,
        "maxActiveDSPs": max(active_counts) if active_counts else 0,
        "activeCounts": active_counts,
        "activeRuns": active_runs
    }


class SnapshotHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_text(self, text, content_type="text/html; charset=utf-8", status=200):
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(HTML)
            return

        if parsed.path == "/api/snapshots":
            snapshots = []
            if SNAPSHOTS_DIR.is_dir():
                snapshots = [path.name for path in sorted(SNAPSHOTS_DIR.iterdir(), key=numeric_sort_key) if path.is_dir()]
            self.send_json(snapshots)
            return

        if parsed.path == "/api/start-of-layer":
            query = parse_qs(parsed.query)
            memory_type = query.get("type", ["LRAM"])[0]
            self.send_json(read_start_of_layer(memory_type))
            return

        if parsed.path == "/api/message":
            query = parse_qs(parsed.query)
            cycle = query.get("cycle", [""])[0]
            self.send_json(read_snapshot_message(cycle))
            return

        if parsed.path == "/api/dsp-activations":
            query = parse_qs(parsed.query)
            cycle = query.get("cycle", [""])[0]
            dsp_start = int(query.get("dsp_start", ["0"])[0])
            dsp_count_arg = query.get("dsp_count", [None])[0]
            dsp_count = int(dsp_count_arg) if dsp_count_arg is not None else None
            try:
                self.send_json(read_dsp_activations(cycle, dsp_start, dsp_count))
            except Exception as error:
                self.send_json({"hasData": False, "error": str(error)}, status=400)
            return

        if parsed.path == "/api/memory":
            query = parse_qs(parsed.query)
            cycle = query.get("cycle", [""])[0]
            memory_type = query.get("type", ["SRAM"])[0]
            try:
                self.send_json(read_memory_snapshot(cycle, memory_type))
            except Exception as error:
                self.send_json({"error": str(error)}, status=400)
            return

        content_type = mimetypes.guess_type(parsed.path)[0] or "text/plain"
        self.send_text("Not found", content_type=content_type, status=404)


def main():
    server = ThreadingHTTPServer((HOST, PORT), SnapshotHandler)
    print(f"Snapshot viewer running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
