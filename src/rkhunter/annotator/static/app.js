"use strict";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#annotationCanvas");
const ctx = canvas.getContext("2d");
const classColors = ["#d5ff5f", "#61c7ff", "#ffcb62", "#d98cff", "#8a9692", "#ff746d"];

const state = {
  projects: [], models: [], project: null, items: [], filteredItems: [], current: null,
  image: null, boxes: [], selected: -1, tool: "select", dirty: false,
  zoom: 1, panX: 0, panY: 0, drag: null, history: [], future: [],
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.add("hidden"), 4200);
}

function setLoading(active, message = "处理中…") {
  $("#loadingOverlay").textContent = message;
  $("#loadingOverlay").classList.toggle("hidden", !active);
}

function setDirty(value, message) {
  state.dirty = value;
  const node = $("#saveState");
  node.textContent = message || (value ? "有未保存修改" : "已同步");
  node.className = `save-state${value ? " dirty" : ""}`;
}

function cloneBoxes() { return JSON.parse(JSON.stringify(state.boxes)); }
function checkpoint() {
  state.history.push(cloneBoxes());
  if (state.history.length > 60) state.history.shift();
  state.future = [];
}

function undo() {
  if (!state.history.length) return;
  state.future.push(cloneBoxes());
  state.boxes = state.history.pop();
  state.selected = Math.min(state.selected, state.boxes.length - 1);
  setDirty(true); renderAll();
}

function redo() {
  if (!state.future.length) return;
  state.history.push(cloneBoxes());
  state.boxes = state.future.pop();
  state.selected = Math.min(state.selected, state.boxes.length - 1);
  setDirty(true); renderAll();
}

async function boot() {
  try {
    const health = await api("/api/v1/health");
    $("#versionText").textContent = `本地 · v${health.tool_version} · DB ${health.database_schema_version}`;
    state.projects = await api("/api/v1/projects");
    state.models = await api("/api/v1/models");
    renderProjectSelect(); renderModelSelect();
    if (state.projects.length) await chooseProject(state.projects[0].id);
    else toast("尚未注册项目，请使用启动参数 --dataset 创建项目", true);
  } catch (error) { toast(`启动失败：${error.message}`, true); }
}

function renderProjectSelect() {
  $("#projectSelect").innerHTML = state.projects.map(project =>
    `<option value="${escapeHtml(project.id)}">${escapeHtml(project.name)}</option>`).join("");
}

function renderModelSelect() {
  $("#modelSelect").innerHTML = state.models.length
    ? state.models.map(model => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)} · ${escapeHtml(model.version)}</option>`).join("")
    : `<option value="">未注册本地模型</option>`;
}

async function chooseProject(id) {
  if (state.dirty && !confirm("当前图片有未保存修改，确定切换项目？")) return;
  state.project = state.projects.find(project => project.id === id);
  $("#projectSelect").value = id;
  renderClassSelect();
  await loadQueue(); await loadStats();
}

function renderClassSelect() {
  const classes = state.project?.classes || {};
  $("#classSelect").innerHTML = Object.entries(classes)
    .filter(([, name]) => name.toLowerCase() !== "background")
    .map(([id, name]) => `<option value="${id}">${escapeHtml(name)}</option>`).join("");
}

function renderImageMeta(detail) {
  $("#imageName").textContent = detail.rel_path;
  $("#imageInfo").textContent = `${detail.width}×${detail.height} · ${detail.split} · revision ${detail.revision}`;
  $("#warningText").textContent = detail.warning || "";
}

async function loadQueue() {
  if (!state.project) return;
  const filter = $("#statusFilter").value;
  const items = [];
  let offset = 0;
  let total = 0;
  do {
    const query = new URLSearchParams({ limit: "500", offset: String(offset) });
    if (filter) query.set("status", filter);
    const payload = await api(`/api/v1/projects/${state.project.id}/images?${query}`);
    total = payload.total;
    if (!payload.items.length) break;
    items.push(...payload.items);
    offset += payload.items.length;
  } while (offset < total);
  state.items = items;
  applySearch();
  if (state.current) {
    const refreshed = state.items.find(item => item.id === state.current.id);
    if (refreshed) Object.assign(state.current, refreshed);
  }
}

function applySearch() {
  const needle = $("#searchInput").value.trim().toLowerCase();
  state.filteredItems = state.items.filter(item => item.rel_path.toLowerCase().includes(needle));
  $("#queueCount").textContent = `${state.filteredItems.length}/${state.items.length}`;
  renderImageList();
}

function renderImageList() {
  const list = $("#imageList");
  list.innerHTML = state.filteredItems.map(item => `
    <div class="image-item ${state.current?.id === item.id ? "active" : ""}" data-image-id="${item.id}">
      <span class="status-dot ${item.status}"></span>
      <span class="image-item-name" title="${escapeHtml(item.rel_path)}">${escapeHtml(item.rel_path.split("/").pop())}</span>
      <small>${item.annotation_count || 0}</small>
    </div>`).join("");
  list.querySelectorAll(".image-item").forEach(node => node.addEventListener("click", () => selectImage(node.dataset.imageId)));
}

async function selectImage(id, force = false) {
  if (!force && state.dirty && !confirm("当前图片有未保存修改，确定离开？")) return;
  setLoading(true, "加载图片…");
  try {
    const detail = await api(`/api/v1/projects/${state.project.id}/images/${id}`);
    const image = new Image();
    image.decoding = "async";
    image.src = `/api/v1/projects/${state.project.id}/images/${id}/content?revision=${detail.revision}`;
    await image.decode();
    state.current = detail;
    state.image = image;
    state.boxes = detail.annotations.map(row => ({
      class_id: row.class_id, x1: row.x1, y1: row.y1, x2: row.x2, y2: row.y2,
      confidence: row.confidence, source: row.source, model_id: row.model_id,
      model_revision_id: row.model_revision_id, warning: row.warning,
    }));
    state.selected = -1; state.zoom = 1; state.panX = 0; state.panY = 0;
    state.history = []; state.future = []; setDirty(false);
    $("#emptyState").classList.add("hidden");
    renderImageMeta(detail);
    renderAll(); renderImageList();
  } catch (error) { toast(`加载失败：${error.message}`, true); }
  finally { setLoading(false); }
}

function canvasMetrics() {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width, height = rect.height;
  if (!state.current) return { width, height, scale: 1, ox: 0, oy: 0 };
  const fit = Math.min(width / state.current.width, height / state.current.height);
  const scale = fit * state.zoom;
  return {
    width, height, scale,
    ox: (width - state.current.width * scale) / 2 + state.panX,
    oy: (height - state.current.height * scale) / 2 + state.panY,
  };
}

function imageToScreen(x, y) {
  const m = canvasMetrics(); return { x: m.ox + x * m.scale, y: m.oy + y * m.scale };
}
function screenToImage(x, y) {
  const m = canvasMetrics();
  return {
    x: Math.max(0, Math.min(state.current.width, (x - m.ox) / m.scale)),
    y: Math.max(0, Math.min(state.current.height, (y - m.oy) / m.scale)),
  };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); renderCanvas();
}

function renderCanvas() {
  const m = canvasMetrics(); ctx.clearRect(0, 0, m.width, m.height);
  if (!state.image || !state.current) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(state.image, m.ox, m.oy, state.current.width * m.scale, state.current.height * m.scale);
  state.boxes.forEach((box, index) => drawBox(box, index, m));
  if (state.drag?.type === "draw") drawBox(state.drag.preview, -2, m);
}

function drawBox(box, index, m) {
  const color = classColors[box.class_id % classColors.length];
  const p1 = imageToScreen(box.x1, box.y1), p2 = imageToScreen(box.x2, box.y2);
  ctx.save(); ctx.lineWidth = index === state.selected ? 3 : 2; ctx.strokeStyle = color;
  ctx.setLineDash(box.source === "auto" ? [7, 4] : []);
  ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  const label = `${className(box.class_id)}${box.confidence != null ? ` ${(box.confidence * 100).toFixed(0)}%` : ""}`;
  ctx.font = "12px Segoe UI"; const labelWidth = ctx.measureText(label).width + 10;
  ctx.fillStyle = color; ctx.fillRect(p1.x, Math.max(0, p1.y - 20), labelWidth, 20);
  ctx.fillStyle = "#101614"; ctx.fillText(label, p1.x + 5, Math.max(14, p1.y - 6));
  if (index === state.selected) {
    ctx.fillStyle = color; ctx.fillRect(p2.x - 6, p2.y - 6, 12, 12);
  }
  ctx.restore();
}

function renderAll() { renderCanvas(); renderBoxList(); updateControls(); }

function renderBoxList() {
  const list = $("#boxList");
  if (!state.boxes.length) { list.innerHTML = `<p class="muted">尚无框；纯背景可直接确认。</p>`; return; }
  list.innerHTML = state.boxes.map((box, index) => `
    <div class="box-row ${index === state.selected ? "active" : ""}" data-box-index="${index}">
      <span class="class-swatch" style="background:${classColors[box.class_id % classColors.length]}"></span>
      <span><strong>${escapeHtml(className(box.class_id))}</strong>
      <small>${box.source}${box.confidence != null ? ` · ${(box.confidence * 100).toFixed(1)}%` : ""}</small>
      ${box.warning ? `<small class="warn">${escapeHtml(box.warning)}</small>` : ""}</span>
      <small>${Math.round(box.x2 - box.x1)}×${Math.round(box.y2 - box.y1)}</small>
    </div>`).join("");
  list.querySelectorAll(".box-row").forEach(node => node.addEventListener("click", () => {
    state.selected = Number(node.dataset.boxIndex); syncClassSelect(); renderAll();
  }));
}

function updateControls() {
  $("#deleteButton").disabled = state.selected < 0;
  $("#undoButton").disabled = !state.history.length;
  $("#redoButton").disabled = !state.future.length;
  $("#saveButton").disabled = !state.current;
  $("#autoButton").disabled = !state.current || !$("#modelSelect").value
    || ["reviewed", "rejected"].includes(state.current.status);
  $("#zoomText").textContent = `${Math.round(state.zoom * 100)}%`;
}

function syncClassSelect() {
  if (state.selected >= 0) $("#classSelect").value = String(state.boxes[state.selected].class_id);
}

function className(id) { return state.project?.classes?.[id] ?? state.project?.classes?.[String(id)] ?? `class_${id}`; }

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function hitTest(screen) {
  for (let index = state.boxes.length - 1; index >= 0; index--) {
    const box = state.boxes[index], p1 = imageToScreen(box.x1, box.y1), p2 = imageToScreen(box.x2, box.y2);
    if (Math.abs(screen.x - p2.x) <= 10 && Math.abs(screen.y - p2.y) <= 10) return { index, handle: "resize" };
    if (screen.x >= p1.x && screen.x <= p2.x && screen.y >= p1.y && screen.y <= p2.y) return { index, handle: "move" };
  }
  return null;
}

function onPointerDown(event) {
  if (!state.current) return;
  canvas.setPointerCapture(event.pointerId); const screen = pointerPosition(event); const imagePoint = screenToImage(screen.x, screen.y);
  if (state.tool === "draw") {
    checkpoint(); state.drag = { type: "draw", start: imagePoint, preview: { class_id: Number($("#classSelect").value), x1: imagePoint.x, y1: imagePoint.y, x2: imagePoint.x + 1, y2: imagePoint.y + 1, source: "manual" } };
  } else if (state.tool === "pan") {
    state.drag = { type: "pan", start: screen, panX: state.panX, panY: state.panY };
  } else {
    const hit = hitTest(screen);
    if (!hit) { state.selected = -1; renderAll(); return; }
    checkpoint(); state.selected = hit.index; syncClassSelect();
    state.drag = { type: hit.handle, start: imagePoint, original: { ...state.boxes[hit.index] } };
  }
}

function onPointerMove(event) {
  if (!state.drag || !state.current) return;
  const screen = pointerPosition(event), point = screenToImage(screen.x, screen.y);
  if (state.drag.type === "draw") {
    const start = state.drag.start;
    state.drag.preview = { ...state.drag.preview, x1: Math.min(start.x, point.x), y1: Math.min(start.y, point.y), x2: Math.max(start.x, point.x), y2: Math.max(start.y, point.y) };
  } else if (state.drag.type === "pan") {
    state.panX = state.drag.panX + screen.x - state.drag.start.x; state.panY = state.drag.panY + screen.y - state.drag.start.y;
  } else if (state.selected >= 0) {
    const box = state.boxes[state.selected], original = state.drag.original;
    if (state.drag.type === "resize") {
      box.x2 = Math.max(original.x1 + 1, point.x); box.y2 = Math.max(original.y1 + 1, point.y);
    } else {
      const dx = point.x - state.drag.start.x, dy = point.y - state.drag.start.y;
      const width = original.x2 - original.x1, height = original.y2 - original.y1;
      box.x1 = Math.max(0, Math.min(state.current.width - width, original.x1 + dx));
      box.y1 = Math.max(0, Math.min(state.current.height - height, original.y1 + dy));
      box.x2 = box.x1 + width; box.y2 = box.y1 + height;
    }
    box.source = "manual"; box.warning = null;
  }
  renderCanvas();
}

function onPointerUp() {
  if (!state.drag) return;
  if (state.drag.type === "draw") {
    const box = state.drag.preview;
    if (box.x2 - box.x1 >= 2 && box.y2 - box.y1 >= 2) { state.boxes.push(box); state.selected = state.boxes.length - 1; setDirty(true); }
    else state.history.pop();
  } else if (state.drag.type !== "pan") setDirty(true);
  state.drag = null; renderAll();
}

function deleteSelected() {
  if (state.selected < 0) return;
  checkpoint(); state.boxes.splice(state.selected, 1); state.selected = -1; setDirty(true); renderAll();
}

function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll(".tool").forEach(node => node.classList.toggle("active", node.dataset.tool === tool));
  canvas.style.cursor = tool === "draw" ? "crosshair" : tool === "pan" ? "grab" : "default";
}

async function saveCurrent(status = "reviewed", boxes = state.boxes) {
  if (!state.current) return;
  setLoading(true, "保存中…");
  try {
    const saved = await api(`/api/v1/projects/${state.project.id}/images/${state.current.id}/annotations`, {
      method: "PUT", body: JSON.stringify({ expected_revision: state.current.revision, status, actor: "local-web", annotations: boxes }),
    });
    state.current = saved; state.boxes = saved.annotations.map(row => ({ ...row }));
    renderImageMeta(saved);
    state.history = []; state.future = []; state.selected = -1; setDirty(false, "保存成功");
    await loadQueue(); await loadStats(); renderAll(); toast(status === "reviewed" ? "已保存并批准" : "状态已更新");
  } catch (error) {
    $("#saveState").textContent = error.status === 409 ? "版本冲突，请重新加载" : "保存失败";
    $("#saveState").className = "save-state error";
    toast(`保存失败：${error.message}`, true);
  } finally { setLoading(false); }
}

async function autoCurrent() {
  if (!state.current) return;
  if (state.dirty) { toast("请先保存或撤销当前修改", true); return; }
  setLoading(true, "模型正在生成草稿…");
  try {
    const detail = await api(`/api/v1/projects/${state.project.id}/images/${state.current.id}/auto-label`, {
      method: "POST", body: JSON.stringify({ model_id: $("#modelSelect").value, params: { conf: 0.05, imgsz: 640, max_det: 30, device: "cpu" }, replace_auto: true }),
    });
    state.current = detail; state.boxes = detail.annotations.map(row => ({ ...row })); state.selected = -1; setDirty(false);
    renderImageMeta(detail);
    await loadQueue(); await loadStats(); renderAll(); toast(`生成 ${state.boxes.length} 个待审核候选`);
  } catch (error) { toast(`自动标注失败：${error.message}`, true); }
  finally { setLoading(false); }
}

async function autoQueue() {
  const imageIds = state.filteredItems.filter(item => item.status !== "reviewed" && item.status !== "rejected").map(item => item.id);
  if (!imageIds.length) { toast("当前队列没有可自动标注的图片"); return; }
  try {
    const run = await api("/api/v1/autolabel-runs", { method: "POST", body: JSON.stringify({ project_id: state.project.id, model_id: $("#modelSelect").value, image_ids: imageIds, params: { conf: 0.05, imgsz: 640, max_det: 30, device: "cpu" } }) });
    pollRun(run.id);
  } catch (error) { toast(`任务创建失败：${error.message}`, true); }
}

async function pollRun(id) {
  setLoading(true, "批量自动标注 0%…");
  try {
    while (true) {
      const run = await api(`/api/v1/autolabel-runs/${id}`);
      const percent = run.total ? Math.round(run.completed / run.total * 100) : 0;
      $("#loadingOverlay").textContent = `批量自动标注 ${percent}% (${run.completed}/${run.total})`;
      if (["completed", "completed_with_errors", "failed", "cancelled"].includes(run.status)) {
        if (run.status === "failed") throw new Error(run.error || "任务失败");
        toast(`任务 ${run.status}：${run.completed}/${run.total}`); break;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    await loadQueue(); await loadStats();
    if (state.current) await selectImage(state.current.id, true);
  } catch (error) { toast(`批量任务失败：${error.message}`, true); }
  finally { setLoading(false); }
}

async function exportYolo() {
  setLoading(true, "正在创建不可变导出版本…");
  try {
    const result = await api(`/api/v1/projects/${state.project.id}/exports/yolo`, { method: "POST", body: JSON.stringify({ reviewed_only: true, output_root: "experiments/annotation-tool/exports", copy_mode: "copy" }) });
    const readinessLabels = {
      missing_train_images: "训练集尚无已审核图片",
      missing_train_boxes: "训练集尚无已批准目标框",
      missing_val_images: "验证集尚无已审核图片",
      missing_val_boxes: "验证集尚无已批准目标框",
    };
    const issues = result.readiness_issues.map(value => readinessLabels[value] || value);
    const readiness = result.train_ready ? "可用于训练" : `仅快照：${issues.join("；")}`;
    const audit = result.audit_recorded ? "" : "；审计写入失败，请保留此 revision 编号";
    toast(`已导出 ${result.images} 图 / ${result.boxes} 框：${result.revision}（${readiness}${audit}）`, !result.audit_recorded);
  } catch (error) { toast(`导出失败：${error.message}`, true); }
  finally { setLoading(false); }
}

async function loadStats() {
  if (!state.project) return;
  const stats = await api(`/api/v1/projects/${state.project.id}/stats`);
  const imageEntries = Object.entries(stats.images);
  $("#statsPanel").innerHTML = imageEntries.map(([name, count]) => `<div class="stat"><strong>${count}</strong><span>${escapeHtml(name)}</span></div>`).join("") || `<p class="muted">无数据</p>`;
}

function navigate(delta) {
  if (!state.filteredItems.length) return;
  const currentIndex = state.filteredItems.findIndex(item => item.id === state.current?.id);
  const nextIndex = Math.max(0, Math.min(state.filteredItems.length - 1, currentIndex + delta));
  selectImage(state.filteredItems[nextIndex].id);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
canvas.addEventListener("wheel", event => { if (!state.current) return; event.preventDefault(); state.zoom = Math.max(.25, Math.min(8, state.zoom * (event.deltaY < 0 ? 1.12 : .89))); renderAll(); }, { passive: false });
window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe($("#canvasWrap"));

document.querySelectorAll(".tool").forEach(button => button.addEventListener("click", () => setTool(button.dataset.tool)));
$("#projectSelect").addEventListener("change", event => chooseProject(event.target.value));
$("#statusFilter").addEventListener("change", loadQueue);
$("#searchInput").addEventListener("input", applySearch);
$("#refreshButton").addEventListener("click", async () => { await loadQueue(); await loadStats(); });
$("#classSelect").addEventListener("change", event => { if (state.selected < 0) return; checkpoint(); const box = state.boxes[state.selected]; box.class_id = Number(event.target.value); box.source = "manual"; setDirty(true); renderAll(); });
$("#deleteButton").addEventListener("click", deleteSelected);
$("#undoButton").addEventListener("click", undo); $("#redoButton").addEventListener("click", redo);
$("#zoomOutButton").addEventListener("click", () => { state.zoom = Math.max(.25, state.zoom / 1.2); renderAll(); });
$("#zoomInButton").addEventListener("click", () => { state.zoom = Math.min(8, state.zoom * 1.2); renderAll(); });
$("#fitButton").addEventListener("click", () => { state.zoom = 1; state.panX = state.panY = 0; renderAll(); });
$("#saveButton").addEventListener("click", () => saveCurrent("reviewed"));
$("#backgroundButton").addEventListener("click", () => { checkpoint(); state.boxes = []; state.selected = -1; saveCurrent("reviewed", []); });
$("#rejectButton").addEventListener("click", () => saveCurrent("rejected", []));
$("#prevButton").addEventListener("click", () => navigate(-1)); $("#nextButton").addEventListener("click", () => navigate(1));
$("#autoButton").addEventListener("click", autoCurrent); $("#batchButton").addEventListener("click", autoQueue);
$("#exportButton").addEventListener("click", exportYolo);

document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrent("reviewed"); }
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
  else if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteSelected(); }
  else if (event.key.toLowerCase() === "v") setTool("select");
  else if (event.key.toLowerCase() === "b") setTool("draw");
  else if (event.key.toLowerCase() === "h") setTool("pan");
  else if (event.key === "ArrowLeft") navigate(-1);
  else if (event.key === "ArrowRight") navigate(1);
});

boot();
