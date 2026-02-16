const state = {
  activeRunId: null,
  nextIndex: 0,
  pollHandle: null,
  runRefreshHandle: null,
  finalRendered: false,
  progressPct: 0,
  pattern: "idle",
};

const timelineEl = document.getElementById("timeline");
const runListEl = document.getElementById("run-list");
const fileTreeEl = document.getElementById("file-tree");
const memoryListEl = document.getElementById("memory-list");
const workspaceInputEl = document.getElementById("workspace-input");
const uploadDestinationInputEl = document.getElementById("upload-destination-input");
const fileUploadInputEl = document.getElementById("file-upload-input");
const uploadFilesButtonEl = document.getElementById("upload-files-button");
const statusPillEl = document.getElementById("status-pill");
const progressFillEl = document.getElementById("progress-fill");
const progressTextEl = document.getElementById("progress-text");
const patternTextEl = document.getElementById("pattern-text");
const taskFormEl = document.getElementById("task-form");
const taskInputEl = document.getElementById("task-input");
const runButtonEl = document.getElementById("run-button");
const imagePathInputEl = document.getElementById("image-path-input");
const imageUploadInputEl = document.getElementById("image-upload-input");
const uploadImageButtonEl = document.getElementById("upload-image-button");
const clearImageButtonEl = document.getElementById("clear-image-button");

const maxStepsInputEl = document.getElementById("max-steps-input");
const minNewTokensInputEl = document.getElementById("min-new-tokens-input");
const maxNewTokensInputEl = document.getElementById("max-new-tokens-input");
const temperatureInputEl = document.getElementById("temperature-input");
const maxGpuMemoryInputEl = document.getElementById("max-gpu-memory-input");
const coderModelInputEl = document.getElementById("coder-model-input");
const lazyLoadInputEl = document.getElementById("lazy-load-input");
const sparseLoadInputEl = document.getElementById("sparse-load-input");
const noShellInputEl = document.getElementById("no-shell-input");
const noOcrInputEl = document.getElementById("no-ocr-input");

document.getElementById("refresh-files").addEventListener("click", loadFiles);
document.getElementById("refresh-runs").addEventListener("click", loadRuns);
document.getElementById("refresh-memory").addEventListener("click", loadMemory);
document.getElementById("clear-memory").addEventListener("click", clearMemory);
uploadFilesButtonEl.addEventListener("click", uploadWorkspaceFiles);
uploadImageButtonEl.addEventListener("click", uploadImageFile);
clearImageButtonEl.addEventListener("click", clearImageSelection);
taskFormEl.addEventListener("submit", onSubmitTask);

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

function setStatus(text, mode = "idle") {
  statusPillEl.textContent = text;
  statusPillEl.className = `status-pill ${mode}`;
}

function setProgress(progressPct, pattern = null) {
  state.progressPct = Math.max(0, Math.min(100, Number(progressPct) || 0));
  progressFillEl.style.width = `${state.progressPct}%`;
  progressTextEl.textContent = `${Math.round(state.progressPct)}%`;
  if (pattern) {
    state.pattern = String(pattern);
    patternTextEl.textContent = state.pattern;
  }
}

function addEventCard({ title, text = "", code = "", type = "info", timestamp = null }) {
  const card = document.createElement("article");
  card.className = `event-card ${type}`;

  const eventHeader = document.createElement("div");
  eventHeader.className = "event-header";
  const titleEl = document.createElement("div");
  titleEl.className = "event-title";
  titleEl.textContent = title;
  const timeEl = document.createElement("div");
  timeEl.className = "event-time";
  if (timestamp) {
    timeEl.textContent = new Date(timestamp).toLocaleTimeString();
  }

  eventHeader.appendChild(titleEl);
  eventHeader.appendChild(timeEl);

  const bodyEl = document.createElement("div");
  bodyEl.className = "event-body";
  if (text) {
    bodyEl.textContent = text;
  }
  if (code) {
    const codeEl = document.createElement("code");
    codeEl.innerHTML = escapeHtml(code);
    bodyEl.appendChild(codeEl);
  }

  card.appendChild(eventHeader);
  card.appendChild(bodyEl);
  timelineEl.appendChild(card);
  timelineEl.scrollTop = timelineEl.scrollHeight;
}

function renderEvent(event) {
  if (event.type === "run_started") {
    setProgress(0, event.has_session_memory ? "planning_with_memory" : "planning");
    addEventCard({
      title: "Run Started",
      text: `${event.task}\nWorkspace: ${event.workspace}\nStep limit: ${event.step_limit}\nModel: ${event.coder_model || "unknown"}\nLazy: ${event.lazy_load ? "on" : "off"}\nSparse: ${event.sparse_load ? "on" : "off"}${event.max_gpu_memory_gib ? `\nGPU cap: ${event.max_gpu_memory_gib} GiB` : ""}`,
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "memory_context_used") {
    addEventCard({
      title: "Session Memory Loaded",
      text: `Using ${event.memory_entries || 0} prior run summaries as context.`,
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "step_decision") {
    const thought = event.thought ? `Thought: ${event.thought}` : "Thought: (none)";
    const actions = event.actions && event.actions.length > 0
      ? JSON.stringify(event.actions, null, 2)
      : "[]";
    addEventCard({
      title: `Step ${event.step} Decision`,
      text: thought,
      code: `Actions:\n${actions}`,
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "progress_update") {
    setProgress(event.progress_pct, event.pattern || "reasoning");
    addEventCard({
      title: `Progress: ${event.progress_pct}%`,
      text: `Step ${event.step}/${event.step_limit}\nPattern: ${event.pattern}\nTools executed: ${event.tools_executed}`,
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "tool_started") {
    addEventCard({
      title: `Tool Started: ${event.tool}`,
      text: `Step ${event.step}`,
      code: JSON.stringify(event.args || {}, null, 2),
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "tool_result") {
    addEventCard({
      title: `Tool Result: ${event.tool}`,
      text: `Step ${event.step}`,
      code: event.output || "",
      type: "tool",
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "format_retry") {
    addEventCard({
      title: "Format Retry",
      text: `Step ${event.step}: model output format invalid, retrying.`,
      type: "warning",
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "run_timeout") {
    setProgress(100, "timeout");
    addEventCard({
      title: "Run Timeout",
      text: event.message || "Max steps reached.",
      type: "warning",
      timestamp: event.timestamp,
    });
    return;
  }

  if (event.type === "run_completed") {
    setProgress(100, "completed");
    addEventCard({
      title: "Final Answer",
      text: event.final_answer || "",
      type: "final",
      timestamp: event.timestamp,
    });
    state.finalRendered = true;
    return;
  }

  if (event.type === "run_error") {
    setProgress(state.progressPct, "error");
    addEventCard({
      title: "Run Error",
      text: event.message || "Unknown error",
      type: "error",
      timestamp: event.timestamp,
    });
    return;
  }

  addEventCard({
    title: event.type || "Event",
    code: JSON.stringify(event, null, 2),
    timestamp: event.timestamp,
  });
}

function renderMemory(entries) {
  memoryListEl.innerHTML = "";
  if (!entries || entries.length === 0) {
    memoryListEl.textContent = "No memory yet. Completed runs will appear here.";
    return;
  }

  for (const item of entries.slice().reverse()) {
    const card = document.createElement("article");
    card.className = "memory-item";
    const head = document.createElement("div");
    head.className = "head";
    head.textContent = `${item.status} | ${new Date(item.finished_at).toLocaleTimeString()}`;
    const task = document.createElement("div");
    task.className = "task";
    task.textContent = item.task || "(no task)";
    const patterns = document.createElement("div");
    patterns.className = "patterns";
    patterns.textContent = `pattern: ${(item.patterns || []).join(", ") || "unknown"}`;
    const outcome = document.createElement("div");
    outcome.className = "outcome";
    const rawOutcome = String(item.outcome || "").replaceAll("\n", " ");
    outcome.textContent = rawOutcome.length > 180 ? `${rawOutcome.slice(0, 177)}...` : rawOutcome;
    card.appendChild(head);
    card.appendChild(task);
    card.appendChild(patterns);
    card.appendChild(outcome);
    memoryListEl.appendChild(card);
  }
}

function renderRunList(runs) {
  runListEl.innerHTML = "";
  if (!runs || runs.length === 0) {
    runListEl.textContent = "No runs yet.";
    return;
  }

  for (const run of runs) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `run-item ${run.id === state.activeRunId ? "active" : ""}`;
    item.addEventListener("click", () => {
      loadRun(run.id, true);
    });

    const task = document.createElement("div");
    task.className = "task";
    task.textContent = run.task.length > 110 ? `${run.task.slice(0, 107)}...` : run.task;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${run.status} | ${new Date(run.created_at).toLocaleTimeString()}`;

    item.appendChild(task);
    item.appendChild(meta);
    runListEl.appendChild(item);
  }
}

async function loadRuns() {
  try {
    const payload = await fetchJson("/api/runs");
    renderRunList(payload.runs || []);
  } catch (error) {
    addEventCard({ title: "Runs Refresh Failed", text: String(error), type: "error" });
  }
}

async function loadFiles() {
  try {
    const workspace = workspaceInputEl.value.trim() || ".";
    const payload = await fetchJson(`/api/files?workspace_path=${encodeURIComponent(workspace)}&limit=700`);
    fileTreeEl.textContent = payload.files.join("\n");
  } catch (error) {
    fileTreeEl.textContent = `error: ${String(error)}`;
  }
}

async function loadMemory() {
  try {
    const payload = await fetchJson("/api/memory?limit=30");
    renderMemory(payload.entries || []);
  } catch (error) {
    memoryListEl.textContent = `error: ${String(error)}`;
  }
}

async function clearMemory() {
  try {
    await fetchJson("/api/memory/clear", { method: "POST" });
    await loadMemory();
    addEventCard({ title: "Session Memory", text: "Cleared." });
  } catch (error) {
    addEventCard({ title: "Clear Memory Failed", text: String(error), type: "error" });
  }
}

function parseIntOrNull(value) {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseFloatOrNull(value) {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number.parseFloat(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`failed reading file: ${file.name}`));
    reader.onload = () => {
      const raw = String(reader.result || "");
      const marker = raw.indexOf(",");
      if (marker < 0) {
        reject(new Error(`invalid data url for file: ${file.name}`));
        return;
      }
      resolve(raw.slice(marker + 1));
    };
    reader.readAsDataURL(file);
  });
}

async function buildUploadPayload(files) {
  const entries = [];
  for (const file of files) {
    const encoded = await readFileAsBase64(file);
    entries.push({
      name: file.name,
      content_base64: encoded,
    });
  }
  return entries;
}

async function uploadWorkspaceFiles() {
  const files = Array.from(fileUploadInputEl.files || []);
  if (files.length === 0) {
    addEventCard({ title: "Upload Skipped", text: "Select at least one file first.", type: "warning" });
    return;
  }

  const workspace = workspaceInputEl.value.trim() || ".";
  const destination = uploadDestinationInputEl.value.trim() || ".";
  setStatus("Uploading", "running");
  addEventCard({
    title: "Uploading Files",
    text: `${files.length} file(s) to ${destination}`,
  });

  try {
    const payload = {
      workspace,
      destination,
      files: await buildUploadPayload(files),
    };
    const response = await fetchJson("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const uploaded = response.saved || [];
    addEventCard({
      title: "Upload Complete",
      text: uploaded.length > 0
        ? uploaded.map((item) => `${item.path} (${item.bytes} bytes)`).join("\n")
        : "No files uploaded.",
      type: "tool",
    });
    fileUploadInputEl.value = "";
    await loadFiles();
    setStatus("Idle", "idle");
  } catch (error) {
    addEventCard({ title: "Upload Failed", text: String(error), type: "error" });
    setStatus("Error", "error");
  }
}

async function uploadImageFile() {
  const selected = Array.from(imageUploadInputEl.files || []);
  if (selected.length === 0) {
    addEventCard({ title: "Image Upload Skipped", text: "Select an image first.", type: "warning" });
    return;
  }

  const imageFile = selected[0];
  const workspace = workspaceInputEl.value.trim() || ".";
  const destination = uploadDestinationInputEl.value.trim() || ".";
  setStatus("Uploading", "running");
  addEventCard({ title: "Uploading Image", text: `${imageFile.name} -> ${destination}` });

  try {
    const payload = {
      workspace,
      destination,
      files: await buildUploadPayload([imageFile]),
    };
    const response = await fetchJson("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const saved = response.saved || [];
    if (saved.length > 0) {
      const uploadedPath = saved[0].path;
      imagePathInputEl.value = uploadedPath;
      noOcrInputEl.checked = false;
      addEventCard({
        title: "Image Uploaded",
        text: `Image path set to ${uploadedPath}`,
        type: "tool",
      });
      await loadFiles();
    } else {
      addEventCard({ title: "Image Upload", text: "No image returned by server.", type: "warning" });
    }
    setStatus("Idle", "idle");
  } catch (error) {
    addEventCard({ title: "Image Upload Failed", text: String(error), type: "error" });
    setStatus("Error", "error");
  }
}

function clearImageSelection() {
  imageUploadInputEl.value = "";
  imagePathInputEl.value = "";
  addEventCard({ title: "Image Cleared", text: "OCR image path cleared." });
}

async function onSubmitTask(event) {
  event.preventDefault();
  const task = taskInputEl.value.trim();
  if (!task) {
    return;
  }

  runButtonEl.disabled = true;
  setStatus("Starting", "running");

  try {
    const payload = {
      task,
      workspace: workspaceInputEl.value.trim() || ".",
      image_path: imagePathInputEl.value.trim() || null,
      max_steps: parseIntOrNull(maxStepsInputEl.value),
      min_new_tokens: parseIntOrNull(minNewTokensInputEl.value),
      max_new_tokens: parseIntOrNull(maxNewTokensInputEl.value),
      temperature: parseFloatOrNull(temperatureInputEl.value),
      max_gpu_memory_gib: parseIntOrNull(maxGpuMemoryInputEl.value),
      coder_model: coderModelInputEl.value.trim() || null,
      lazy_load: lazyLoadInputEl.checked,
      sparse_load: sparseLoadInputEl.checked,
      no_shell: noShellInputEl.checked,
      no_ocr: noOcrInputEl.checked,
    };
    const response = await fetchJson("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeRunId = response.run_id;
    state.nextIndex = 0;
    state.finalRendered = false;
    setProgress(0, "planning");
    timelineEl.innerHTML = "";
    addEventCard({ title: "Task Submitted", text: task });
    setStatus("Running", "running");
    await loadRuns();
    await loadMemory();
    await loadRun(state.activeRunId, false);
    startPolling();
  } catch (error) {
    addEventCard({ title: "Run Start Failed", text: String(error), type: "error" });
    setStatus("Error", "error");
    runButtonEl.disabled = false;
  }
}

async function loadRun(runId, resetTimeline) {
  try {
    if (resetTimeline) {
      timelineEl.innerHTML = "";
      state.nextIndex = 0;
      state.finalRendered = false;
      state.activeRunId = runId;
      startPolling();
    }

    const payload = await fetchJson(`/api/runs/${encodeURIComponent(runId)}?since=${state.nextIndex}`);
    state.nextIndex = payload.next_index;
    for (const event of payload.events || []) {
      renderEvent(event);
    }

    if (payload.status === "completed") {
      setStatus("Completed", "idle");
      runButtonEl.disabled = false;
      await loadMemory();
      if (!state.finalRendered && payload.final_answer) {
        addEventCard({ title: "Final Answer", text: payload.final_answer, type: "final" });
        state.finalRendered = true;
      }
      stopPolling();
    } else if (payload.status === "error") {
      setStatus("Error", "error");
      runButtonEl.disabled = false;
      if (payload.error) {
        addEventCard({ title: "Run Error", text: payload.error, type: "error" });
      }
      await loadMemory();
      stopPolling();
    } else {
      setStatus("Running", "running");
    }
  } catch (error) {
    addEventCard({ title: "Run Load Failed", text: String(error), type: "error" });
    setStatus("Error", "error");
    runButtonEl.disabled = false;
    stopPolling();
  }
}

function startPolling() {
  stopPolling();
  state.pollHandle = window.setInterval(async () => {
    if (!state.activeRunId) {
      return;
    }
    await loadRun(state.activeRunId, false);
    await loadRuns();
  }, 1200);
}

function stopPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

async function boot() {
  addEventCard({
    title: "DeepSeek Agent GUI",
    text: "Ready. Submit a task to start an agent run.",
  });
  await loadFiles();
  await loadRuns();
  await loadMemory();
  setProgress(0, "idle");
  state.runRefreshHandle = window.setInterval(loadRuns, 5000);
}

boot();
