// RAW AI Denoise - file dialogs, drag & drop and "queue all" for the loader nodes.
//
// Two ways to pick files:
//  * "Browse this computer…": an in-app dialog listing folders on the machine
//    that runs ComfyUI. Files are used where they are (no copying) - best for
//    big video files. Only offered when ComfyUI is opened on that machine.
//  * "Upload…" / drag & drop: the browser's own file dialog. Browsers never
//    reveal a file's path, so the files are copied into ComfyUI/input/raw or
//    input/video through a streaming upload route (no size cap).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const RAW_NODE = "RawDenoise_LoadRAW";
const VIDEO_NODE = "RawDenoise_LoadVideo";
const VIDEO_DENOISE_NODE = "RawDenoise_VideoDenoise";
const RENDER_MODE = "render full video";

const RAW_EXT = [
  ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf",
  ".fff", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw",
  ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
];
const VIDEO_EXT = [
  ".mp4", ".mov", ".mxf", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".webm", ".mpg",
  ".mpeg", ".wmv", ".3gp", ".ts", ".hevc", ".h264", ".dv", ".flv",
];

const extOf = (name) => {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
};
const isRaw = (name) => RAW_EXT.includes(extOf(name));
const isVideo = (name) => VIDEO_EXT.includes(extOf(name));
const baseName = (p) => String(p).split(/[\\/]/).pop();
const widget = (node, name) => node.widgets?.find((w) => w.name === name);

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: severity === "error" ? 10000 : 4500 });
  else if (severity === "error") alert(`${summary}\n${detail ?? ""}`);
  else console.log(`[RAW Denoise] ${summary} ${detail ?? ""}`);
}

function setLabel(node, w, text) {
  if (!w) return;
  w.label = text;
  node.setDirtyCanvas?.(true, true);
}

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w) return;
  w.value = value;
  w.callback?.(value);
}

function addButton(node, name, label, fn) {
  const b = node.addWidget("button", name, null, fn);
  // never serialised: saved workflows keep their widget order
  b.serialize = false;
  b.options = { ...(b.options ?? {}), serialize: false };
  setLabel(node, b, label);
  return b;
}

// ---------------------------------------------------------------------------
// browser file dialog + streaming upload
// ---------------------------------------------------------------------------

function pickLocalFiles(exts) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = exts.join(",");
    input.style.display = "none";
    document.body.appendChild(input);
    const done = (files) => { input.remove(); resolve(files); };
    input.addEventListener("change", () => done([...(input.files ?? [])]));
    input.addEventListener("cancel", () => done([]));
    input.click();
  });
}

async function uploadFiles(node, btn, files, kind, okFn, resetLabel) {
  const good = files.filter((f) => okFn(f.name));
  const skipped = files.filter((f) => !okFn(f.name)).map((f) => f.name);
  if (skipped.length) toast("warn", `Not ${kind === "video" ? "video" : "RAW"} files - skipped`, skipped.join(", "));
  const chosen = [];
  try {
    for (let i = 0; i < good.length; i++) {
      const f = good[i];
      setLabel(node, btn, `Copying ${i + 1}/${good.length}: ${f.name} (${(f.size / 1e6).toFixed(0)} MB)…`);
      const body = new FormData();
      body.append("file", f, f.name);
      const resp = await api.fetchApi(`/raw_denoise/upload?kind=${kind}`, { method: "POST", body });
      if (resp.status !== 200) throw new Error(`${f.name}: server answered ${resp.status} ${await resp.text()}`);
      const j = await resp.json();
      chosen.push(...j.files);
      if (j.rejected?.length) toast("warn", "Rejected by server", j.rejected.join(", "));
    }
  } catch (e) {
    toast("error", "Copying the file failed", String(e?.message ?? e));
    return [];
  } finally {
    setLabel(node, btn, resetLabel);
  }
  return chosen;
}

// ---------------------------------------------------------------------------
// "Browse this computer…" dialog (lists folders on the ComfyUI machine)
// ---------------------------------------------------------------------------

const LAST_DIR_KEY = (kind) => `rawdenoise.lastdir.${kind}`;
const fmtSize = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${Math.ceil(n / 1e3)} kB`);

function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) e.style.cssText = style;
  if (text != null) e.textContent = text;
  return e;
}

async function browseRequest(path, kind) {
  const q = new URLSearchParams({ kind, path: path ?? "" });
  const resp = await api.fetchApi(`/raw_denoise/browse?${q}`);
  const j = await resp.json().catch(() => ({ error: "bad", message: `HTTP ${resp.status}` }));
  if (resp.status !== 200) throw Object.assign(new Error(j.message || `HTTP ${resp.status}`), { code: j.error });
  return j;
}

function openBrowser({ kind = "video", multiple = true, title = "Choose files" } = {}) {
  return new Promise(async (resolve) => {
    let start = "";
    try { start = localStorage.getItem(LAST_DIR_KEY(kind)) || ""; } catch { /* storage blocked */ }
    let listing;
    try {
      listing = await browseRequest(start, kind).catch(() => browseRequest("", kind));
    } catch (e) {
      toast(e.code === "remote" ? "warn" : "error", "Can't browse folders", e.message);
      resolve(null);
      return;
    }

    const overlay = el("div", "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;");
    overlay.className = "rawdenoise-browser";
    const box = el("div", "background:var(--comfy-menu-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#444);border-radius:8px;width:min(860px,94vw);height:min(640px,88vh);display:flex;flex-direction:column;font:13px system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,.5);");
    const head = el("div", "display:flex;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--border-color,#444);");
    const titleEl = el("div", "font-weight:600;white-space:nowrap;", title);
    const up = el("button", "padding:3px 10px;cursor:pointer;", "⬆ Up");
    const pathInput = el("input", "flex:1;min-width:0;padding:4px 6px;background:var(--comfy-input-bg,#111);color:inherit;border:1px solid var(--border-color,#555);border-radius:4px;");
    pathInput.spellcheck = false;
    head.append(titleEl, up, pathInput);
    const bodyRow = el("div", "flex:1;display:flex;min-height:0;");
    const side = el("div", "width:170px;overflow:auto;border-right:1px solid var(--border-color,#444);padding:6px 0;");
    const list = el("div", "flex:1;overflow:auto;padding:4px 0;");
    bodyRow.append(side, list);
    const foot = el("div", "display:flex;gap:8px;align-items:center;padding:10px 12px;border-top:1px solid var(--border-color,#444);");
    const status = el("div", "flex:1;opacity:.8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;");
    const allBtn = el("button", "padding:5px 10px;cursor:pointer;", "Select all");
    const cancel = el("button", "padding:5px 14px;cursor:pointer;", "Cancel");
    const ok = el("button", "padding:5px 16px;cursor:pointer;font-weight:600;", "Use selected");
    foot.append(status, ...(multiple ? [allBtn] : []), cancel, ok);
    box.append(head, bodyRow, foot);
    overlay.append(box);
    document.body.append(overlay);

    const selected = new Set();
    const close = (val) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(val);
    };
    const join = (dir, name) => (dir.endsWith(listing.sep) ? dir + name : dir + listing.sep + name);
    const updateStatus = () => {
      status.textContent = selected.size
        ? `${selected.size} selected: ${[...selected].map(baseName).slice(0, 3).join(", ")}${selected.size > 3 ? "…" : ""}`
        : `${listing.files.length} ${kind === "video" ? "video" : "RAW"} file(s) here - click to select${multiple ? " (Ctrl/Shift for several)" : ""}, double-click to use`;
      ok.disabled = selected.size === 0;
    };
    let lastClicked = -1;

    const render = () => {
      pathInput.value = listing.path;
      up.disabled = !listing.parent;
      side.replaceChildren();
      for (const p of listing.places || []) {
        const b = el("div", "padding:5px 12px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", p.label);
        b.title = p.path;
        b.onmouseenter = () => (b.style.background = "rgba(255,255,255,.07)");
        b.onmouseleave = () => (b.style.background = "");
        b.onclick = () => go(p.path);
        side.append(b);
      }
      list.replaceChildren();
      for (const d of listing.dirs) {
        const row = el("div", "padding:4px 12px;cursor:pointer;display:flex;gap:8px;", null);
        row.append(el("span", "", "📁"), el("span", "flex:1;", d));
        row.onmouseenter = () => (row.style.background = "rgba(255,255,255,.07)");
        row.onmouseleave = () => (row.style.background = "");
        row.ondblclick = row.onclick = () => go(join(listing.path, d));
        list.append(row);
      }
      listing.files.forEach((f, idx) => {
        const full = join(listing.path, f.name);
        const row = el("div", "padding:4px 12px;cursor:pointer;display:flex;gap:8px;user-select:none;", null);
        row.dataset.path = full;
        row.append(el("span", "", kind === "video" ? "🎞️" : "🖼️"), el("span", "flex:1;", f.name),
          el("span", "opacity:.7;", fmtSize(f.size)));
        const paint = () => (row.style.background = selected.has(full) ? "rgba(80,140,255,.35)" : "");
        row.onclick = (ev) => {
          if (multiple && ev.shiftKey && lastClicked >= 0) {
            const [a, b] = [Math.min(lastClicked, idx), Math.max(lastClicked, idx)];
            for (let k = a; k <= b; k++) selected.add(join(listing.path, listing.files[k].name));
          } else if (multiple && (ev.ctrlKey || ev.metaKey)) {
            selected.has(full) ? selected.delete(full) : selected.add(full);
          } else {
            selected.clear();
            selected.add(full);
          }
          lastClicked = idx;
          list.querySelectorAll("[data-path]").forEach((r) => (r.style.background = selected.has(r.dataset.path) ? "rgba(80,140,255,.35)" : ""));
          updateStatus();
        };
        row.ondblclick = () => { selected.clear(); selected.add(full); finish(); };
        paint();
        list.append(row);
      });
      if (!listing.dirs.length && !listing.files.length) list.append(el("div", "padding:12px;opacity:.6;", "(empty folder)"));
      updateStatus();
    };
    const go = async (p) => {
      try {
        listing = await browseRequest(p, kind);
        selected.clear();
        lastClicked = -1;
        try { localStorage.setItem(LAST_DIR_KEY(kind), listing.path); } catch { /* ignore */ }
        render();
      } catch (e) {
        toast("error", "Can't open folder", e.message);
      }
    };
    const finish = () => {
      if (!selected.size) return;
      try { localStorage.setItem(LAST_DIR_KEY(kind), listing.path); } catch { /* ignore */ }
      // keep the on-screen order
      const order = listing.files.map((f) => join(listing.path, f.name));
      close(order.filter((p) => selected.has(p)));
    };
    const onKey = (ev) => {
      if (ev.key === "Escape") { ev.stopPropagation(); close(null); }
      if (ev.key === "Enter" && document.activeElement !== pathInput) { ev.stopPropagation(); finish(); }
    };
    document.addEventListener("keydown", onKey, true);
    up.onclick = () => listing.parent && go(listing.parent);
    pathInput.onkeydown = (ev) => { if (ev.key === "Enter") { ev.stopPropagation(); go(pathInput.value); } };
    allBtn.onclick = () => { listing.files.forEach((f) => selected.add(join(listing.path, f.name))); render(); };
    cancel.onclick = () => close(null);
    ok.onclick = finish;
    overlay.onclick = (ev) => { if (ev.target === overlay) close(null); };
    render();
  });
}

// ---------------------------------------------------------------------------
// Load RAW Image
// ---------------------------------------------------------------------------

function refreshRawQueue(node) {
  const n = node.properties?.raw_selection?.length ?? 0;
  setLabel(node, node.__rawQueueBtn,
    n > 1 ? `▶ Queue all ${n} selected files` : "▶ Queue all selected files (choose 2+ first)");
}

function selectRaw(node, chosen) {
  if (!chosen.length) return;
  const first = chosen[0];
  if (/^([A-Za-z]:[\\/]|\/)/.test(first)) {
    // absolute path from the folder browser: used in place via path_override
    setWidget(node, "path_override", first);
  } else {
    const fileW = widget(node, "raw_file");
    if (fileW) {
      const known = (fileW.options?.values ?? []).filter((v) => !String(v).startsWith("("));
      fileW.options.values = [...new Set([...known, ...chosen])].sort();
      setWidget(node, "raw_file", first);
    }
    setWidget(node, "path_override", "");
  }
  node.properties = node.properties ?? {};
  node.properties.raw_selection = chosen;
  refreshRawQueue(node);
  toast("success", chosen.length === 1 ? "RAW file ready" : `${chosen.length} RAW files ready`,
    chosen.length === 1 ? `${baseName(first)} - press Run` : "Press Run for the first one, or 'Queue all' for every file");
}

async function queueAllRaw(node) {
  const files = node.properties?.raw_selection ?? [];
  if (files.length < 2) {
    toast("info", "Nothing to batch", "Choose several files first (Ctrl/Shift-click). Run processes the current file.");
    return;
  }
  for (const f of files) {
    if (/^([A-Za-z]:[\\/]|\/)/.test(f)) {
      setWidget(node, "path_override", f);
    } else {
      setWidget(node, "path_override", "");
      setWidget(node, "raw_file", f);
    }
    node.setDirtyCanvas?.(true, true);
    await app.queuePrompt(0, 1);
  }
  toast("success", `Queued ${files.length} files`, "Each one is denoised and saved in turn.");
}

// ---------------------------------------------------------------------------
// Load Video (for AI Denoise)
// ---------------------------------------------------------------------------

function refreshVideoQueue(node) {
  const n = node.properties?.video_selection?.length ?? 0;
  setLabel(node, node.__vidQueueBtn,
    n > 1 ? `▶ Render all ${n} selected videos` : "▶ Render all selected videos (choose 2+ first)");
}

function selectVideos(node, chosen) {
  if (!chosen?.length) return;
  setWidget(node, "video", chosen[0]);
  node.properties = node.properties ?? {};
  node.properties.video_selection = chosen;
  refreshVideoQueue(node);
  toast("success", chosen.length === 1 ? "Video ready" : `${chosen.length} videos ready`,
    chosen.length === 1 ? `${baseName(chosen[0])} - press Run for a preview`
      : "Run previews the first one; 'Render all' processes every video");
}

function downstreamDenoisers() {
  return (app.graph?._nodes ?? []).filter((n) => n.type === VIDEO_DENOISE_NODE && n.mode === 0);
}

async function queueAllVideos(node) {
  const files = node.properties?.video_selection ?? [];
  if (files.length < 2) {
    toast("info", "Nothing to batch", "Choose several videos first (Ctrl/Shift-click in the browser dialog).");
    return;
  }
  const dens = downstreamDenoisers();
  const saved = dens.map((d) => widget(d, "mode")?.value);
  try {
    dens.forEach((d) => setWidget(d, "mode", RENDER_MODE));
    for (const f of files) {
      setWidget(node, "video", f);
      node.setDirtyCanvas?.(true, true);
      await app.queuePrompt(0, 1);
    }
  } finally {
    dens.forEach((d, i) => saved[i] != null && setWidget(d, "mode", saved[i]));
  }
  toast("success", `Queued ${files.length} videos`, "Each one is rendered in full, one after the other.");
}

// ---------------------------------------------------------------------------

app.registerExtension({
  name: "RawDenoise.FilePicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === RAW_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const label = "📂 Choose RAW file(s)…";
        const pick = addButton(this, "choose_raw_files", label, async () => {
          const files = await pickLocalFiles(RAW_EXT);
          if (files.length) selectRaw(this, await uploadFiles(this, pick, files, "raw", isRaw, label));
        });
        addButton(this, "browse_raw_files", "📁 Browse this computer (no copy)…", async () => {
          const chosen = await openBrowser({ kind: "raw", title: "Choose RAW file(s)" });
          if (chosen?.length) selectRaw(this, chosen);
        });
        this.__rawPickBtn = pick;
        this.__rawQueueBtn = addButton(this, "queue_all_selected", "", () => queueAllRaw(this));
        refreshRawQueue(this);
        return r;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        refreshRawQueue(this);
        return r;
      };
      nodeType.prototype.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
      nodeType.prototype.onDragDrop = async function (e) {
        const files = [...(e?.dataTransfer?.files ?? [])];
        if (!files.some((f) => isRaw(f.name))) return false;
        selectRaw(this, await uploadFiles(this, this.__rawPickBtn, files, "raw", isRaw, "📂 Choose RAW file(s)…"));
        return true;
      };
    }

    if (nodeData?.name === VIDEO_NODE) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        addButton(this, "browse_videos", "📁 Browse this computer…", async () => {
          const chosen = await openBrowser({ kind: "video", title: "Choose video(s)" });
          if (chosen?.length) selectVideos(this, chosen);
        });
        const upLabel = "⬆ Upload video(s) from this browser…";
        const upBtn = addButton(this, "upload_videos", upLabel, async () => {
          const files = await pickLocalFiles(VIDEO_EXT);
          if (files.length) selectVideos(this, await uploadFiles(this, upBtn, files, "video", isVideo, upLabel));
        });
        this.__vidUpBtn = upBtn;
        this.__vidQueueBtn = addButton(this, "render_all_selected", "", () => queueAllVideos(this));
        refreshVideoQueue(this);
        return r;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        refreshVideoQueue(this);
        return r;
      };
      nodeType.prototype.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
      nodeType.prototype.onDragDrop = async function (e) {
        const files = [...(e?.dataTransfer?.files ?? [])];
        if (!files.some((f) => isVideo(f.name))) return false;
        selectVideos(this, await uploadFiles(this, this.__vidUpBtn, files, "video", isVideo,
          "⬆ Upload video(s) from this browser…"));
        return true;
      };
    }
  },
});
