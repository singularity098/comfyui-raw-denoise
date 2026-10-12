// RAW AI Denoise - file dialog, drag & drop and "process all" for the Load RAW Image node.
//
// Browsers don't expose a picked file's path, so the chosen RAW files are
// copied into ComfyUI/input/raw/ through a streaming upload route (no size
// cap, duplicates are detected by content) and then selected on the node.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "RawDenoise_LoadRAW";
const RAW_EXT = [
  ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf",
  ".fff", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw",
  ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
];

const isRaw = (name) => {
  const i = name.lastIndexOf(".");
  return i >= 0 && RAW_EXT.includes(name.slice(i).toLowerCase());
};

const widget = (node, name) => node.widgets?.find((w) => w.name === name);

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: severity === "error" ? 10000 : 4000 });
  else if (severity === "error") alert(`${summary}\n${detail ?? ""}`);
  else console.log(`[RAW Denoise] ${summary} ${detail ?? ""}`);
}

function pickFiles() {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = RAW_EXT.join(",");
    input.style.display = "none";
    document.body.appendChild(input);
    const done = (files) => { input.remove(); resolve(files); };
    input.addEventListener("change", () => done([...(input.files ?? [])]));
    input.addEventListener("cancel", () => done([]));
    input.click();
  });
}

function setLabel(node, w, text) {
  if (!w) return;
  w.label = text;
  node.setDirtyCanvas?.(true, true);
}

function refreshQueueButton(node) {
  const n = node.properties?.raw_selection?.length ?? 0;
  setLabel(node, node.__rawQueueBtn,
    n > 1 ? `▶ Queue all ${n} selected files` : "▶ Queue all selected files (choose 2+ first)");
}

async function uploadAndSelect(node, files) {
  const raws = files.filter((f) => isRaw(f.name));
  const skipped = files.filter((f) => !isRaw(f.name)).map((f) => f.name);
  if (skipped.length) toast("warn", "Not RAW files - skipped", skipped.join(", "));
  if (!raws.length) return;

  const btn = node.__rawPickBtn;
  const chosen = [];
  try {
    for (let i = 0; i < raws.length; i++) {
      const f = raws[i];
      setLabel(node, btn, `Copying ${i + 1}/${raws.length}: ${f.name} (${(f.size / 1e6).toFixed(0)} MB)…`);
      const body = new FormData();
      body.append("file", f, f.name);
      const resp = await api.fetchApi("/raw_denoise/upload", { method: "POST", body });
      if (resp.status !== 200) {
        throw new Error(`${f.name}: server answered ${resp.status} ${await resp.text()}`);
      }
      const j = await resp.json();
      chosen.push(...j.files);
      if (j.rejected?.length) toast("warn", "Rejected by server", j.rejected.join(", "));
    }
  } catch (e) {
    toast("error", "Copying the RAW file failed", String(e?.message ?? e));
    return;
  } finally {
    setLabel(node, btn, "📂 Choose RAW file(s)…");
  }
  if (!chosen.length) return;

  const fileW = widget(node, "raw_file");
  if (fileW) {
    const known = (fileW.options?.values ?? []).filter((v) => !String(v).startsWith("("));
    fileW.options.values = [...new Set([...known, ...chosen])].sort();
    fileW.value = chosen[0];
    fileW.callback?.(fileW.value);
  }
  // a path in path_override would take precedence over the picked file
  const pathW = widget(node, "path_override");
  if (pathW && pathW.value) {
    pathW.value = "";
    pathW.callback?.("");
  }
  node.properties = node.properties ?? {};
  node.properties.raw_selection = chosen;
  refreshQueueButton(node);
  node.setDirtyCanvas?.(true, true);
  toast("success", chosen.length === 1 ? "RAW file ready" : `${chosen.length} RAW files ready`,
    chosen.length === 1 ? `${chosen[0]} - press Run` : "Press Run for the first one, or 'Queue all' for every file");
}

async function queueAll(node) {
  const files = node.properties?.raw_selection ?? [];
  if (files.length < 2) {
    toast("info", "Nothing to batch", "Use 'Choose RAW file(s)…' and select several files (Ctrl/Shift-click). Run processes the file shown in raw_file.");
    return;
  }
  const fileW = widget(node, "raw_file");
  const pathW = widget(node, "path_override");
  if (pathW?.value) { pathW.value = ""; pathW.callback?.(""); }
  for (const f of files) {
    fileW.value = f;
    fileW.callback?.(f);
    node.setDirtyCanvas?.(true, true);
    await app.queuePrompt(0, 1);
  }
  toast("success", `Queued ${files.length} files`, "Each one is denoised and saved in turn.");
}

app.registerExtension({
  name: "RawDenoise.FilePicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      // buttons are appended after the real widgets and never serialised,
      // so saved workflows keep their widget order
      const pick = this.addWidget("button", "choose_raw_files", null, async () => {
        const files = await pickFiles();
        if (files.length) await uploadAndSelect(this, files);
      });
      const queue = this.addWidget("button", "queue_all_selected", null, () => queueAll(this));
      for (const b of [pick, queue]) {
        b.serialize = false;
        b.options = { ...(b.options ?? {}), serialize: false };
      }
      this.__rawPickBtn = pick;
      this.__rawQueueBtn = queue;
      setLabel(this, pick, "📂 Choose RAW file(s)…");
      refreshQueueButton(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      refreshQueueButton(this);
      return r;
    };

    // drag RAW files from the file manager onto the node
    nodeType.prototype.onDragOver = function (e) {
      return !!e?.dataTransfer?.types?.includes?.("Files");
    };
    nodeType.prototype.onDragDrop = async function (e) {
      const files = [...(e?.dataTransfer?.files ?? [])];
      if (!files.some((f) => isRaw(f.name))) return false;
      await uploadAndSelect(this, files);
      return true;
    };
  },
});
