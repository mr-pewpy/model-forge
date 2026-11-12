// Model Forge — cockpit pane.
//
// Targets + engines + live model counts for every compute target. Target
// state is the union of both places the plugin keeps it (mirroring the
// backend's targets.load_targets()):
//   1. tool-registered targets — what model_forge_target_add persists in the
//      plugin data dir (plugin-data/model-forge/targets.json), and
//   2. the "Compute targets" Desktop setting (targets_json) — which wins on
//      id conflicts, exactly like the backend store.
// A hardcoded "This machine" default fills in only when neither source
// declares it. Engine probes run directly against each target's HTTP API
// (no generation, just /props, /version, /v1/models and friends) with the
// same fingerprints and probe ports as the plugin's engine adapters. A
// blocked fetch degrades to a hint row; a source read failure is surfaced,
// never silently defaulted away.

import { host, PALETTE_AREA } from "@hermes/plugin-sdk";
import { jsx } from "react/jsx-runtime";
import { useEffect, useState } from "react";

const DEFAULT_TARGETS = [
  { id: "local", label: "This machine", base_url: "http://127.0.0.1:8080", kind: "local", source: "default" },
];

// Local scan ports mirror the engine adapters' probe ports
// (model_forge/engines/): llama.cpp/Ollama PROBE_PORTS 8080/8081/8000/8001/
// 11434, vLLM default 8000 (covered above), ComfyUI 8188, SGLang 30000.
const LOCAL_PROBE_PORTS = [8080, 8081, 8000, 8001, 11434, 8188, 30000];
const PROBE_TIMEOUT_MS = 2500;

// Set by the pane on mount so the palette command can force a reload.
let refreshHook = null;

function rootUrl(base) {
  return String(base || "").replace(/\/+$/, "").replace(/\/v1$/, "");
}

// --- target sources -----------------------------------------------------------

async function readDataDirTargets() {
  // Tool-registered targets (model_forge_target_add) persist via the plugin
  // backend's target store in the plugin data dir; the gateway's shell.exec
  // is the pane's read door to it. A missing file is the normal empty state
  // (no tool has registered a target) — only real read failures are errors.
  try {
    const res = await host.request("shell.exec", {
      command: "cat \"$HERMES_HOME/plugin-data/model-forge/targets.json\" 2>/dev/null",
    });
    if (res && res.blocked) {
      return { targets: [], error: "tool-registered targets unreadable: " + (res.hint || "blocked") };
    }
    const raw = String((res && res.stdout) || "").trim();
    if (!raw) return { targets: [], error: "" };
    const rows = JSON.parse(raw);
    const targets = (Array.isArray(rows) ? rows : [])
      .filter((t) => t && t.id && t.base_url)
      .map((t) => ({
        id: t.id,
        label: t.label || t.id,
        base_url: t.base_url,
        kind: t.kind || "lan",
        vram_gb: t.vram_gb || 0,
        ram_gb: t.ram_gb || 0,
        source: "tool",
      }));
    return { targets, error: "" };
  } catch (err) {
    return { targets: [], error: "tool-registered targets unreadable: " + ((err && err.message) || err) };
  }
}

async function readSettingsTargets() {
  try {
    const res = await host.request("plugins.manage", { action: "list" });
    // v6 rows: an array of { key, name, settings_schema: [{ key, value }] };
    // older shims: an entries dict with .settings. Support both shapes.
    let raw = null;
    if (res && Array.isArray(res.plugins)) {
      const entry = res.plugins.find((r) => r && (r.key === "model-forge" || r.name === "model-forge")) || {};
      const field = (entry.settings_schema || []).find((f) => f && f.key === "targets_json") || {};
      raw = field.value;
    } else {
      const entries = (res && res.entries) || {};
      raw = ((entries["model-forge"] || {}).settings || {}).targets_json;
    }
    const rows = Array.isArray(raw) ? raw : [];
    const targets = rows
      .filter((t) => t && t.id && t.base_url)
      .map((t) => ({
        id: t.id,
        label: t.label || t.id,
        base_url: t.base_url,
        kind: t.kind || "lan",
        vram_gb: t.vram_gb || 0,
        ram_gb: t.ram_gb || 0,
        source: "settings",
      }));
    return { targets, error: "" };
  } catch (err) {
    return { targets: [], error: "settings targets unreadable: " + ((err && err.message) || err) };
  }
}

async function readTargets() {
  const [tool, settings] = await Promise.all([
    readDataDirTargets(),
    readSettingsTargets(),
  ]);
  // Merge like targets.load_targets(): tool-registered rows first, settings
  // replace by id (settings win), the local default first only if missing.
  const byId = new Map();
  tool.targets.forEach((t) => byId.set(t.id, t));
  settings.targets.forEach((t) => byId.set(t.id, t));
  let list = [...byId.values()];
  if (!list.some((t) => t.id === "local")) {
    list = [...DEFAULT_TARGETS, ...list];
  }
  return { targets: list, errors: [tool.error, settings.error].filter(Boolean) };
}

// --- engine probes ------------------------------------------------------------

async function probeRoot(root) {
  const tryFetch = async (path) => {
    try {
      const r = await fetch(root + path, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (r.ok) return await r.json();
    } catch (err) { /* offline, CORS-blocked, or timed out */ }
    return null;
  };
  // Fingerprints mirror the plugin's engine adapters (model_forge/engines/):
  // llama.cpp /props, Ollama /api/tags, Maya /api/catalog (which plain Ollama
  // never serves), ComfyUI /system_stats, vLLM /version, SGLang
  // /get_model_info; /v1/models alone never identifies an engine.
  const [props, tags, catalog, stats, version, modelInfo, models] = await Promise.all([
    tryFetch("/props"),
    tryFetch("/api/tags"),
    tryFetch("/api/catalog"),
    tryFetch("/system_stats"),
    tryFetch("/version"),
    tryFetch("/get_model_info"),
    tryFetch("/v1/models"),
  ]);
  const engines = [];
  if (stats && (stats.system || stats.devices)) {
    engines.push({
      name: "ComfyUI", ok: true,
      detail: (stats.system || {})["comfyui-version"] || "running",
    });
  }
  if (version && typeof version.version !== "undefined") {
    engines.push({ name: "vLLM", ok: true, detail: "vllm " + version.version });
  }
  if (modelInfo && typeof modelInfo === "object" && !Array.isArray(modelInfo)) {
    const model = modelInfo.model_path || modelInfo.model || "";
    engines.push({
      name: "SGLang", ok: true,
      detail: model ? "model " + model : "running",
    });
  }
  if (props && (props.build_info || props.model_path)) {
    const model = props.model_path || props.model || "";
    engines.push({
      name: "llama.cpp (llama-server)", ok: true,
      detail: model ? "serving " + model : "running",
    });
  } else if (tags && Array.isArray(tags.models)) {
    if (catalog && Array.isArray(catalog.models)) {
      // Maya serves the Ollama-style /api/tags too; /api/catalog is its
      // fingerprint (same no-double-label rule as the adapters).
      engines.push({
        name: "Project Maya", ok: true,
        detail: "catalog: " + catalog.models.slice(0, 3)
          .map((m) => m.name || "").filter(Boolean).join(", "),
      });
    } else {
      engines.push({
        name: "Ollama", ok: true,
        detail: "models: " + tags.models.slice(0, 3)
          .map((m) => m.name || "").filter(Boolean).join(", "),
        models: tags.models.length,
      });
    }
  }
  if (models && Array.isArray(models.data)) {
    const named = engines.some((e) => e.ok);
    if (!named) {
      // Same server already claimed by a distinctive fingerprint? Then no
      // generic row — and /v1/models alone never names the engine.
      engines.push({
        name: "OpenAI-compatible", ok: true,
        detail: models.data.slice(0, 6).map((m) => m.id).join(", ") +
          " — engine identity unknown",
        models: models.data.length,
      });
    }
  }
  return engines;
}

async function probeTarget(target) {
  const roots = target.kind === "local"
    ? LOCAL_PROBE_PORTS.map((p) => "http://127.0.0.1:" + p)
    : [rootUrl(target.base_url)];
  const perRoot = await Promise.all(
    roots.map(async (root) => ({ root, engines: await probeRoot(root) })));
  const engines = [];
  perRoot.forEach(({ root, engines: es }) => {
    es.forEach((e) => engines.push({ ...e, root }));
  });
  if (!engines.some((e) => e.ok)) {
    const shown = roots.slice(0, 4).join(", ") + (roots.length > 4 ? ", …" : "");
    engines.push({
      name: "none detected", ok: false,
      detail: "no engine answered at " + shown +
        " (or the app blocked the probe) — use the model_forge_status tool",
    });
  }
  return { target, engines };
}

// --- UI -----------------------------------------------------------------------

function Row(props) {
  return jsx("div", {
    style: {
      display: "flex", alignItems: "baseline", gap: "8px",
      padding: "4px 0", borderBottom: "1px solid var(--ui-stroke-secondary)",
    },
    children: props.children,
  });
}

function EngineRow(props) {
  const e = props.engine;
  return jsx(Row, {
    children: [
      jsx("span", {
        key: "n",
        style: { color: e.ok ? "var(--ui-accent)" : "var(--ui-text-quaternary)",
                 minWidth: "130px" },
        children: e.name,
      }),
      jsx("span", {
        key: "d",
        style: { color: "var(--ui-text-secondary)", fontSize: "12px" },
        children: e.detail,
      }),
    ],
  });
}

function Cockpit(props) {
  const ctx = props.ctx;
  const [rows, setRows] = useState(null);
  const [errors, setErrors] = useState([]);
  const load = () => {
    readTargets().then(({ targets, errors: errs }) => {
      setErrors(errs);
      return Promise.all(targets.map(probeTarget));
    }).then((probed) => {
      if (probed) setRows(probed);
    });
  };
  useEffect(() => {
    load();
    refreshHook = load;
    const disposer = ctx.onEvent("plugin.model-forge.settings.changed", load);
    return () => {
      refreshHook = null;
      disposer();
    };
  }, []);
  if (!rows) {
    return jsx("div", {
      style: { color: "var(--ui-text-secondary)", padding: "8px" },
      children: "Probing compute targets…",
    });
  }
  const children = [];
  errors.forEach((msg, i) => {
    children.push(jsx("div", {
      key: "err-" + i,
      style: { color: "var(--ui-text-quaternary)", fontSize: "11px",
               padding: "2px 0" },
      children: msg,
    }));
  });
  rows.forEach((r) => {
    children.push(jsx("div", {
      key: "t-" + r.target.id,
      style: { margin: "10px 0 4px", fontWeight: 600 },
      children: (r.target.label || r.target.id) +
        " · " + (r.target.kind || "") +
        (r.target.vram_gb ? " · " + r.target.vram_gb + " GB VRAM" : "") +
        (r.target.ram_gb ? " · " + r.target.ram_gb + " GB RAM" : "") +
        (r.target.source ? " · " + r.target.source : ""),
    }));
    children.push(jsx("div", {
      key: "u-" + r.target.id,
      style: { color: "var(--ui-text-quaternary)", fontSize: "11px" },
      children: r.target.base_url,
    }));
    r.engines.forEach((e, i) => {
      children.push(jsx(EngineRow, {
        key: "e-" + r.target.id + "-" + i, engine: e,
      }));
    });
  });
  return jsx("div", {
    style: { padding: "8px", overflowY: "auto" },
    children,
  });
}

export default {
  id: "model-forge-desktop",
  name: "Model Forge",
  register(ctx) {
    ctx.register({
      id: "model-forge-desktop-pane",
      area: "panes",
      title: "Model Forge",
      data: { placement: "right" },
      render: () => jsx(Cockpit, { ctx }),
    });
    ctx.register({
      id: "model-forge-desktop-refresh",
      area: PALETTE_AREA,
      title: "Model Forge: refresh cockpit",
      run: () => {
        if (refreshHook) refreshHook();
        host.notify({ kind: "info", message: "Model Forge cockpit refreshed." });
      },
    });
  },
};
