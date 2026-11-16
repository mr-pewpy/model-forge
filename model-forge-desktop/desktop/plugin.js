// Model Forge — product cockpit pane.
//
// Four sections, all read-only:
//   1. GPU & compute targets — hardware + targets, with the pointing
//      invitation (the exact next steps) when nothing is detected yet.
//   2. Engines — declared (the "Declared engine endpoints" setting,
//      engines_json: bring your own engine by pointing at its HTTP API) and
//      detected (live probes of every target's HTTP API).
//   3. Models — curated rows reusing the model_forge_browse fit data.
//   4. Installed model storage — the model_forge_storage list shape (names,
//      sizes, modification times). Read-only: deletion is a tool verb that
//      rides the human-approval gate, so the pane names the delete verb
//      honestly instead of offering an in-pane delete.
//
// Data doors (documented desktop SDK only):
//   - plugins.manage list  — the plugin's settings (targets_json, engines_json)
//   - shell.exec           — the tool-registered targets file in the plugin
//                            data dir (the backend target store)
//   - slash.exec           — the read-only /model-forge cockpit command (the
//                            browse fit data + the model_forge_storage list);
//                            needs an open chat session, degrades honestly
//   - fetch()              — live engine probes against each target's own
//                            HTTP API (no generation: /props, /version,
//                            /v1/models and friends)
// Target state is the union of the tool store and the settings field (the
// settings win on id conflicts, mirroring the backend's targets.load_targets).
// A blocked fetch degrades to a hint row; a source read failure is surfaced,
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

// The pointing invitation, pane-side copy (the tools return the same next
// steps; when the /model-forge bridge answers, its invitation wins).
const PANE_INVITATION = {
  reason: "Nothing is detected yet: no compute targets are registered and no inference engine answered the pane's probes.",
  next_steps: [
    "Add a cloud GPU or another machine as a compute target with model_forge_target_add (id, base_url, and vram_gb / ram_gb for fit estimates) — the human-approval gate confirms each target before it is used.",
    "Or declare an engine endpoint in Settings -> Plugins -> Model Forge ('Declared engine endpoints', the engines_json field): each entry is a label plus the engine's HTTP API base URL, and Model Forge probes it everywhere engines are detected.",
  ],
};

// Set by the pane on mount so the palette command can force a reload.
let refreshHook = null;

function rootUrl(base) {
  return String(base || "").replace(/\/+$/, "").replace(/\/v1$/, "");
}

function fmtBytes(n) {
  const bytes = Number(n) || 0;
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + " GB";
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
  return bytes + " B";
}

// --- settings (plugins.manage) -----------------------------------------------

async function readSettings() {
  // The plugin's settings through the documented read recipe: current values
  // ride each settings_schema field's `value` (v6 rows), older shims expose an
  // entries dict. Read failures are surfaced, never silently defaulted away.
  try {
    const res = await host.request("plugins.manage", { action: "list" });
    let rawTargets = null;
    let rawEngines = null;
    if (res && Array.isArray(res.plugins)) {
      const entry = res.plugins.find((r) => r && (r.key === "model-forge" || r.name === "model-forge")) || {};
      const fields = entry.settings_schema || [];
      rawTargets = (fields.find((f) => f && f.key === "targets_json") || {}).value;
      rawEngines = (fields.find((f) => f && f.key === "engines_json") || {}).value;
    } else {
      const entries = (res && res.entries) || {};
      const settings = (entries["model-forge"] || {}).settings || {};
      rawTargets = settings.targets_json;
      rawEngines = settings.engines_json;
    }
    const targets = (Array.isArray(rawTargets) ? rawTargets : [])
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
    const declared = (Array.isArray(rawEngines) ? rawEngines : [])
      .filter((e) => e && /^https?:\/\//.test(String(e.base_url || "")))
      .map((e) => ({
        label: String(e.label || e.id || e.base_url),
        base_url: String(e.base_url).replace(/\/+$/, ""),
        source: "user-declared",
      }));
    return { targets, declared, error: "" };
  } catch (err) {
    return { targets: [], declared: [], error: "settings unreadable: " + ((err && err.message) || err) };
  }
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

async function readTargets(settingsRows) {
  const [tool, settings] = await Promise.all([
    readDataDirTargets(),
    settingsRows ? Promise.resolve({ targets: settingsRows, error: "" }) : readSettings(),
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

// --- the /model-forge cockpit bridge (browse fit data + storage list) ---------

async function readCockpitBridge() {
  // The plugin's read-only /model-forge cockpit command, run through the
  // documented slash.exec door. It returns the browse fit rows and the
  // model_forge_storage list shape. Needs an active chat session; without
  // one (or on any failure) the pane degrades to an honest hint.
  let sid = null;
  try {
    sid = host.state.activeSessionId.get();
  } catch (err) {
    sid = null;
  }
  if (!sid) {
    return { data: null, hint: "Model and storage data need an open chat session (the /model-forge cockpit bridge); open a chat and refresh. Or ask for model_forge_browse / model_forge_storage." };
  }
  try {
    const res = await host.request("slash.exec", { session_id: sid, command: "/model-forge cockpit" });
    if (res && res.error) {
      return { data: null, hint: "the /model-forge cockpit bridge answered with an error: " + (res.error.message || res.error) };
    }
    const raw = String((res && res.output) || "").trim();
    if (!raw) return { data: null, hint: "the /model-forge cockpit bridge returned no output" };
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") {
      return { data: null, hint: "the /model-forge cockpit bridge returned an unexpected payload" };
    }
    return { data, hint: "" };
  } catch (err) {
    return { data: null, hint: "the /model-forge cockpit bridge is unavailable: " + ((err && err.message) || err) };
  }
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
  return { target, roots, engines };
}

async function probeDeclared(declared) {
  // A declared engine endpoint is probed exactly like a local engine root —
  // the bring-your-own-engine path, live.
  return Promise.all(declared.map(async (row) => {
    const engines = await probeRoot(rootUrl(row.base_url));
    return { row, engines };
  }));
}

// --- UI -----------------------------------------------------------------------

function SectionTitle(props) {
  return jsx("div", {
    style: {
      margin: "14px 0 4px", fontWeight: 600, fontSize: "12px",
      textTransform: "uppercase", letterSpacing: "0.04em",
      color: "var(--ui-text-secondary)",
    },
    children: props.children,
  });
}

function Row(props) {
  return jsx("div", {
    style: {
      display: "flex", alignItems: "baseline", gap: "8px",
      padding: "4px 0", borderBottom: "1px solid var(--ui-stroke-secondary)",
    },
    children: props.children,
  });
}

function Hint(props) {
  return jsx("div", {
    style: {
      color: "var(--ui-text-quaternary)", fontSize: "11px",
      padding: "3px 0", lineHeight: "1.5",
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

function Invitation(props) {
  const inv = props.invitation;
  return jsx("div", {
    style: {
      margin: "6px 0", padding: "8px",
      border: "1px solid var(--ui-stroke-secondary)", borderRadius: "6px",
    },
    children: [
      jsx("div", {
        key: "r",
        style: { color: "var(--ui-text-secondary)", fontSize: "12px",
                 fontWeight: 600, marginBottom: "4px" },
        children: "Point Model Forge at your compute",
      }),
      jsx("div", {
        key: "why",
        style: { color: "var(--ui-text-secondary)", fontSize: "12px",
                 marginBottom: "4px" },
        children: inv.reason,
      }),
      ...(inv.next_steps || []).map((step, i) =>
        jsx("div", {
          key: "s" + i,
          style: { color: "var(--ui-text-secondary)", fontSize: "12px",
                   padding: "2px 0 2px 10px", lineHeight: "1.5" },
          children: (i + 1) + ". " + step,
        })),
    ],
  });
}

function TargetSection(props) {
  const { probed, bridge } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "GPU & compute targets" })];
  // GPU hardware comes from the /model-forge cockpit bridge (the backend
  // hardware probe); without it the pane says so instead of guessing.
  const local = (bridge && bridge.targets || []).find((t) => t.id === "local");
  const hw = local && local.hardware;
  if (hw) {
    const gpus = (hw.gpus || []).map((g) => (g.name || "GPU") + (g.vram_gb ? " (" + g.vram_gb + " GB)" : "")).join(", ");
    children.push(jsx("div", {
      key: "gpu",
      style: { color: "var(--ui-text-secondary)", fontSize: "12px", padding: "2px 0" },
      children: "This machine: " +
        (gpus || "no GPU detected") +
        (hw.total_vram_gb ? " · " + hw.total_vram_gb + " GB VRAM" : "") +
        (hw.ram_gb ? " · " + hw.ram_gb + " GB RAM" : "") +
        (hw.platform ? " · " + hw.platform : ""),
    }));
  } else {
    children.push(jsx(Hint, {
      key: "gpu",
      children: "GPU hardware for this machine comes from the model_forge_status probe (the pane bridge needs an open chat session).",
    }));
  }
  probed.forEach((r) => {
    const t = r.target;
    children.push(jsx("div", {
      key: "h-" + t.id,
      style: { margin: "8px 0 2px", fontWeight: 600 },
      children: (t.label || t.id) +
        " · " + (t.kind || "") +
        (t.vram_gb ? " · " + t.vram_gb + " GB VRAM" : "") +
        (t.ram_gb ? " · " + t.ram_gb + " GB RAM" : "") +
        (t.source ? " · " + t.source : ""),
    }));
    children.push(jsx("div", {
      key: "u-" + t.id,
      style: { color: "var(--ui-text-quaternary)", fontSize: "11px" },
      children: t.base_url,
    }));
  });
  return children;
}

function EnginesSection(props) {
  const { probed, declared } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "Engines" })];
  children.push(jsx("div", {
    key: "dl",
    style: { color: "var(--ui-text-quaternary)", fontSize: "11px", padding: "2px 0" },
    children: "Declared engine endpoints (user-declared — Settings -> Plugins -> Model Forge):",
  }));
  if (!declared.length) {
    children.push(jsx(Hint, {
      key: "dn",
      children: "None declared. Point Model Forge at any engine HTTP API here (label + base_url) — the bring-your-own-engine path.",
    }));
  }
  declared.forEach((d, i) => {
    children.push(jsx(Row, {
      key: "d-" + i,
      children: [
        jsx("span", {
          key: "n",
          style: { color: "var(--ui-accent)", minWidth: "130px" },
          children: d.label,
        }),
        jsx("span", {
          key: "u",
          style: { color: "var(--ui-text-secondary)", fontSize: "12px" },
          children: d.base_url,
        }),
        jsx("span", {
          key: "s",
          style: { color: "var(--ui-text-quaternary)", fontSize: "11px" },
          children: d.source,
        }),
      ],
    }));
    (d.engines || []).forEach((e, j) => {
      children.push(jsx(EngineRow, { key: "de-" + i + "-" + j, engine: e }));
    });
    if (!(d.engines || []).length) {
      children.push(jsx(Hint, {
        key: "dx-" + i,
        children: "no engine answered at " + d.base_url + " (or the app blocked the probe)",
      }));
    }
  });
  children.push(jsx("div", {
    key: "dt",
    style: { color: "var(--ui-text-quaternary)", fontSize: "11px", padding: "6px 0 2px" },
    children: "Detected on compute targets:",
  }));
  let any = false;
  probed.forEach((r) => {
    const okRows = r.engines.filter((e) => e.ok);
    okRows.forEach((e, i) => { any = true; children.push(jsx(EngineRow, { key: "e-" + r.target.id + "-" + i, engine: e })); });
    if (!okRows.length) {
      const shown = r.roots.slice(0, 4).join(", ") + (r.roots.length > 4 ? ", …" : "");
      children.push(jsx(EngineRow, {
        key: "e-" + r.target.id + "-none",
        engine: {
          name: "none on " + r.target.id, ok: false,
          detail: "no engine answered at " + shown + " (or the app blocked the probe)",
        },
      }));
    }
  });
  return { children, anyDetected: any };
}

function ModelsSection(props) {
  const { bridge, bridgeHint } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "Models (browse fit data)" })];
  const rows = (bridge && bridge.models) || [];
  if (!rows.length) {
    children.push(jsx(Hint, {
      key: "h",
      children: "Model fit rows come from model_forge_browse (curated catalog). " +
        (bridgeHint || "Ask for model_forge_browse, or open a chat session so this pane can load them."),
    }));
    return children;
  }
  children.push(jsx(Hint, {
    key: "n",
    children: "The model_forge_browse fit data (advisory; engines do authoritative memory sizing at launch).",
  }));
  rows.forEach((m, i) => {
    const fit = m.fit || {};
    children.push(jsx(Row, {
      key: "m-" + i,
      children: [
        jsx("span", {
          key: "n",
          style: { minWidth: "150px", color: "var(--ui-text-secondary)" },
          children: m.name,
        }),
        jsx("span", {
          key: "mod",
          style: { color: "var(--ui-text-quaternary)", fontSize: "11px", minWidth: "70px" },
          children: m.modality_label || m.modality,
        }),
        jsx("span", {
          key: "f",
          style: { color: "var(--ui-accent)", fontSize: "12px", minWidth: "110px" },
          children: fit.verdict || m.status || "",
        }),
        jsx("span", {
          key: "d",
          style: { color: "var(--ui-text-secondary)", fontSize: "12px" },
          children: ((m.engines || []).join(", ") || "") +
            (m.size_gb ? " · ~" + m.size_gb + " GB" : "") +
            (fit.advisory ? " — " + fit.advisory : ""),
        }),
      ],
    }));
  });
  return children;
}

function StorageSection(props) {
  const { bridge, bridgeHint } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "Installed model storage" })];
  const storage = bridge && bridge.storage;
  const entries = (storage && storage.entries) || [];
  if (!entries.length) {
    children.push(jsx(Hint, {
      key: "h",
      children: "The installed-models list (names, sizes, modification times) comes from model_forge_storage. " +
        (bridgeHint || "Ask for model_forge_storage, or open a chat session so this pane can load it."),
    }));
    return children;
  }
  children.push(jsx(Hint, {
    key: "n",
    children: storage.note || "Sizes are on-disk sizes; modification times are filesystem times.",
  }));
  entries.forEach((e, i) => {
    children.push(jsx(Row, {
      key: "s-" + i,
      children: [
        jsx("span", {
          key: "n",
          style: { minWidth: "150px", color: "var(--ui-text-secondary)" },
          children: e.name,
        }),
        jsx("span", {
          key: "z",
          style: { color: "var(--ui-text-secondary)", fontSize: "12px", minWidth: "70px" },
          children: fmtBytes(e.size_bytes),
        }),
        jsx("span", {
          key: "m",
          style: { color: "var(--ui-text-quaternary)", fontSize: "11px", minWidth: "150px" },
          children: e.last_modified || "",
        }),
        jsx("span", {
          key: "r",
          style: { color: "var(--ui-text-quaternary)", fontSize: "11px" },
          children: (e.root_label || e.root || "") + (e.type ? " · " + e.type : ""),
        }),
      ],
    }));
  });
  // Deletion cannot ride the human-approval gate through a pane data read,
  // so the pane renders this list read-only and names the delete verb.
  children.push(jsx(Hint, {
    key: "del",
    children: bridge.delete_note ||
      "Read-only listing. To delete an entry, run the model_forge_storage tool with action=delete and the entry id — the human-approval gate shows the exact path and size first. This pane never deletes files.",
  }));
  return children;
}

function Cockpit(props) {
  const ctx = props.ctx;
  const [state, setState] = useState(null);
  const load = () => {
    Promise.all([readSettings(), readCockpitBridge()]).then(([settings, bridge]) => {
      return readTargets(settings.targets).then(({ targets, errors: errs }) => {
        return Promise.all([
          Promise.all(targets.map(probeTarget)),
          probeDeclared(settings.declared),
        ]).then(([probed, declaredProbed]) => {
          const declared = settings.declared.map((row, i) => ({
            ...row,
            engines: (declaredProbed[i] && declaredProbed[i].engines) || [],
          }));
          setState({
            probed,
            declared,
            bridge: bridge.data,
            bridgeHint: bridge.hint,
            errors: [settings.error, ...errs].filter(Boolean),
          });
        });
      });
    }).catch((err) => {
      setState({
        probed: [], declared: [], bridge: null,
        bridgeHint: "", errors: ["cockpit load failed: " + ((err && err.message) || err)],
      });
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
  if (!state) {
    return jsx("div", {
      style: { color: "var(--ui-text-secondary)", padding: "8px" },
      children: "Loading the Model Forge cockpit…",
    });
  }
  const engineSection = EnginesSection(state);
  const anyEngine = engineSection.anyDetected;
  const anyTarget = state.probed.some((r) => r.target.id !== "local");
  const localHw = state.bridge && (state.bridge.targets || []).find((t) => t.id === "local");
  const anyGpu = Boolean(localHw && localHw.hardware && ((localHw.hardware.gpus || []).length || localHw.hardware.total_vram_gb > 0));
  const gpuKnown = Boolean(localHw);
  const nothingDetected = !anyTarget && !anyEngine && (!gpuKnown || !anyGpu);
  const invitation = (state.bridge && state.bridge.invitation) ||
    (nothingDetected ? PANE_INVITATION : null);

  const children = [];
  state.errors.forEach((msg, i) => {
    children.push(jsx("div", {
      key: "err-" + i,
      style: { color: "var(--ui-text-quaternary)", fontSize: "11px",
               padding: "2px 0" },
      children: msg,
    }));
  });
  children.push(...TargetSection(state));
  if (invitation) children.push(jsx(Invitation, { key: "inv", invitation }));
  children.push(...engineSection.children);
  children.push(...ModelsSection({ bridge: state.bridge, bridgeHint: state.bridgeHint }));
  children.push(...StorageSection({ bridge: state.bridge, bridgeHint: state.bridgeHint }));
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
