// Model Forge — unified native Hermes Desktop UI (bundled inside model-forge).
//
// Because this Desktop half ships inside the core `model-forge` package
// (id: "model-forge"), `ctx.rest(...)` calls `/api/plugins/model-forge/*`
// directly on the Hermes Python backend:
//   - ZERO browser CORS failures on Ollama, ComfyUI, vLLM, SGLang, Exo, LAN GPUs
//   - ZERO chat-session requirement (full GPU hardware, engines, catalog,
//     Hugging Face live search, job progress, storage, residency, and on-demand
//     resources work immediately on first launch)
//   - ZERO shell reads from the Desktop renderer
//   - ONE Plugin Store install gives users tools, Desktop UI, and native
//     image/video providers (including direct ComfyUI /prompt workflows)
//
// Policy & honesty:
//   - Uses documented @hermes/plugin-sdk components and gateway RPCs only.
//   - Resource view is quiet by design: collapsed by default, loads once on
//     expand, manual Refresh only, zero timers or polling.
//   - Destructive deletes, vendor engine installers, local engine launches,
//     and config-rewriting activations route to Hermes chat behind the
//     human-approval gate.

import {
  Badge,
  Button,
  Codicon,
  ConfirmDialog,
  CopyButton,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DisclosureCaret,
  EmptyState,
  Input,
  ListRow,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  SearchField,
  SegmentedControl,
  Separator,
  StatusDot,
  cn,
  host,
} from "@hermes/plugin-sdk";
import { jsx, jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";

const MODEL_FORGE_ROUTE = "/model-forge";
const VRAM_PRESETS = [8, 12, 16, 24, 32, 48, 80];

const MODALITY_OPTIONS = [
  { id: "all", label: "All" },
  { id: "chat", label: "Chat" },
  { id: "vision", label: "Vision" },
  { id: "ocr", label: "OCR" },
  { id: "image", label: "Images" },
  { id: "image_edit", label: "Image Edit" },
  { id: "video", label: "Video" },
  { id: "3d", label: "3D" },
  { id: "music", label: "Music" },
  { id: "sfx", label: "SFX" },
  { id: "voice_clone", label: "Voice" },
];

const ENGINE_GUIDE = [
  {
    id: "llamacpp",
    name: "Hermes Local (llama.cpp)",
    modalities: ["chat", "vision"],
    defaultPort: 8080,
    installMode: "core_managed",
    summary: "Built right into Hermes Desktop for GGUF chat and vision models.",
    quickStart: "Open Settings -> Providers -> Local Models to download and run GGUF models automatically.",
  },
  {
    id: "ollama",
    name: "Ollama",
    modalities: ["chat", "vision", "ocr"],
    defaultPort: 11434,
    installMode: "approved_installer",
    canStart: true,
    summary: "Easiest local runner. Pulls Ollama tags and Hugging Face GGUFs over HTTP with byte progress.",
    quickStart: "OLLAMA_HOST=127.0.0.1:11434 ollama serve",
  },
  {
    id: "comfyui",
    name: "ComfyUI",
    modalities: ["image", "image_edit", "video", "3d"],
    defaultPort: 8188,
    installMode: "connect_or_manual",
    summary: "Direct image, edit, and video workflows via native /prompt API — no separate facade daemon required.",
    quickStart: "python main.py --listen 127.0.0.1 --port 8188",
  },
  {
    id: "vllm",
    name: "vLLM / vLLM-Omni",
    modalities: ["chat", "vision", "ocr", "image", "image_edit"],
    defaultPort: 8000,
    installMode: "connect_or_manual",
    canStart: true,
    summary: "High-speed GPU server for chat, vision, OCR, and OpenAI-compatible Omni image generation.",
    quickStart: "vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000",
  },
  {
    id: "sglang",
    name: "SGLang / Diffusion",
    modalities: ["chat", "vision", "image", "image_edit"],
    defaultPort: 30000,
    installMode: "connect_or_manual",
    summary: "Fast GPU inference for chat, vision, and SGLang Diffusion image editing.",
    quickStart: "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 30000",
  },
  {
    id: "strata",
    name: "Strata",
    modalities: ["chat", "vision"],
    defaultPort: 8080,
    installMode: "approved_installer",
    summary: "Streams giant MoE chat models from RAM to consumer GPUs and Apple Silicon.",
    quickStart: "strata launch --help",
  },
  {
    id: "maya",
    name: "Project Maya",
    modalities: ["chat", "vision", "ocr"],
    defaultPort: 8080,
    installMode: "approved_installer",
    summary: "Multi-GPU MoE server with built-in catalog, HTTP pulls, and automatic model swapping.",
    quickStart: "./setup.sh",
  },
  {
    id: "exo",
    name: "Exo",
    modalities: ["chat", "vision", "ocr"],
    defaultPort: 52415,
    installMode: "connect_or_manual",
    canStart: true,
    summary: "Pools Macs and Linux GPU machines on your network into one distributed local cluster.",
    quickStart: "exo",
  },
];

const refreshHooks = new Set();
let openAddGpuHook = null;

function notifyRefreshHooks() {
  refreshHooks.forEach((fn) => {
    try { fn(); } catch (err) { /* isolated */ }
  });
}

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

function slugifyId(label) {
  return String(label || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32) || "gpu-target";
}

function fitBadgeMeta(verdict) {
  const map = {
    fits_gpu: { label: "Fits your GPU", variant: "success" },
    fits_memory: { label: "Fits memory", variant: "default" },
    engine_managed: { label: "Auto-sized", variant: "default" },
    engine_unavailable: { label: "Needs engine", variant: "warn" },
    needs_gpu: { label: "Needs GPU", variant: "warn" },
    needs_ram: { label: "Needs more RAM", variant: "warn" },
    too_big: { label: "Too large", variant: "destructive" },
    unavailable: { label: "Coming soon", variant: "muted" },
    unknown: { label: "Check size", variant: "outline" },
  };
  return map[verdict] || { label: verdict || "Advisory", variant: "outline" };
}

function activeChatSessionId() {
  try {
    if (host.state && host.state.focusedSessionId) {
      const f = host.state.focusedSessionId.get();
      if (f) return f;
    }
    if (host.state && host.state.activeSessionId) {
      return host.state.activeSessionId.get() || null;
    }
  } catch (err) {
    return null;
  }
  return null;
}

// --- Backend REST + Gateway RPC data doors (ZERO CORS, ZERO shell reads) -----

function parsePluginRow(row) {
  const fields = (row && row.settings_schema) || [];
  const byKey = {};
  fields.forEach((f) => {
    if (f && f.key) byKey[f.key] = f;
  });
  return { row, byKey };
}

async function readPluginSettings() {
  try {
    const [listRes, factsRes] = await Promise.all([
      host.request("plugins.manage", { action: "list" }),
      host.request("machine.facts", {}).catch(() => null),
    ]);
    const list = (listRes && Array.isArray(listRes.plugins)) ? listRes.plugins : [];
    const findPkg = (id) => list.find((r) => r && (r.key === id || r.name === id)) || null;
    const core = parsePluginRow(findPkg("model-forge")).byKey;
    const img = parsePluginRow(findPkg("model-forge-image")).byKey;
    const vid = parsePluginRow(findPkg("model-forge-video")).byKey;

    const rawTargets = core.targets_json ? core.targets_json.value : [];
    const rawEngines = core.engines_json ? core.engines_json.value : [];
    const targets = (Array.isArray(rawTargets) ? rawTargets : [])
      .filter((t) => t && t.id && t.base_url)
      .map((t) => ({
        id: String(t.id),
        label: String(t.label || t.id),
        base_url: String(t.base_url),
        kind: String(t.kind || "lan"),
        vram_gb: Number(t.vram_gb) || 0,
        ram_gb: Number(t.ram_gb) || 0,
        vram_source: String(t.vram_source || "declared"),
        ram_source: String(t.ram_source || "declared"),
      }));
    const declared = (Array.isArray(rawEngines) ? rawEngines : [])
      .filter((e) => e && /^https?:\/\//.test(String(e.base_url || "")))
      .map((e) => ({
        label: String(e.label || e.id || e.base_url),
        base_url: String(e.base_url).replace(/\/+$/, ""),
      }));
    return {
      targets,
      declared,
      credentials: {
        HF_TOKEN: Boolean(core.hf_token && core.hf_token.has_value),
        MODELSCOPE_API_TOKEN: Boolean(core.modelscope_token && core.modelscope_token.has_value),
      },
      mediaSettings: {
        imageBaseUrl: String((core.image_base_url && core.image_base_url.value) || (img.base_url && img.base_url.value) || ""),
        imageDefaultModel: String((core.image_default_model && core.image_default_model.value) || (img.default_model && img.default_model.value) || ""),
        videoBaseUrl: String((core.video_base_url && core.video_base_url.value) || (vid.base_url && vid.base_url.value) || ""),
        videoDefaultModel: String((core.video_default_model && core.video_default_model.value) || (vid.default_model && vid.default_model.value) || ""),
      },
      hasImageCompanion: Boolean(findPkg("model-forge-image")),
      hasVideoCompanion: Boolean(findPkg("model-forge-video")),
      hasProvidersCompanion: Boolean(findPkg("model-forge-providers")),
      machineFacts: factsRes || null,
      error: "",
    };
  } catch (err) {
    return {
      targets: [],
      declared: [],
      credentials: { HF_TOKEN: false, MODELSCOPE_API_TOKEN: false },
      mediaSettings: { imageBaseUrl: "", imageDefaultModel: "", videoBaseUrl: "", videoDefaultModel: "" },
      hasImageCompanion: false,
      hasVideoCompanion: false,
      hasProvidersCompanion: false,
      machineFacts: null,
      error: "settings unreadable: " + ((err && err.message) || err),
    };
  }
}

async function savePluginSettings(pluginKey, values) {
  const res = await host.request("plugins.manage", {
    action: "settings",
    key: pluginKey,
    values,
  });
  if (res && res.ok === false) {
    throw new Error(res.error || "Failed to save settings for " + pluginKey);
  }
  notifyRefreshHooks();
  return res;
}

async function saveHfToken(token) {
  const clean = String(token || "").trim();
  if (!clean) throw new Error("Paste your Hugging Face token first.");
  await host.request("model.save_key", { slug: "huggingface", api_key: clean });
  await host.request("plugins.manage", { action: "settings", key: "model-forge", values: {} }).catch(() => null);
  notifyRefreshHooks();
}

async function removeHfToken() {
  await host.request("model.disconnect", { slug: "huggingface" });
  await host.request("plugins.manage", { action: "settings", key: "model-forge", values: {} }).catch(() => null);
  notifyRefreshHooks();
}

async function loadUnifiedState(ctx) {
  const [settings, cockpit, jobsRes, clisRes] = await Promise.all([
    readPluginSettings(),
    ctx.rest("/cockpit").catch((err) => ({ ok: false, error: (err && err.message) || String(err) })),
    ctx.rest("/jobs").catch(() => ({ ok: false, jobs: [] })),
    ctx.rest("/engines/local-cli").catch(() => ({ ok: false, clis: {} })),
  ]);
  const errors = [];
  if (settings.error) errors.push(settings.error);
  if (cockpit && cockpit.ok === false && cockpit.error) errors.push(cockpit.error);
  return {
    settings,
    cockpit: cockpit && cockpit.ok !== false ? cockpit : null,
    jobs: (jobsRes && Array.isArray(jobsRes.jobs)) ? jobsRes.jobs : [],
    clis: (clisRes && clisRes.clis) || {},
    errors,
  };
}

// --- Shared visual primitives ------------------------------------------------

function Card(props) {
  return jsx("div", {
    className: cn(
      "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/40 p-3.5",
      props.className,
    ),
    children: props.children,
  });
}

function Hint(props) {
  return jsx("div", {
    className: "py-1 text-[11px] leading-relaxed text-(--ui-text-quaternary)",
    children: props.children,
  });
}

function MemoryBar(props) {
  const row = props.row;
  const label = String(row.kind || "");
  if (row.status !== "ok" || row.used_gb == null || row.total_gb == null) {
    return jsxs("div", {
      className: "flex gap-2 py-0.5 text-[11px] text-(--ui-text-quaternary)",
      children: [
        jsx("span", { className: "min-w-11", children: label }),
        jsx("span", { children: "unavailable — " + (row.detail || "no reading") }),
      ],
    });
  }
  const pct = Math.max(0, Math.min(100, Number(row.used_pct) || 0));
  return jsxs("div", {
    className: "flex items-center gap-2 py-0.5",
    children: [
      jsx("span", { className: "min-w-11 text-[11px] text-(--ui-text-secondary)", children: label }),
      jsx("div", {
        className: "h-2 w-28 overflow-hidden rounded border border-(--ui-stroke-secondary)",
        children: jsx("div", { style: { width: pct + "%", height: "100%", background: "var(--ui-accent)" } }),
      }),
      jsx("span", {
        className: "text-[11px] text-(--ui-text-secondary)",
        children: row.used_gb + "/" + row.total_gb + " GB (" + Math.round(pct) + "%)" +
          (row.booked_now_gb != null ? " · booked_now " + row.booked_now_gb + " GB" : "") +
          (row.free_gb != null ? " · free " + row.free_gb + " GB" : ""),
      }),
    ],
  });
}

function ResourceView(props) {
  const ctx = props.ctx;
  const [open, setOpen] = useState(false);
  const [state, setState] = useState({ phase: "idle", data: null, hint: "" });
  const load = () => {
    setState({ phase: "loading", data: null, hint: "" });
    ctx.rest("/resources")
      .then((res) => {
        if (res && res.ok !== false && !res.error) {
          setState({ phase: "ready", data: res, hint: "" });
        } else {
          setState({ phase: "error", data: null, hint: (res && res.error) || "no data" });
        }
      })
      .catch((err) => {
        setState({ phase: "error", data: null, hint: (err && err.message) || String(err) });
      });
  };
  const toggle = () => {
    if (!open && state.phase === "idle") load();
    setOpen(!open);
  };
  return jsxs("div", {
    className: "pt-2",
    children: [
      jsxs("button", {
        type: "button",
        onClick: toggle,
        className: "flex items-center gap-1.5 py-1 text-xs font-semibold text-(--ui-text-secondary) hover:text-(--ui-text-primary)",
        children: [
          jsx(DisclosureCaret, { open }),
          jsx("span", { children: "Resource view (on-demand)" }),
        ],
      }),
      !open ? jsx(Hint, {
        children: "Memory use (VRAM/RAM) and resident model blocks — loaded once when you open it, never a persistent display.",
      }) : null,
      open && state.phase === "loading" ? jsx(Hint, { children: "Loading the resource snapshot…" }) : null,
      open && state.phase === "error" ? jsx(Hint, {
        children: "Resource snapshot unavailable: " + (state.hint || "no data") + " Nothing was changed — Refresh to retry.",
      }) : null,
      open && state.data ? jsxs("div", {
        className: "grid gap-1.5 pt-1",
        children: [
          ...(state.data.targets || []).map((t) => jsxs("div", {
            key: t.target,
            className: "grid gap-0.5",
            children: [
              jsx("div", { className: "text-[11px] font-semibold text-(--ui-text-primary)", children: (t.label || t.target) + " · " + (t.kind || "") }),
              ...(t.memory || []).map((row, i) => jsx(MemoryBar, { key: "m-" + i, row })),
              jsx(Hint, { children: "resident (" + (t.residents || []).length + ")" }),
              ...(t.residents || []).map((r, i) => jsx("div", {
                key: "r-" + i,
                className: "pl-2.5 text-[11px] text-(--ui-text-secondary)",
                children: r.model_id + " · " + (r.size_gb != null ? r.size_gb + " GB (" + r.size_label + ")" : (r.size_label || "size unknown")) + " · engine=" + r.engine + " · lane=" + r.lane,
              })),
            ],
          })),
          state.data.note ? jsx(Hint, { children: state.data.note }) : null,
        ],
      }) : null,
      open ? jsx("button", {
        type: "button",
        onClick: load,
        className: "pt-1 text-[11px] text-(--ui-accent) hover:underline",
        children: "Refresh (manual — nothing polls)",
      }) : null,
    ],
  });
}

// --- Guided Dialogs (Zero CORS, Zero JSON) -----------------------------------

function HfCredentialsDialog(props) {
  const { open, onOpenChange, connected, onSaved } = props;
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const handleSave = async () => {
    setError("");
    setBusy(true);
    try {
      await saveHfToken(token);
      setToken("");
      host.notify({ kind: "success", message: "Hugging Face token saved to Hermes credentials." });
      if (onSaved) onSaved();
      onOpenChange(false);
    } catch (err) {
      setError((err && err.message) || String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnect = async () => {
    setError("");
    setBusy(true);
    try {
      await removeHfToken();
      host.notify({ kind: "info", message: "Hugging Face token removed." });
      if (onSaved) onSaved();
      onOpenChange(false);
    } catch (err) {
      setError((err && err.message) || String(err));
    } finally {
      setBusy(false);
    }
  };

  return jsx(Dialog, {
    open,
    onOpenChange,
    children: jsxs(DialogContent, {
      className: "max-w-md",
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: "Hugging Face Credentials" }),
            jsx(DialogDescription, {
              children: "Unlocks gated models (like LTX-2.5 or FLUX) and full-speed downloads. Stored in Hermes' secret store under HF_TOKEN and sent only to huggingface.co.",
            }),
          ],
        }),
        jsxs("div", {
          className: "grid gap-3 py-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between text-xs",
              children: [
                jsx("span", { className: "text-(--ui-text-secondary)", children: "Status" }),
                jsx(Badge, {
                  variant: connected ? "success" : "muted",
                  children: connected ? "Token saved (value hidden)" : "Not set (public models still work)",
                }),
              ],
            }),
            jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: connected ? "Replace Hugging Face token" : "Hugging Face access token (Read)" }),
                jsx(Input, { type: "password", value: token, onChange: (e) => setToken(e.target.value), placeholder: "hf_...", disabled: busy }),
                jsx("span", { className: "text-[11px] text-(--ui-text-quaternary)", children: "Create a free Read token at huggingface.co/settings/tokens." }),
              ],
            }),
            error ? jsx("div", { className: "text-xs text-destructive", children: error }) : null,
          ],
        }),
        jsxs(DialogFooter, {
          className: "flex flex-wrap items-center justify-between gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                connected ? jsx(Button, { size: "sm", variant: "destructive", disabled: busy, onClick: handleDisconnect, children: "Remove Token" }) : null,
                jsx(Button, {
                  size: "sm",
                  variant: "ghost",
                  onClick: () => { onOpenChange(false); host.navigate("/settings?tab=plugins&agent=model-forge"); },
                  children: "ModelScope & Advanced…",
                }),
              ],
            }),
            jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                jsx(Button, { size: "sm", variant: "outline", onClick: () => onOpenChange(false), children: "Cancel" }),
                jsx(Button, { size: "sm", disabled: busy || !token.trim(), onClick: handleSave, children: busy ? "Saving…" : "Save Token" }),
              ],
            }),
          ],
        }),
      ],
    }),
  });
}

function AddComputeDialog(props) {
  const { ctx, open, onOpenChange, initialTarget, existingTargets, existingEngines, defaultMode, onSaved } = props;
  const isEditingExisting = Boolean(initialTarget && initialTarget.id);
  const [mode, setMode] = useState(defaultMode || (initialTarget ? initialTarget.kind : "lan"));
  const [label, setLabel] = useState((initialTarget && initialTarget.label) || "");
  const [baseUrl, setBaseUrl] = useState((initialTarget && initialTarget.base_url) || "");
  const [vramGb, setVramGb] = useState(initialTarget && initialTarget.vram_gb ? String(initialTarget.vram_gb) : "");
  const [ramGb, setRamGb] = useState(initialTarget && initialTarget.ram_gb ? String(initialTarget.ram_gb) : "");
  const [vramSource, setVramSource] = useState((initialTarget && initialTarget.vram_source) || "declared");
  const [testing, setTesting] = useState(false);
  const [probeResult, setProbeResult] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (open) {
      setMode(defaultMode || (initialTarget ? initialTarget.kind : "lan"));
      setLabel((initialTarget && initialTarget.label) || "");
      setBaseUrl((initialTarget && initialTarget.base_url) || "");
      setVramGb(initialTarget && initialTarget.vram_gb ? String(initialTarget.vram_gb) : "");
      setRamGb(initialTarget && initialTarget.ram_gb ? String(initialTarget.ram_gb) : "");
      setVramSource((initialTarget && initialTarget.vram_source) || "declared");
      setProbeResult(null);
      setError("");
    }
  }, [open, initialTarget, defaultMode]);

  const handleTest = async () => {
    setError("");
    const url = rootUrl(baseUrl.trim());
    if (!/^https?:\/\//.test(url)) {
      setError("Enter a full http:// or https:// address, e.g. http://gpu-box.example.com:11434");
      return;
    }
    setTesting(true);
    try {
      // Backend Python probe via ctx.rest — ZERO browser CORS failures!
      const res = await ctx.rest("/probe", { method: "POST", body: { base_url: url } });
      if (!res || res.ok === false) {
        setError((res && res.error) || "Probe failed.");
        return;
      }
      const sug = res.budget_suggestion || {};
      if (sug.suggested) {
        if (sug.vram_gb && !vramGb) {
          setVramGb(String(sug.vram_gb));
          setVramSource("suggested");
        }
        if (sug.ram_gb && !ramGb) {
          setRamGb(String(sug.ram_gb));
        }
      }
      setProbeResult({ url: res.base_url || url, engines: res.engines || [] });
    } catch (err) {
      setError("Probe failed: " + ((err && err.message) || err));
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    setError("");
    const url = baseUrl.trim().replace(/\/+$/, "");
    if (!/^https?:\/\//.test(url)) {
      setError("Enter a full http:// or https:// URL first.");
      return;
    }
    const cleanLabel = label.trim() || (mode === "cloud" ? "Cloud GPU" : mode === "engine" ? "Custom Engine" : "Network GPU");
    setSaving(true);
    try {
      if (mode === "engine") {
        const current = Array.isArray(existingEngines) ? existingEngines : [];
        const next = [
          ...current.filter((e) => rootUrl(e.base_url) !== rootUrl(url)),
          { label: cleanLabel, base_url: url },
        ];
        await savePluginSettings("model-forge", { engines_json: next });
        host.notify({ kind: "success", message: "Connected engine endpoint: " + cleanLabel });
      } else {
        const current = Array.isArray(existingTargets) ? existingTargets : [];
        const id = (initialTarget && initialTarget.id && initialTarget.id !== "local")
          ? initialTarget.id
          : slugifyId(cleanLabel);
        const row = {
          id,
          label: cleanLabel,
          base_url: url,
          kind: mode === "cloud" ? "cloud" : "lan",
          vram_gb: Number(vramGb) || 0,
          ram_gb: Number(ramGb) || 0,
          vram_source: Number(vramGb) > 0 ? vramSource : "",
          ram_source: Number(ramGb) > 0 ? "declared" : "",
        };
        const next = [...current.filter((t) => t.id !== id), row];
        await savePluginSettings("model-forge", { targets_json: next });
        host.notify({ kind: "success", message: "Saved compute target: " + cleanLabel });
      }
      if (onSaved) onSaved();
      onOpenChange(false);
    } catch (err) {
      setError((err && err.message) || String(err));
    } finally {
      setSaving(false);
    }
  };

  return jsx(Dialog, {
    open,
    onOpenChange,
    children: jsxs(DialogContent, {
      className: "max-w-lg",
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, {
              children: isEditingExisting
                ? "Edit GPU Target"
                : mode === "engine"
                  ? "Connect Engine URL"
                  : "Point Model Forge at a GPU or Engine",
            }),
            jsx(DialogDescription, {
              children: "Connect another computer on your home/office network, a rented cloud GPU, or any running inference server. Tests directly from the Hermes backend (no CORS limits, no JSON).",
            }),
          ],
        }),
        jsxs("div", {
          className: "grid gap-3 py-2",
          children: [
            !isEditingExisting ? jsx(SegmentedControl, {
              value: mode,
              onChange: setMode,
              options: [
                { id: "lan", label: "Network GPU (LAN)" },
                { id: "cloud", label: "Cloud GPU" },
                { id: "engine", label: "Engine URL Only" },
              ],
            }) : null,
            jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: "Friendly name" }),
                jsx(Input, { value: label, onChange: (e) => setLabel(e.target.value), placeholder: mode === "cloud" ? "Cloud A100 Box" : mode === "engine" ? "Workstation Ollama" : "Gaming PC GPU" }),
              ],
            }),
            jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: "Server address (URL)" }),
                jsxs("div", {
                  className: "flex gap-2",
                  children: [
                    jsx(Input, { className: "flex-1", value: baseUrl, onChange: (e) => setBaseUrl(e.target.value), placeholder: mode === "cloud" ? "https://gpu.example.com:8080" : "http://gpu-box.example.com:11434" }),
                    jsx(Button, { size: "sm", variant: "outline", disabled: testing || !baseUrl.trim(), onClick: handleTest, children: testing ? "Testing…" : "Test Connection" }),
                  ],
                }),
              ],
            }),
            probeResult ? jsxs(Card, {
              className: "p-2.5 text-xs",
              children: [
                jsx("div", {
                  className: "mb-1 font-medium text-(--ui-text-primary)",
                  children: probeResult.engines.length
                    ? "Connected! Detected " + probeResult.engines.length + " engine(s):"
                    : "No engine answered at " + probeResult.url + " (you can still save it and start the engine later).",
                }),
                ...probeResult.engines.map((e, i) => jsxs("div", {
                  key: "pe-" + i,
                  className: "flex items-center gap-2 py-0.5 text-(--ui-text-secondary)",
                  children: [
                    jsx(StatusDot, { tone: "good" }),
                    jsx("span", { className: "font-medium", children: e.name || e.id }),
                    jsx("span", { className: "text-(--ui-text-quaternary)", children: e.detail }),
                  ],
                })),
              ],
            }) : null,
            mode !== "engine" ? jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsxs("div", {
                  className: "flex items-center justify-between",
                  children: [
                    jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: "GPU VRAM (GB) — used for model fit badges" }),
                    vramSource === "suggested" ? jsx(Badge, { variant: "success", size: "xs", children: "Auto-detected from engine" }) : null,
                  ],
                }),
                jsx("div", {
                  className: "flex flex-wrap gap-1.5",
                  children: VRAM_PRESETS.map((gb) => jsx(Button, {
                    key: "v" + gb,
                    size: "xs",
                    variant: Number(vramGb) === gb ? "default" : "chip",
                    onClick: () => { setVramGb(String(gb)); setVramSource("declared"); },
                    children: gb + " GB",
                  })),
                }),
                jsxs("div", {
                  className: "grid grid-cols-2 gap-2 pt-1",
                  children: [
                    jsx(Input, { type: "number", value: vramGb, onChange: (e) => { setVramGb(e.target.value); setVramSource("declared"); }, placeholder: "Custom VRAM (GB)" }),
                    jsx(Input, { type: "number", value: ramGb, onChange: (e) => setRamGb(e.target.value), placeholder: "System RAM (GB, optional)" }),
                  ],
                }),
              ],
            }) : null,
            error ? jsx("div", { className: "text-xs text-destructive", children: error }) : null,
          ],
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, { size: "sm", variant: "outline", onClick: () => onOpenChange(false), children: "Cancel" }),
            jsx(Button, { size: "sm", disabled: saving || !baseUrl.trim(), onClick: handleSave, children: saving ? "Saving…" : "Save" }),
          ],
        }),
      ],
    }),
  });
}

function ActionPromptDialog(props) {
  const { action, onClose } = props;
  if (!action) return null;
  return jsx(Dialog, {
    open: Boolean(action),
    onOpenChange: (v) => { if (!v) onClose(); },
    children: jsxs(DialogContent, {
      className: "max-w-lg",
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: action.title || "Run in Hermes Chat" }),
            jsx(DialogDescription, {
              children: "This action runs through Hermes' human-approval gate so you can inspect the exact change, approve it, and watch live progress.",
            }),
          ],
        }),
        jsxs("div", {
          className: "grid gap-2 py-2",
          children: [
            jsx("div", {
              className: "rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-2.5 font-mono text-xs text-(--ui-text-primary) break-words",
              children: action.prompt,
            }),
          ],
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(CopyButton, { text: action.prompt, label: "Copy Prompt", showLabel: true, buttonSize: "sm", buttonVariant: "outline" }),
            jsx(Button, {
              size: "sm",
              onClick: () => {
                onClose();
                if (typeof host.newChat === "function") host.newChat();
                else host.navigate("/");
              },
              children: "Open Chat",
            }),
          ],
        }),
      ],
    }),
  });
}

// --- Full Unified Native Workspace Page (/model-forge) -----------------------

function ModelForgeWorkspace(props) {
  const ctx = props.ctx;
  const [state, setState] = useState(null);
  const [showFullCockpit, setShowFullCockpit] = useState(false);
  const [tab, setTab] = useState("models");
  const [selectedTarget, setSelectedTarget] = useState("local");
  const [modality, setModality] = useState("all");
  const [search, setSearch] = useState("");
  const [liveBrowse, setLiveBrowse] = useState({ phase: "idle", models: null, fresh: [], query: "", error: "" });
  const [customModelId, setCustomModelId] = useState("");
  const [customEngine, setCustomEngine] = useState("auto");
  const [hfOpen, setHfOpen] = useState(false);
  const [addGpuOpen, setAddGpuOpen] = useState(false);
  const [addGpuMode, setAddGpuMode] = useState("lan");
  const [editingTarget, setEditingTarget] = useState(null);
  const [removeConfirm, setRemoveConfirm] = useState(null);
  const [pendingAction, setPendingAction] = useState(null);
  const [engineCmdDialog, setEngineCmdDialog] = useState(null);
  const [imgUrlDraft, setImgUrlDraft] = useState("");
  const [imgModelDraft, setImgModelDraft] = useState("");
  const [vidUrlDraft, setVidUrlDraft] = useState("");
  const [vidModelDraft, setVidModelDraft] = useState("");
  const [savingMedia, setSavingMedia] = useState(false);

  const load = () => {
    loadUnifiedState(ctx).then((next) => {
      setState(next);
      setImgUrlDraft(next.settings.mediaSettings.imageBaseUrl || "");
      setImgModelDraft(next.settings.mediaSettings.imageDefaultModel || "");
      setVidUrlDraft(next.settings.mediaSettings.videoBaseUrl || "");
      setVidModelDraft(next.settings.mediaSettings.videoDefaultModel || "");
    });
  };

  useEffect(() => {
    load();
    refreshHooks.add(load);
    openAddGpuHook = () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); };
    const disposer = ctx.onEvent("plugin.model-forge.settings.changed", load);
    return () => {
      refreshHooks.delete(load);
      openAddGpuHook = null;
      disposer();
    };
  }, []);

  const runLiveBrowse = async (opts) => {
    const tId = (opts && opts.target) || selectedTarget || "local";
    const mod = (opts && opts.modality) || modality || "all";
    const q = (opts && opts.query != null) ? opts.query : search;
    const fresh = (opts && opts.fresh) ? "true" : "false";
    setLiveBrowse({ phase: "loading", models: liveBrowse.models, fresh: [], query: q, error: "" });
    const qs = new URLSearchParams();
    qs.set("target", tId);
    if (mod && mod !== "all") qs.set("modality", mod);
    if (q && q.trim()) qs.set("query", q.trim());
    qs.set("fresh", fresh);
    try {
      const res = await ctx.rest("/browse?" + qs.toString());
      if (!res || res.ok === false || res.error) {
        setLiveBrowse({ phase: "error", models: null, fresh: [], query: q, error: (res && res.error) || "browse failed" });
        return;
      }
      setLiveBrowse({
        phase: "ready",
        models: Array.isArray(res.models) ? res.models : [],
        fresh: Array.isArray(res.fresh) ? res.fresh : [],
        query: q,
        error: "",
      });
    } catch (err) {
      setLiveBrowse({ phase: "error", models: null, fresh: [], query: q, error: (err && err.message) || String(err) });
    }
  };

  const triggerAgentAction = (title, prompt) => {
    const sid = activeChatSessionId();
    try {
      if (host.composer && typeof host.composer.submit === "function" && sid) {
        if (host.composer.submit(sid, prompt)) {
          host.notify({ kind: "info", message: title + " — sent to your active chat." });
          return;
        }
      }
    } catch (err) { /* fall through */ }
    if (ctx && ctx.os && typeof ctx.os.writeClipboard === "function") {
      ctx.os.writeClipboard(prompt).catch(() => null);
    }
    setPendingAction({ title, prompt });
  };

  const cockpitTargets = (state && state.cockpit && Array.isArray(state.cockpit.targets)) ? state.cockpit.targets : [];

  const runningEngines = useMemo(() => {
    const rows = [];
    cockpitTargets.forEach((t) => {
      (t.engines || []).filter((e) => e.available).forEach((e) => {
        rows.push({
          ...e,
          targetId: t.id,
          targetLabel: t.label || t.id,
          root: e.endpoint || t.base_url,
        });
      });
    });
    return rows;
  }, [state]);

  const modelRows = useMemo(() => {
    const raw = (liveBrowse.phase === "ready" && Array.isArray(liveBrowse.models))
      ? liveBrowse.models
      : ((state && state.cockpit && Array.isArray(state.cockpit.models)) ? state.cockpit.models : []);
    const q = search.trim().toLowerCase();
    return raw.filter((m) => {
      if (modality !== "all" && m.modality !== modality) return false;
      if (!q) return true;
      const blob = [m.id, m.name, m.modality, m.modality_label, m.note, ...(m.engines || [])].join(" ").toLowerCase();
      return blob.includes(q);
    });
  }, [state, liveBrowse, modality, search]);

  if (!state) {
    return jsx("div", {
      className: "flex h-full items-center justify-center text-sm text-(--ui-text-secondary)",
      children: "Detecting your GPU and local inference engines…",
    });
  }

  const localTarget = cockpitTargets.find((t) => t.id === "local");
  const localHw = localTarget && localTarget.hardware;
  const localGpuSummary = localHw
    ? ((localHw.gpus || []).map((g) => (g.name || "GPU") + (g.vram_gb ? " (" + g.vram_gb + " GB)" : "")).join(", ") || "No local GPU detected")
    : (state.settings.machineFacts && state.settings.machineFacts.machine && state.settings.machineFacts.machine.gpu_class
      ? String(state.settings.machineFacts.machine.gpu_class).toUpperCase() + " GPU · " + (state.settings.machineFacts.machine.ram_gb || "?") + " GB RAM"
      : "No local GPU detected");
  const remoteTargets = cockpitTargets.filter((t) => t.id !== "local");
  const hfSet = Boolean((state.cockpit && state.cockpit.pull_credentials && state.cockpit.pull_credentials.HF_TOKEN) || state.settings.credentials.HF_TOKEN);
  const activeWiring = (state.cockpit && state.cockpit.active_wiring) || null;
  const runningEngineIds = new Set(runningEngines.map((e) => e.id).filter(Boolean));
  const targetOptions = cockpitTargets.map((t) => ({ id: t.id, label: t.label || t.id }));
  const ollamaCli = state.clis && state.clis.ollama;

  // Foolproof front door: when zero engines are active AND zero remote targets
  // are saved yet, show a calm 3-step quickstart hero (with 1-click escape to
  // the full cockpit).
  const needsFirstRun = runningEngines.length === 0 && remoteTargets.length === 0 && !showFullCockpit;

  const handleConfirmRemove = async () => {
    if (!removeConfirm) return;
    if (removeConfirm.kind === "target") {
      const next = (state.settings.targets || []).filter((t) => t.id !== removeConfirm.id);
      await savePluginSettings("model-forge", { targets_json: next });
      host.notify({ kind: "info", message: "Removed target: " + removeConfirm.label });
    } else if (removeConfirm.kind === "engine") {
      const next = (state.settings.declared || [])
        .filter((d) => rootUrl(d.base_url) !== rootUrl(removeConfirm.base_url));
      await savePluginSettings("model-forge", { engines_json: next });
      host.notify({ kind: "info", message: "Removed engine endpoint: " + removeConfirm.label });
    }
    setRemoveConfirm(null);
    load();
  };

  const handleSaveMedia = async (which) => {
    setSavingMedia(true);
    try {
      if (which === "image") {
        await savePluginSettings("model-forge", {
          image_base_url: imgUrlDraft.trim(),
          image_default_model: imgModelDraft.trim(),
        });
        if (state.settings.hasImageCompanion) {
          await savePluginSettings("model-forge-image", {
            base_url: imgUrlDraft.trim(),
            default_model: imgModelDraft.trim(),
          }).catch(() => null);
        }
        host.notify({ kind: "success", message: "Saved image generation endpoint." });
      } else {
        await savePluginSettings("model-forge", {
          video_base_url: vidUrlDraft.trim(),
          video_default_model: vidModelDraft.trim(),
        });
        if (state.settings.hasVideoCompanion) {
          await savePluginSettings("model-forge-video", {
            base_url: vidUrlDraft.trim(),
            default_model: vidModelDraft.trim(),
          }).catch(() => null);
        }
        host.notify({ kind: "success", message: "Saved video generation endpoint." });
      }
      load();
    } catch (err) {
      host.notifyError(err, "Could not save media backend settings.");
    } finally {
      setSavingMedia(false);
    }
  };

  const renderQuickstartHero = () => jsxs("div", {
    className: "mx-auto flex h-full max-w-3xl flex-col justify-center gap-6 px-6 py-8",
    children: [
      jsxs("div", {
        className: "text-center",
        children: [
          jsx("div", {
            className: "mx-auto mb-3 flex size-12 items-center justify-center rounded-xl bg-(--ui-bg-tertiary)",
            children: jsx(Codicon, { name: "layers", size: "1.5rem" }),
          }),
          jsx("h1", { className: "text-xl font-semibold text-(--ui-text-primary)", children: "Run Open-Weight Models Natively in Hermes" }),
          jsx("p", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: "Detected on this machine: " + localGpuSummary }),
        ],
      }),
      jsxs("div", {
        className: "grid gap-3 md:grid-cols-3",
        children: [
          jsxs(Card, {
            className: "flex flex-col justify-between gap-3",
            children: [
              jsxs("div", {
                children: [
                  jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "1. Start a Local Engine" }),
                  jsx("div", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: ollamaCli && ollamaCli.installed ? "Ollama is already installed on this machine. Start it in one click." : "Install Ollama for easy chat/vision pulls, or open Hermes built-in Local Models." }),
                ],
              }),
              jsxs("div", {
                className: "flex flex-wrap gap-1.5",
                children: [
                  ollamaCli && ollamaCli.installed
                    ? jsx(Button, {
                        size: "sm",
                        onClick: () => triggerAgentAction("Start Ollama", "Call model_forge_install with action=\"engine_start\", engine=\"ollama\", and execute=true so I can approve launching Ollama locally."),
                        children: "Start Ollama",
                      })
                    : jsx(Button, {
                        size: "sm",
                        onClick: () => triggerAgentAction("Install Ollama", "Call model_forge_install with action=\"engine_install\", engine=\"ollama\", and execute=true so I can review and approve installing Ollama."),
                        children: "Install Ollama",
                      }),
                  jsx(Button, { size: "sm", variant: "outline", onClick: () => host.navigate("/settings?tab=providers"), children: "Hermes Local" }),
                ],
              }),
            ],
          }),
          jsxs(Card, {
            className: "flex flex-col justify-between gap-3",
            children: [
              jsxs("div", {
                children: [
                  jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "2. Point at Another GPU" }),
                  jsx("div", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: "Have a gaming PC, ComfyUI box, or cloud GPU? Paste its URL and test live — no JSON." }),
                ],
              }),
              jsx(Button, {
                size: "sm",
                variant: "outline",
                onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); },
                children: "Add Network / Cloud GPU",
              }),
            ],
          }),
          jsxs(Card, {
            className: "flex flex-col justify-between gap-3",
            children: [
              jsxs("div", {
                children: [
                  jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "3. Hugging Face Key" }),
                  jsx("div", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: hfSet ? "Connected! Gated models and full-speed downloads are unlocked." : "Optional. Add a free Read token for gated repos like LTX-2.5 or FLUX." }),
                ],
              }),
              jsx(Button, {
                size: "sm",
                variant: hfSet ? "outline" : "secondary",
                onClick: () => setHfOpen(true),
                children: hfSet ? "Manage HF Token" : "Add HF Token",
              }),
            ],
          }),
        ],
      }),
      jsxs("div", {
        className: "flex items-center justify-center gap-3",
        children: [
          jsx(Button, { size: "sm", variant: "ghost", onClick: load, children: "Re-scan Hardware & Engines" }),
          jsx(Button, { size: "sm", variant: "link", onClick: () => setShowFullCockpit(true), children: "Explore full cockpit →" }),
        ],
      }),
    ],
  });

  const renderJobsBanner = () => {
    const active = (state.jobs || []).filter((j) => j && (j.state === "running" || j.state === "starting"));
    if (!active.length) return null;
    return jsxs(Card, {
      className: "grid gap-2",
      children: [
        jsxs("div", {
          className: "flex items-center justify-between",
          children: [
            jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Active Downloads & Pulls (" + active.length + ")" }),
            jsx(Button, { size: "xs", variant: "ghost", onClick: load, children: "Refresh Progress" }),
          ],
        }),
        ...active.map((j) => {
          const pct = j.total_bytes > 0 ? Math.min(100, Math.round((j.done_bytes / j.total_bytes) * 100)) : 0;
          return jsxs("div", {
            key: j.job_id,
            className: "grid gap-1 text-xs",
            children: [
              jsxs("div", {
                className: "flex items-center justify-between text-(--ui-text-secondary)",
                children: [
                  jsx("span", { className: "font-medium", children: j.model || j.dest || j.job_id }),
                  jsx("span", { children: (j.status_text || j.state) + (j.total_bytes > 0 ? " · " + fmtBytes(j.done_bytes) + " / " + fmtBytes(j.total_bytes) + " (" + pct + "%)" : "") }),
                ],
              }),
              j.total_bytes > 0 ? jsx("div", {
                className: "h-1.5 w-full overflow-hidden rounded bg-(--ui-bg-tertiary)",
                children: jsx("div", { style: { width: pct + "%", height: "100%", background: "var(--ui-accent)" } }),
              }) : null,
            ],
          });
        }),
      ],
    });
  };

  const renderModelsTab = () => {
    const liveModels = [];
    const seenLive = new Set();
    runningEngines.forEach((eng) => {
      (eng.models_loaded || []).forEach((mid) => {
        const key = eng.targetId + "::" + eng.id + "::" + mid;
        if (mid && !seenLive.has(key)) {
          seenLive.add(key);
          liveModels.push({ modelId: mid, engineName: eng.name, targetId: eng.targetId, targetLabel: eng.targetLabel });
        }
      });
    });

    return jsxs("div", {
      className: "grid gap-4",
      children: [
        renderJobsBanner(),

        // Live models ready on active engines
        liveModels.length > 0 ? jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between",
              children: [
                jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Ready right now (" + liveModels.length + " loaded on your engines)" }),
                jsx(Badge, { variant: "success", size: "xs", children: "Live" }),
              ],
            }),
            jsx("div", {
              className: "grid gap-1.5 sm:grid-cols-2",
              children: liveModels.slice(0, 8).map((lm, i) => jsxs("div", {
                key: "lm-" + i,
                className: "flex items-center justify-between gap-2 rounded border border-(--ui-stroke-tertiary) px-2.5 py-1.5 text-xs",
                children: [
                  jsxs("div", {
                    className: "min-w-0",
                    children: [
                      jsx("div", { className: "truncate font-medium text-(--ui-text-primary)", children: lm.modelId }),
                      jsx("div", { className: "truncate text-[11px] text-(--ui-text-tertiary)", children: lm.engineName + " · " + lm.targetLabel }),
                    ],
                  }),
                  jsx(Button, {
                    size: "xs",
                    onClick: () => triggerAgentAction(
                      "Use " + lm.modelId + " in Hermes",
                      "Call model_forge_activate with model_id=" + JSON.stringify(lm.modelId) + ", target=" + JSON.stringify(lm.targetId) + ", and apply=true so I can approve wiring it natively into Hermes.",
                    ),
                    children: "Use in Hermes",
                  }),
                ],
              })),
            }),
          ],
        }) : null,

        // Pull ANY Hugging Face or Ollama model + inline live Hugging Face Search
        jsxs(Card, {
          className: "grid gap-2.5",
          children: [
            jsxs("div", {
              className: "flex flex-wrap items-center justify-between gap-2",
              children: [
                jsxs("div", {
                  children: [
                    jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Pull any model or search Hugging Face" }),
                    jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Paste any Hugging Face repo (e.g. unsloth/Qwen3-8B-GGUF) or Ollama tag (e.g. qwen3:8b), or search Hugging Face live right here." }),
                  ],
                }),
                jsx(Button, {
                  size: "xs",
                  variant: "outline",
                  disabled: liveBrowse.phase === "loading",
                  onClick: () => runLiveBrowse({ query: customModelId.trim() || search.trim(), fresh: true }),
                  children: liveBrowse.phase === "loading" ? "Searching…" : "Search Hugging Face",
                }),
              ],
            }),
            jsxs("div", {
              className: "flex flex-wrap items-center gap-2",
              children: [
                jsx(Input, {
                  className: "min-w-56 flex-1",
                  value: customModelId,
                  onChange: (e) => setCustomModelId(e.target.value),
                  placeholder: "Search keyword, owner/repo, or Ollama tag (e.g. qwen3:8b)",
                }),
                jsx(SegmentedControl, {
                  value: customEngine,
                  onChange: setCustomEngine,
                  sizing: "content",
                  options: [
                    { id: "auto", label: "Auto" },
                    { id: "ollama", label: "Ollama" },
                    { id: "vllm", label: "vLLM" },
                    { id: "comfyui", label: "ComfyUI" },
                    { id: "exo", label: "Exo" },
                  ],
                }),
                jsx(Button, {
                  size: "sm",
                  disabled: !customModelId.trim(),
                  onClick: () => {
                    const mid = customModelId.trim();
                    const engPart = customEngine !== "auto" ? " and engine=" + JSON.stringify(customEngine) : "";
                    triggerAgentAction(
                      "Pull " + mid,
                      "Call model_forge_install with model_id=" + JSON.stringify(mid) + ", target=" + JSON.stringify(selectedTarget) + engPart + " and report download progress.",
                    );
                  },
                  children: "Pull Model",
                }),
              ],
            }),
          ],
        }),

        // Live Hugging Face search results (inline, no chat round-trip!)
        liveBrowse.fresh && liveBrowse.fresh.length > 0 ? jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between",
              children: [
                jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Live Hugging Face Results" + (liveBrowse.query ? " for \"" + liveBrowse.query + "\"" : "") + " (" + liveBrowse.fresh.length + ")" }),
                jsx(Badge, { variant: "outline", size: "xs", children: "Unvetted live HF" }),
              ],
            }),
            jsx("div", {
              className: "grid gap-1.5 sm:grid-cols-2",
              children: liveBrowse.fresh.slice(0, 8).map((row) => jsxs("div", {
                key: row.id,
                className: "flex items-center justify-between gap-2 rounded border border-(--ui-stroke-tertiary) px-2.5 py-1.5 text-xs",
                children: [
                  jsxs("div", {
                    className: "min-w-0",
                    children: [
                      jsx("div", { className: "truncate font-medium text-(--ui-text-primary)", children: row.id }),
                      jsx("div", { className: "truncate text-[11px] text-(--ui-text-tertiary)", children: (row.pipeline_tag || "model") + (row.downloads ? " · " + Number(row.downloads).toLocaleString() + " pulls" : "") }),
                    ],
                  }),
                  jsx(Button, {
                    size: "xs",
                    variant: "secondary",
                    onClick: () => {
                      const engPart = customEngine !== "auto" ? " and engine=" + JSON.stringify(customEngine) : "";
                      triggerAgentAction(
                        "Pull " + row.id,
                        "Call model_forge_install with model_id=" + JSON.stringify(row.id) + ", target=" + JSON.stringify(selectedTarget) + engPart + ".",
                      );
                    },
                    children: "Pull",
                  }),
                ],
              })),
            }),
          ],
        }) : null,

        // Modality chips + filter
        jsxs("div", {
          className: "flex flex-wrap items-center justify-between gap-2",
          children: [
            jsx("div", {
              className: "flex flex-wrap gap-1",
              children: MODALITY_OPTIONS.map((opt) => jsx(Button, {
                key: opt.id,
                size: "xs",
                variant: modality === opt.id ? "default" : "chip",
                onClick: () => setModality(opt.id),
                children: opt.label,
              })),
            }),
            jsx(SearchField, { variant: "box", containerClassName: "w-56", placeholder: "Filter curated catalog…", value: search, onChange: setSearch }),
          ],
        }),

        // Curated catalog grid
        modelRows.length === 0
          ? jsx(EmptyState, { title: "No curated models match", description: "Click 'Search Hugging Face' above to search all open-weight models live." })
          : jsx("div", {
              className: "grid gap-2.5 md:grid-cols-2",
              children: modelRows.map((m) => {
                const fit = m.fit || {};
                const badge = fitBadgeMeta(fit.verdict || m.status);
                const isCoreGguf = m.id === "core-gguf-catalog";
                const canActivate = ["chat", "vision", "image", "image_edit", "video"].includes(m.modality) && !isCoreGguf;
                const matchedRunning = (m.engines || []).find((id) => runningEngineIds.has(id));
                return jsxs(Card, {
                  key: m.id,
                  className: "flex flex-col justify-between gap-2.5",
                  children: [
                    jsxs("div", {
                      className: "grid gap-1",
                      children: [
                        jsxs("div", {
                          className: "flex items-start justify-between gap-2",
                          children: [
                            jsx("div", { className: "text-sm font-medium text-(--ui-text-primary)", children: m.name }),
                            jsx(Badge, { variant: badge.variant, children: badge.label }),
                          ],
                        }),
                        jsxs("div", {
                          className: "flex flex-wrap items-center gap-1.5 text-[11px] text-(--ui-text-tertiary)",
                          children: [
                            jsx(Badge, { variant: "outline", size: "xs", children: m.modality_label || m.modality }),
                            m.size_gb ? jsx("span", { children: "~" + m.size_gb + " GB" }) : null,
                            (m.engines || []).length ? jsx("span", { children: "Engines: " + m.engines.join(", ") }) : null,
                            matchedRunning ? jsx(Badge, { variant: "success", size: "xs", children: matchedRunning + " ready" }) : null,
                          ],
                        }),
                        (fit.advisory || m.note) ? jsx("div", { className: "text-xs leading-relaxed text-(--ui-text-secondary)", children: fit.advisory || m.note }) : null,
                      ],
                    }),
                    jsxs("div", {
                      className: "flex flex-wrap items-center gap-1.5 pt-1",
                      children: [
                        isCoreGguf
                          ? jsx(Button, { size: "xs", onClick: () => host.navigate("/settings?tab=providers"), children: "Open Hermes Local Models" })
                          : jsx(Button, {
                              size: "xs",
                              variant: matchedRunning ? "default" : "secondary",
                              disabled: m.status === "stub",
                              onClick: () => triggerAgentAction("Pull " + m.name, "Call model_forge_install with model_id=" + JSON.stringify(m.id) + " and target=" + JSON.stringify(selectedTarget) + "."),
                              children: "Pull Model",
                            }),
                        canActivate ? jsx(Button, {
                          size: "xs",
                          variant: "outline",
                          onClick: () => triggerAgentAction("Wire " + m.name + " into Hermes", "Call model_forge_activate with model_id=" + JSON.stringify(m.id) + ", target=" + JSON.stringify(selectedTarget) + ", and apply=true so I can approve wiring it natively into Hermes."),
                          children: "Use in Hermes",
                        }) : null,
                      ],
                    }),
                  ],
                });
              }),
            }),
        jsx(Hint, { children: "Fit badges are advisory estimates; each inference engine does authoritative memory sizing at launch." }),
      ],
    });
  };

  const renderComputeTab = () => jsxs("div", {
    className: "grid gap-4",
    children: [
      jsxs(Card, {
        className: "grid gap-2",
        children: [
          jsxs("div", {
            className: "flex flex-wrap items-center justify-between gap-2",
            children: [
              jsxs("div", {
                children: [
                  jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "This Machine (Local)" }),
                  jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: localGpuSummary }),
                ],
              }),
              jsx(Badge, { variant: localHw && localHw.total_vram_gb > 0 ? "success" : "muted", children: localHw && localHw.total_vram_gb > 0 ? localHw.total_vram_gb + " GB VRAM" : "Auto-detected" }),
            ],
          }),
          localHw && localHw.ram_gb ? jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "System RAM: " + localHw.ram_gb + " GB · Platform: " + (localHw.platform || "") }) : null,
        ],
      }),
      jsxs("div", {
        className: "flex items-center justify-between",
        children: [
          jsx("div", { className: "text-xs font-semibold uppercase tracking-wide text-(--ui-text-secondary)", children: "Network & Cloud GPUs (" + remoteTargets.length + ")" }),
          jsx(Button, { size: "sm", onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); }, children: "Add GPU / Cloud Server" }),
        ],
      }),
      remoteTargets.length === 0
        ? jsxs(Card, {
            className: "py-6 text-center",
            children: [
              jsx("div", { className: "text-sm font-medium text-(--ui-text-primary)", children: "No remote or cloud GPUs added yet" }),
              jsx("div", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: "Point Model Forge at a gaming PC on your network or a cloud GPU box without editing JSON." }),
              jsx("div", {
                className: "mt-3 flex justify-center gap-2",
                children: [
                  jsx(Button, { key: "lan", size: "sm", onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); }, children: "Connect Network GPU" }),
                  jsx(Button, { key: "cloud", size: "sm", variant: "outline", onClick: () => { setEditingTarget(null); setAddGpuMode("cloud"); setAddGpuOpen(true); }, children: "Connect Cloud GPU" }),
                ],
              }),
            ],
          })
        : jsx("div", {
            className: "grid gap-2.5",
            children: remoteTargets.map((t) => {
              const live = (t.engines || []).filter((e) => e.available);
              const fromSettings = (state.settings.targets || []).some((st) => st.id === t.id);
              return jsxs(Card, {
                key: t.id,
                className: "grid gap-2",
                children: [
                  jsxs("div", {
                    className: "flex flex-wrap items-start justify-between gap-2",
                    children: [
                      jsxs("div", {
                        children: [
                          jsxs("div", {
                            className: "flex items-center gap-2",
                            children: [
                              jsx(StatusDot, { tone: live.length ? "good" : "muted" }),
                              jsx("span", { className: "text-sm font-semibold text-(--ui-text-primary)", children: t.label || t.id }),
                              jsx(Badge, { variant: "outline", size: "xs", children: t.kind }),
                              t.hardware && t.hardware.total_vram_gb ? jsx(Badge, { variant: "default", size: "xs", children: t.hardware.total_vram_gb + " GB VRAM" }) : null,
                            ],
                          }),
                          jsx("div", { className: "mt-0.5 font-mono text-[11px] text-(--ui-text-quaternary)", children: t.base_url }),
                        ],
                      }),
                      jsxs("div", {
                        className: "flex items-center gap-1.5",
                        children: [
                          jsx(Button, {
                            size: "xs",
                            variant: "outline",
                            onClick: () => {
                              const found = (state.settings.targets || []).find((st) => st.id === t.id) || {
                                id: t.id, label: t.label, base_url: t.base_url, kind: t.kind,
                                vram_gb: (t.hardware && t.hardware.total_vram_gb) || 0,
                                ram_gb: (t.hardware && t.hardware.ram_gb) || 0,
                              };
                              setEditingTarget(found);
                              setAddGpuMode(t.kind || "lan");
                              setAddGpuOpen(true);
                            },
                            children: "Edit",
                          }),
                          fromSettings
                            ? jsx(Button, { size: "xs", variant: "ghost", onClick: () => setRemoveConfirm({ kind: "target", id: t.id, label: t.label || t.id }), children: "Remove" })
                            : jsx(Button, { size: "xs", variant: "ghost", onClick: () => triggerAgentAction("Remove target " + t.label, "Call model_forge_target_add with action=\"remove\" and id=" + JSON.stringify(t.id) + "."), children: "Remove…" }),
                        ],
                      }),
                    ],
                  }),
                  live.length
                    ? jsx("div", { className: "flex flex-wrap gap-1.5", children: live.map((e, i) => jsx(Badge, { key: i, variant: "success", size: "xs", children: e.name + (e.detail ? " · " + e.detail : "") })) })
                    : jsx(Hint, { children: "No engine answered at " + t.base_url + " during the last backend probe." }),
                ],
              });
            }),
          }),
    ],
  });

  const renderEnginesTab = () => jsxs("div", {
    className: "grid gap-4",
    children: [
      jsxs("div", {
        className: "flex flex-wrap items-center justify-between gap-2",
        children: [
          jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: "Start an installed local engine, install a vendor engine behind approval, or connect any running URL." }),
          jsx(Button, { size: "sm", onClick: () => { setEditingTarget(null); setAddGpuMode("engine"); setAddGpuOpen(true); }, children: "Connect Engine URL" }),
        ],
      }),
      jsx("div", {
        className: "grid gap-2.5 md:grid-cols-2",
        children: ENGINE_GUIDE.map((eng) => {
          const live = runningEngines.find((r) => r.id === eng.id) || null;
          const cli = (state.clis && state.clis[eng.id]) || null;
          const installed = Boolean(cli && cli.installed);
          const managedRunning = Boolean(cli && cli.managed_running);
          return jsxs(Card, {
            key: eng.id,
            className: "flex flex-col justify-between gap-2.5",
            children: [
              jsxs("div", {
                className: "grid gap-1",
                children: [
                  jsxs("div", {
                    className: "flex items-center justify-between gap-2",
                    children: [
                      jsxs("div", {
                        className: "flex items-center gap-2",
                        children: [
                          jsx(StatusDot, { tone: live ? "good" : installed ? "warn" : "muted" }),
                          jsx("span", { className: "text-sm font-semibold text-(--ui-text-primary)", children: eng.name }),
                        ],
                      }),
                      jsx(Badge, {
                        variant: live ? "success" : installed ? "default" : "muted",
                        size: "xs",
                        children: live ? "Running" : installed ? "Installed (stopped)" : "Not detected",
                      }),
                    ],
                  }),
                  jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: eng.summary }),
                  jsx("div", { className: "text-[11px] text-(--ui-text-quaternary)", children: "Modalities: " + eng.modalities.join(", ") + " · port :" + eng.defaultPort }),
                  live ? jsx("div", { className: "text-[11px] text-(--ui-accent)", children: "Active at " + (live.root || "") + (live.detail ? " — " + live.detail : "") }) : null,
                ],
              }),
              jsxs("div", {
                className: "flex flex-wrap items-center gap-1.5",
                children: [
                  eng.installMode === "core_managed"
                    ? jsx(Button, { size: "xs", onClick: () => host.navigate("/settings?tab=providers"), children: "Open Local Models" })
                    : managedRunning
                      ? jsx(Button, {
                          size: "xs",
                          variant: "destructive",
                          onClick: () => triggerAgentAction("Stop " + eng.name, "Call model_forge_install with action=\"engine_stop\" and engine=" + JSON.stringify(eng.id) + "."),
                          children: "Stop " + eng.name,
                        })
                      : (!live && installed && eng.canStart && eng.id !== "vllm")
                        ? jsx(Button, {
                            size: "xs",
                            onClick: () => triggerAgentAction("Start " + eng.name, "Call model_forge_install with action=\"engine_start\", engine=" + JSON.stringify(eng.id) + ", and execute=true so I can approve launching it locally on 127.0.0.1."),
                            children: "Start " + eng.name,
                          })
                        : eng.installMode === "approved_installer"
                          ? jsx(Button, {
                              size: "xs",
                              variant: live ? "secondary" : "default",
                              onClick: () => triggerAgentAction("Install " + eng.name, "Call model_forge_install with action=\"engine_install\", engine=" + JSON.stringify(eng.id) + ", and execute=true so I can review the pinned vendor installer."),
                              children: live ? "Update…" : "Install " + eng.name,
                            })
                          : jsx(Button, { size: "xs", variant: "secondary", onClick: () => setEngineCmdDialog(eng), children: "Quick-Start Command" }),
                  jsx(Button, {
                    size: "xs",
                    variant: "outline",
                    onClick: () => {
                      setEditingTarget({ label: eng.name, base_url: "http://127.0.0.1:" + eng.defaultPort, kind: "engine" });
                      setAddGpuMode("engine");
                      setAddGpuOpen(true);
                    },
                    children: "Connect URL",
                  }),
                ],
              }),
            ],
          });
        }),
      }),
      (state.settings.declared || []).length > 0 ? jsxs(Card, {
        className: "grid gap-2",
        children: [
          jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Connected Engine URLs (" + state.settings.declared.length + ")" }),
          ...state.settings.declared.map((d, i) => jsxs("div", {
            key: "dec-" + i,
            className: "flex items-center justify-between gap-2 border-t border-(--ui-stroke-tertiary) pt-2 text-xs",
            children: [
              jsxs("div", {
                children: [
                  jsx("span", { className: "font-medium text-(--ui-text-primary)", children: d.label + " " }),
                  jsx("span", { className: "font-mono text-[11px] text-(--ui-text-quaternary)", children: d.base_url }),
                ],
              }),
              jsx(Button, { size: "xs", variant: "ghost", onClick: () => setRemoveConfirm({ kind: "engine", label: d.label, base_url: d.base_url }), children: "Remove" }),
            ],
          })),
        ],
      }) : null,
    ],
  });

  const renderWiringTab = () => {
    // Both OpenAI /v1 media servers AND raw ComfyUI endpoints work natively now!
    const mediaCandidates = runningEngines.filter((e) =>
      (e.id === "comfyui" || e.id === "vllm" || e.id === "sglang" || e.id === "openai-compatible") && e.root,
    );
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        jsxs(Card, {
          className: "grid gap-1",
          children: [
            jsx("div", { className: "mb-1 text-sm font-semibold text-(--ui-text-primary)", children: "Credentials & Package Setup" }),
            jsx(ListRow, {
              title: "Hugging Face Token",
              description: hfSet ? "Connected in Hermes' credential store (sent only to huggingface.co)." : "Optional. Add a free Read token for gated models and faster downloads.",
              action: jsxs("div", {
                className: "flex items-center gap-2",
                children: [
                  jsx(Badge, { variant: hfSet ? "success" : "muted", children: hfSet ? "Connected" : "Not set" }),
                  jsx(Button, { size: "sm", variant: hfSet ? "outline" : "default", onClick: () => setHfOpen(true), children: hfSet ? "Manage" : "Add Token" }),
                ],
              }),
            }),
            jsx(ListRow, {
              title: "Unified Model Forge Mode",
              description: "Tools, Desktop UI, and native image/video providers (including direct ComfyUI workflows) are active inside this single package. Install Model Forge Providers for dedicated Strata/Maya /model profiles.",
              action: jsx(Button, { size: "sm", variant: "outline", onClick: () => host.navigate("/capabilities?tab=plugins"), children: "Plugin Store" }),
            }),
          ],
        }),

        jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between",
              children: [
                jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Active Hermes Piping" }),
                jsx(Button, { size: "xs", variant: "outline", onClick: () => host.navigate("/settings?tab=providers"), children: "Providers Settings" }),
              ],
            }),
            activeWiring && activeWiring.ok ? jsxs("div", {
              className: "grid gap-1 text-xs text-(--ui-text-secondary)",
              children: [
                jsx("div", { children: "Chat default: " + (activeWiring.chat.model || "not set") + (activeWiring.chat.provider ? " (" + activeWiring.chat.provider + ")" : "") }),
                activeWiring.vision && activeWiring.vision.model ? jsx("div", { children: "Auxiliary vision: " + activeWiring.vision.model }) : null,
                jsx("div", { children: "Image provider: " + (activeWiring.image.provider || "default") + (imgUrlDraft ? " → " + imgUrlDraft : "") }),
                jsx("div", { children: "Video provider: " + (activeWiring.video.provider || "default") + (vidUrlDraft ? " → " + vidUrlDraft : "") }),
              ],
            }) : jsx(Hint, { children: "Click 'Use in Hermes' on any model in the Models tab to pipe it natively into Hermes." }),
          ],
        }),

        jsxs("div", {
          className: "grid gap-3 md:grid-cols-2",
          children: [
            jsxs(Card, {
              className: "grid gap-2.5",
              children: [
                jsxs("div", {
                  className: "flex items-center justify-between",
                  children: [
                    jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Image Generation Pipe" }),
                    jsx(Badge, { variant: imgUrlDraft ? "success" : "muted", size: "xs", children: imgUrlDraft ? "Configured" : "Not wired" }),
                  ],
                }),
                jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Accepts either a raw ComfyUI URL (e.g. http://127.0.0.1:8188 — native workflows!) or an OpenAI /v1 images API." }),
                jsx(Input, { value: imgUrlDraft, onChange: (e) => setImgUrlDraft(e.target.value), placeholder: "http://127.0.0.1:8188 or http://gpu-box.example.com:8080/v1" }),
                jsx(Input, { value: imgModelDraft, onChange: (e) => setImgModelDraft(e.target.value), placeholder: "Default model or workflow (blank = auto)" }),
                mediaCandidates.length > 0 ? jsx("div", {
                  className: "flex flex-wrap gap-1",
                  children: mediaCandidates.slice(0, 4).map((e, i) => jsx(Button, {
                    key: "mc-" + i,
                    size: "xs",
                    variant: "chip",
                    onClick: () => setImgUrlDraft(e.id === "comfyui" ? rootUrl(e.root) : rootUrl(e.root) + "/v1"),
                    children: e.id === "comfyui" ? "Use ComfyUI (direct)" : "Use " + e.name,
                  })),
                }) : null,
                jsxs("div", {
                  className: "flex gap-2",
                  children: [
                    jsx(Button, { size: "xs", disabled: savingMedia, onClick: () => handleSaveMedia("image"), children: "Save Endpoint" }),
                    jsx(Button, {
                      size: "xs",
                      variant: "outline",
                      disabled: !imgUrlDraft.trim(),
                      onClick: () => triggerAgentAction("Activate Hermes Image Pipe", "Call model_forge_activate with model_id=" + JSON.stringify(imgModelDraft.trim() || "qwen-image-2.1") + ", surface=\"image\", target=" + JSON.stringify(selectedTarget) + ", and apply=true."),
                      children: "Set as Default…",
                    }),
                  ],
                }),
              ],
            }),

            jsxs(Card, {
              className: "grid gap-2.5",
              children: [
                jsxs("div", {
                  className: "flex items-center justify-between",
                  children: [
                    jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Video Generation Pipe" }),
                    jsx(Badge, { variant: vidUrlDraft ? "success" : "muted", size: "xs", children: vidUrlDraft ? "Configured" : "Not wired" }),
                  ],
                }),
                jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Accepts either a raw ComfyUI URL (e.g. http://127.0.0.1:8188 — native workflows!) or an OpenAI /v1 videos API." }),
                jsx(Input, { value: vidUrlDraft, onChange: (e) => setVidUrlDraft(e.target.value), placeholder: "http://127.0.0.1:8188 or http://gpu-box.example.com:8080/v1" }),
                jsx(Input, { value: vidModelDraft, onChange: (e) => setVidModelDraft(e.target.value), placeholder: "Default model or workflow (blank = auto)" }),
                mediaCandidates.length > 0 ? jsx("div", {
                  className: "flex flex-wrap gap-1",
                  children: mediaCandidates.slice(0, 4).map((e, i) => jsx(Button, {
                    key: "vc-" + i,
                    size: "xs",
                    variant: "chip",
                    onClick: () => setVidUrlDraft(e.id === "comfyui" ? rootUrl(e.root) : rootUrl(e.root) + "/v1"),
                    children: e.id === "comfyui" ? "Use ComfyUI (direct)" : "Use " + e.name,
                  })),
                }) : null,
                jsxs("div", {
                  className: "flex gap-2",
                  children: [
                    jsx(Button, { size: "xs", disabled: savingMedia, onClick: () => handleSaveMedia("video"), children: "Save Endpoint" }),
                    jsx(Button, {
                      size: "xs",
                      variant: "outline",
                      disabled: !vidUrlDraft.trim(),
                      onClick: () => triggerAgentAction("Activate Hermes Video Pipe", "Call model_forge_activate with model_id=" + JSON.stringify(vidModelDraft.trim() || "ltx-2.5") + ", surface=\"video\", target=" + JSON.stringify(selectedTarget) + ", and apply=true."),
                      children: "Set as Default…",
                    }),
                  ],
                }),
              ],
            }),
          ],
        }),
      ],
    });
  };

  const renderStorageTab = () => {
    const entries = (state.cockpit && state.cockpit.storage && state.cockpit.storage.entries) || [];
    const res = state.cockpit && state.cockpit.residency;
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        renderJobsBanner(),
        jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Installed Model Weights (" + entries.length + ")" }),
            entries.length === 0
              ? jsx(Hint, { children: "No downloaded weights found yet in Model Forge downloads, Ollama, or Hugging Face cache." })
              : jsx("div", {
                  className: "grid gap-1",
                  children: entries.map((e, i) => jsxs("div", {
                    key: "st-" + i,
                    className: "flex items-center justify-between gap-2 border-b border-(--ui-stroke-tertiary) py-1.5 text-xs",
                    children: [
                      jsxs("div", {
                        className: "min-w-0",
                        children: [
                          jsx("div", { className: "truncate font-medium text-(--ui-text-primary)", children: e.name }),
                          jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: fmtBytes(e.size_bytes) + " · " + (e.root_label || e.root || "") + (e.last_modified ? " · " + e.last_modified : "") }),
                        ],
                      }),
                      jsx(Button, {
                        size: "xs",
                        variant: "ghost",
                        onClick: () => triggerAgentAction("Delete " + e.name, "Call model_forge_storage with action=\"delete\" and entry=" + JSON.stringify(e.id || e.path) + " so I can approve deleting this file."),
                        children: "Delete…",
                      }),
                    ],
                  })),
                }),
            jsx(ResourceView, { ctx }),
          ],
        }),
        res && res.ok !== false ? jsxs(Card, {
          className: "grid gap-1.5",
          children: [
            jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Broker Weight Residency" }),
            jsx(Hint, {
              children: "budget: " + ((res.budget && res.budget.nominal_gb != null) ? res.budget.nominal_gb + " GB nominal" : "undeclared") +
                " · booked_now " + ((res.budget && res.budget.booked_now_gb != null) ? res.budget.booked_now_gb + " GB" : "?") +
                ((res.budget && res.budget.headroom_gb != null) ? " · headroom " + res.budget.headroom_gb + " GB" : ""),
            }),
            jsx(Hint, { children: "resident (" + (res.resident || []).length + ") · booked (" + (res.booked || []).length + ") · pulled (" + (res.pulled || []).length + ")" }),
          ],
        }) : null,
      ],
    });
  };

  if (needsFirstRun) {
    return jsxs("div", {
      className: "h-full overflow-y-auto",
      children: [
        renderQuickstartHero(),
        jsx(HfCredentialsDialog, { open: hfOpen, onOpenChange: setHfOpen, connected: hfSet, onSaved: load }),
        jsx(AddComputeDialog, { ctx, open: addGpuOpen, onOpenChange: setAddGpuOpen, initialTarget: editingTarget, existingTargets: state.settings.targets, existingEngines: state.settings.declared, defaultMode: addGpuMode, onSaved: () => { setShowFullCockpit(true); load(); } }),
        jsx(ActionPromptDialog, { action: pendingAction, onClose: () => setPendingAction(null) }),
      ],
    });
  }

  return jsxs("div", {
    className: "mx-auto flex h-full max-w-5xl flex-col gap-4 overflow-y-auto px-6 py-5",
    children: [
      jsxs("div", {
        className: "flex flex-wrap items-start justify-between gap-3",
        children: [
          jsxs("div", {
            children: [
              jsxs("div", {
                className: "flex items-center gap-2",
                children: [
                  jsx(Codicon, { name: "layers", size: "1.1rem" }),
                  jsx("h1", { className: "text-lg font-semibold text-(--ui-text-primary)", children: "Model Forge" }),
                  jsx(Badge, { variant: "success", size: "xs", children: "Unified" }),
                ],
              }),
              jsx("p", { className: "mt-0.5 text-xs text-(--ui-text-secondary)", children: "Local open-weight models for every modality — detected via backend, piped natively into Hermes." }),
            ],
          }),
          jsxs("div", {
            className: "flex flex-wrap items-center gap-2",
            children: [
              targetOptions.length > 1 ? jsx(SegmentedControl, {
                value: selectedTarget,
                onChange: (tid) => { setSelectedTarget(tid); runLiveBrowse({ target: tid, fresh: false }); },
                sizing: "content",
                options: targetOptions,
              }) : null,
              jsx(Button, { size: "sm", variant: "outline", onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); }, children: "Add GPU / Server" }),
              jsx(Button, { size: "sm", variant: "ghost", onClick: load, children: "Refresh" }),
            ],
          }),
        ],
      }),

      // 3-Card Status Strip
      jsxs("div", {
        className: "grid gap-2.5 sm:grid-cols-3",
        children: [
          jsxs("button", {
            type: "button",
            onClick: () => setTab("compute"),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", { className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)", children: ["GPU & Compute", jsx(StatusDot, { tone: (localHw && localHw.total_vram_gb > 0) || remoteTargets.length > 0 ? "good" : "warn" })] }),
              jsx("div", { className: "mt-1 truncate text-[11px] text-(--ui-text-secondary)", children: remoteTargets.length > 0 ? localGpuSummary + " + " + remoteTargets.length + " remote" : localGpuSummary }),
            ],
          }),
          jsxs("button", {
            type: "button",
            onClick: () => setHfOpen(true),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", { className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)", children: ["Hugging Face", jsx(StatusDot, { tone: hfSet ? "good" : "muted" })] }),
              jsx("div", { className: "mt-1 text-[11px] text-(--ui-text-secondary)", children: hfSet ? "Token connected" : "Add token for gated models" }),
            ],
          }),
          jsxs("button", {
            type: "button",
            onClick: () => setTab("engines"),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", { className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)", children: ["Inference Engines", jsx(StatusDot, { tone: runningEngines.length > 0 ? "good" : "warn" })] }),
              jsx("div", { className: "mt-1 truncate text-[11px] text-(--ui-text-secondary)", children: runningEngines.length > 0 ? runningEngines.length + " active (" + runningEngines.slice(0, 2).map((e) => e.name).join(", ") + ")" : "Click to start, install, or connect" }),
            ],
          }),
        ],
      }),

      ...(state.errors || []).map((msg, i) => jsx("div", {
        key: "err-" + i,
        className: "rounded border border-(--ui-stroke-secondary) px-3 py-1.5 text-xs text-(--ui-text-secondary)",
        children: msg,
      })),

      jsx(SegmentedControl, {
        value: tab,
        onChange: setTab,
        sizing: "content",
        options: [
          { id: "models", label: "Models" },
          { id: "compute", label: "Compute & GPUs" },
          { id: "engines", label: "Engines" },
          { id: "wiring", label: "Hermes Piping & Keys" },
          { id: "storage", label: "Storage & Memory" },
        ],
      }),

      jsx(Separator, {}),

      tab === "models" ? renderModelsTab() : null,
      tab === "compute" ? renderComputeTab() : null,
      tab === "engines" ? renderEnginesTab() : null,
      tab === "wiring" ? renderWiringTab() : null,
      tab === "storage" ? renderStorageTab() : null,

      jsx(HfCredentialsDialog, { open: hfOpen, onOpenChange: setHfOpen, connected: hfSet, onSaved: load }),
      jsx(AddComputeDialog, { ctx, open: addGpuOpen, onOpenChange: setAddGpuOpen, initialTarget: editingTarget, existingTargets: state.settings.targets, existingEngines: state.settings.declared, defaultMode: addGpuMode, onSaved: load }),
      jsx(ActionPromptDialog, { action: pendingAction, onClose: () => setPendingAction(null) }),
      jsx(ConfirmDialog, {
        open: Boolean(removeConfirm),
        onClose: () => setRemoveConfirm(null),
        onConfirm: handleConfirmRemove,
        destructive: true,
        confirmLabel: "Remove",
        title: removeConfirm ? "Remove " + removeConfirm.label + "?" : "Remove?",
        description: "Only removes this saved address from Model Forge settings. Nothing on the remote machine is deleted.",
      }),
      engineCmdDialog ? jsx(Dialog, {
        open: Boolean(engineCmdDialog),
        onOpenChange: (v) => { if (!v) setEngineCmdDialog(null); },
        children: jsxs(DialogContent, {
          className: "max-w-md",
          children: [
            jsxs(DialogHeader, {
              children: [
                jsx(DialogTitle, { children: "Start " + engineCmdDialog.name }),
                jsx(DialogDescription, { children: engineCmdDialog.summary }),
              ],
            }),
            jsxs("div", {
              className: "grid gap-2 py-2",
              children: [
                jsx("div", { className: "rounded border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-2.5 font-mono text-xs text-(--ui-text-primary) break-all", children: engineCmdDialog.quickStart }),
                jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Once running on port :" + engineCmdDialog.defaultPort + ", click Refresh and Model Forge detects it automatically." }),
              ],
            }),
            jsxs(DialogFooter, {
              children: [
                jsx(CopyButton, { text: engineCmdDialog.quickStart, label: "Copy Command", showLabel: true, buttonSize: "sm", buttonVariant: "outline" }),
                jsx(Button, { size: "sm", onClick: () => setEngineCmdDialog(null), children: "Done" }),
              ],
            }),
          ],
        }),
      }) : null,
    ],
  });
}

// --- Compact side pane -------------------------------------------------------

function CompactSidePane(props) {
  const ctx = props.ctx;
  const [state, setState] = useState(null);
  const load = () => { loadUnifiedState(ctx).then(setState); };
  useEffect(() => {
    load();
    refreshHooks.add(load);
    const disposer = ctx.onEvent("plugin.model-forge.settings.changed", load);
    return () => { refreshHooks.delete(load); disposer(); };
  }, []);
  if (!state) return jsx("div", { className: "p-3 text-xs text-(--ui-text-secondary)", children: "Loading Model Forge…" });
  const targets = (state.cockpit && state.cockpit.targets) || [];
  const liveCount = targets.reduce((n, t) => n + ((t.engines || []).filter((e) => e.available).length), 0);
  return jsxs("div", {
    className: "grid gap-2.5 overflow-y-auto p-3",
    children: [
      jsxs("div", {
        className: "flex items-center justify-between gap-2",
        children: [
          jsx(Badge, { variant: liveCount > 0 ? "success" : "muted", children: liveCount > 0 ? liveCount + " engine(s) live" : "No engine active" }),
          jsx(Button, { size: "xs", onClick: () => host.navigate(MODEL_FORGE_ROUTE), children: "Open Full View" }),
        ],
      }),
      ...targets.map((t) => jsxs(Card, {
        key: t.id,
        className: "p-2.5 text-xs",
        children: [
          jsx("div", { className: "font-semibold text-(--ui-text-primary)", children: t.label || t.id }),
          jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: t.base_url }),
          ...((t.engines || []).filter((e) => e.available)).map((e, i) => jsxs("div", {
            key: "e-" + i,
            className: "mt-1 flex items-center gap-1.5 text-[11px] text-(--ui-text-secondary)",
            children: [jsx(StatusDot, { tone: "good" }), e.name, e.detail ? " · " + e.detail : ""],
          })),
        ],
      })),
      jsx(ResourceView, { ctx }),
    ],
  });
}

// --- Settings -> Plugins -> Model Forge pages --------------------------------

function SettingsOverviewPage(props) {
  const ctx = props.ctx;
  const [settings, setSettings] = useState(null);
  const [hfOpen, setHfOpen] = useState(false);
  const load = () => { readPluginSettings().then(setSettings).catch(() => null); };
  useEffect(() => { load(); }, []);
  const hfSet = Boolean(settings && settings.credentials && settings.credentials.HF_TOKEN);
  return jsxs("div", {
    className: "grid gap-3",
    children: [
      jsx(ListRow, {
        title: "Open Model Forge Workspace",
        description: "Browse open-weight models, search Hugging Face live, start or connect engines, and wire models natively into Hermes.",
        action: jsx(Button, { size: "sm", onClick: () => host.navigate(MODEL_FORGE_ROUTE), children: "Open Model Forge" }),
      }),
      jsx(ListRow, {
        title: "Hugging Face Token",
        description: hfSet ? "Connected (stored in Hermes credentials; sent only to huggingface.co)." : "Add a free Read token for gated models and full-speed pulls.",
        action: jsx(Button, { size: "sm", variant: hfSet ? "outline" : "default", onClick: () => setHfOpen(true), children: hfSet ? "Manage Token" : "Add HF Token" }),
      }),
      jsx(HfCredentialsDialog, { open: hfOpen, onOpenChange: setHfOpen, connected: hfSet, onSaved: load }),
    ],
  });
}

function SettingsComputeSubpage(props) {
  const ctx = props.ctx;
  const [settings, setSettings] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const load = () => { readPluginSettings().then(setSettings).catch(() => null); };
  useEffect(() => { load(); }, []);
  const targets = (settings && settings.targets) || [];
  return jsxs("div", {
    className: "grid gap-3",
    children: [
      jsx(ListRow, {
        title: "Remote & Cloud GPU Targets",
        description: "Connect another machine on your network or a cloud GPU box without editing JSON.",
        action: jsx(Button, { size: "sm", onClick: () => { setEditing(null); setAddOpen(true); }, children: "Add GPU Target" }),
      }),
      ...targets.map((t) => jsx(ListRow, {
        key: t.id,
        title: t.label + " (" + t.kind + ")",
        description: t.base_url + (t.vram_gb ? " · " + t.vram_gb + " GB VRAM" : ""),
        action: jsx(Button, { size: "xs", variant: "outline", onClick: () => { setEditing(t); setAddOpen(true); }, children: "Edit" }),
      })),
      jsx(AddComputeDialog, {
        ctx,
        open: addOpen,
        onOpenChange: setAddOpen,
        initialTarget: editing,
        existingTargets: targets,
        existingEngines: (settings && settings.declared) || [],
        defaultMode: "lan",
        onSaved: load,
      }),
    ],
  });
}

export default {
  id: "model-forge",
  name: "Model Forge",
  register(ctx) {
    ctx.register({
      id: "model-forge-page",
      area: ROUTES_AREA,
      data: { path: MODEL_FORGE_ROUTE },
      render: () => jsx(ModelForgeWorkspace, { ctx }),
    });
    ctx.register({
      id: "model-forge-nav",
      area: SIDEBAR_NAV_AREA,
      data: { path: MODEL_FORGE_ROUTE, label: "Model Forge", codicon: "layers" },
    });
    ctx.registerSettingsPage?.({
      id: "settings",
      title: "Model Forge",
      icon: "layers",
      render: () => jsx(SettingsOverviewPage, { ctx }),
      children: [
        { id: "compute", title: "Compute & GPUs", render: () => jsx(SettingsComputeSubpage, { ctx }) },
      ],
    });
    ctx.register({
      id: "model-forge-pane",
      area: "panes",
      title: "Model Forge",
      data: { placement: "right" },
      render: () => jsx(CompactSidePane, { ctx }),
    });
    ctx.register({
      id: "model-forge-open",
      area: PALETTE_AREA,
      title: "Model Forge: open workspace",
      data: {
        id: "model-forge.open",
        label: "Model Forge: Open",
        keywords: ["model", "forge", "gpu", "local", "ollama", "comfyui"],
        run: () => host.navigate(MODEL_FORGE_ROUTE),
      },
      run: () => host.navigate(MODEL_FORGE_ROUTE),
    });
    ctx.register({
      id: "model-forge-refresh",
      area: PALETTE_AREA,
      title: "Model Forge: refresh",
      data: {
        id: "model-forge.refresh",
        label: "Model Forge: Refresh",
        keywords: ["model", "forge", "refresh"],
        run: () => {
          notifyRefreshHooks();
          host.notify({ kind: "info", message: "Model Forge refreshed." });
        },
      },
      run: () => {
        notifyRefreshHooks();
        host.notify({ kind: "info", message: "Model Forge refreshed." });
      },
    });
  },
};
