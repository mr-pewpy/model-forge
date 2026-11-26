// Model Forge — native Hermes Desktop product UI.
//
// Surfaces (100% documented @hermes/plugin-sdk only):
//   - Full workspace page (/model-forge) + Sidebar nav entry ("Model Forge")
//   - Settings -> Plugins -> Model Forge (Setup & Credentials, Compute & GPUs,
//     Engines & Media — zero raw JSON editing)
//   - Dockable cockpit side pane (compact glanceable view + Open Full View)
//   - Command palette commands (Open, Add GPU, Refresh, Settings)
//
// Data doors (documented SDK & gateway RPCs only):
//   - plugins.manage list/presence — suite status (all 5 Model Forge packages),
//     settings (targets_json, engines_json, download_roots, image/video URLs),
//     and boolean credential presence (never token values)
//   - plugins.manage settings — saves GUI-edited compute targets, declared
//     engine endpoints, and image/video backend URLs without raw JSON
//   - model.save_key / model.disconnect (slug: "huggingface") — official
//     Hermes credential lifecycle for Hugging Face pull token
//   - machine.facts — immediate local OS/CPU/RAM/GPU-class summary (works
//     even before any chat session is opened)
//   - slash.exec (/model-forge cockpit & /model-forge resources) — full
//     hardware probe, curated fit rows, storage inventory, active wiring,
//     broker residency, and the quiet on-demand Resource view
//   - shell.exec — reads tool-registered targets from the plugin data dir
//   - fetch() — read-only live engine probes against user targets/endpoints
//     (llama.cpp, Ollama, Project Maya, ComfyUI, vLLM, SGLang, Exo, generic)
//
// Policy & honesty:
//   - Resource view is quiet by design: collapsed by default, loads once on
//     expand, manual Refresh only, zero timers or polling.
//   - Destructive / config-rewriting / installer tool actions route through
//     Hermes chat so every one passes the human-approval gate; this UI never
//     deletes files or runs shell installers directly.

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

const DEFAULT_TARGETS = [
  { id: "local", label: "This machine", base_url: "http://127.0.0.1:8080", kind: "local", source: "default" },
];

// Local scan ports mirror model_forge/engines/: llama.cpp/Ollama/Maya/Strata
// (8080/8081/8000/8001/11434), vLLM (8000), ComfyUI (8188), SGLang (30000),
// Exo (52415).
const LOCAL_PROBE_PORTS = [8080, 8081, 8000, 8001, 11434, 8188, 30000, 52415];
const PROBE_TIMEOUT_MS = 2500;
const MODEL_FORGE_ROUTE = "/model-forge";

const SUITE_PACKAGES = [
  { name: "model-forge", label: "Model Forge Core", role: "Hardware detection, model catalog, pulls, broker, activation tools" },
  { name: "model-forge-desktop", label: "Model Forge Desktop", role: "Native Desktop page, guided setup wizards, and cockpit pane" },
  { name: "model-forge-providers", label: "Model Forge Providers", role: "Registers Strata, Project Maya, and Model Forge in /model" },
  { name: "model-forge-image", label: "Model Forge Image", role: "Pipes Hermes image_generate to local/LAN image engines" },
  { name: "model-forge-video", label: "Model Forge Video", role: "Pipes Hermes video_generate to local/LAN video engines" },
];

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

// Offline/pre-chat curated preview — replaced automatically by the live
// /model-forge cockpit bridge whenever a chat session is open.
const PREVIEW_MODELS = [
  { id: "core-gguf-catalog", name: "Hermes core GGUF catalog", modality: "chat", modality_label: "Chat / agent model", engines: ["llamacpp"], size_gb: 0, status: "validated", note: "Managed by Hermes built-in Local Models for GGUF chat." },
  { id: "qwen3.8-flash-next", name: "Qwen3.8 Flash Next (125B MoE)", modality: "chat", modality_label: "Chat / agent model", engines: ["strata", "maya"], size_gb: 66, status: "known", note: "Streams experts from RAM to GPU; 12 GB VRAM cards work as hot cache." },
  { id: "deepseek-v4.1-flash", name: "DeepSeek-V4.1-Flash-1017 (748B MoE)", modality: "chat", modality_label: "Chat / agent model", engines: ["strata", "maya"], size_gb: 277, status: "known", note: "Runs on 128 GB+ machines with Maya or Strata." },
  { id: "minimax-m3.1-flash", name: "MiniMax-M3.1-Flash (428B MoE)", modality: "chat", modality_label: "Chat / agent model", engines: ["maya"], size_gb: 93, status: "known", note: "Vision + thinking levels on Maya." },
  { id: "glm-5.3-flash", name: "GLM-5.3 Flash (321B MoE)", modality: "chat", modality_label: "Chat / agent model", engines: ["maya"], size_gb: 73, status: "known", note: "Vision capable; Maya tiers size automatically." },
  { id: "minicpm-v-4.6", name: "MiniCPM-V 4.6", modality: "vision", modality_label: "Image understanding", engines: ["vllm"], size_gb: 2.6, status: "validated", note: "Compact vision model for image understanding." },
  { id: "deepseek-ocr-2", name: "DeepSeek-OCR-2", modality: "ocr", modality_label: "Text extraction from images", engines: ["vllm"], size_gb: 6.8, status: "validated", note: "Document OCR with 8192 context on vLLM." },
  { id: "krea-2-turbo", name: "Krea-2 Turbo", modality: "image", modality_label: "Image generation", engines: ["comfyui"], size_gb: 19, status: "validated", note: "Fast 8-step image generation on ComfyUI." },
  { id: "qwen-image-2.1", name: "Qwen-Image 2.1", modality: "image", modality_label: "Image generation", engines: ["vllm", "comfyui"], size_gb: 25, status: "validated", note: "Native 2K, typography, and RGBA transparency." },
  { id: "qwen-image-2.1-edit", name: "Qwen-Image 2.1 Edit", modality: "image_edit", modality_label: "Image editing", engines: ["vllm", "comfyui"], size_gb: 25, status: "validated", note: "Reference-image editing on vLLM-Omni or ComfyUI." },
  { id: "flux-2-klein-4b", name: "FLUX.2 Klein 4B (edit / outpaint)", modality: "image_edit", modality_label: "Image editing", engines: ["sglang", "vllm", "comfyui"], size_gb: 4, status: "validated", note: "Compact Apache-2.0 image editor (8 GB+ VRAM)." },
  { id: "ltx-2.5", name: "LTX-2.5 (t2v / i2v)", modality: "video", modality_label: "Video generation", engines: ["comfyui"], size_gb: 20, status: "validated", note: "Text-to-video and image-to-video (gated HF repo)." },
  { id: "minimax-h3", name: "MiniMax H3 (image/video)", modality: "video", modality_label: "Video generation", engines: ["comfyui"], size_gb: 25, status: "known", note: "1024x576 video generation on 24 GB GPUs." },
  { id: "hunyuan3d-2", name: "Hunyuan3D-2 (image to GLB)", modality: "3d", modality_label: "3D model generation", engines: ["comfyui"], size_gb: 5, status: "validated", note: "Single-view image to textured 3D mesh via media gateway." },
  { id: "ace-step", name: "ACE-Step 1.5 turbo", modality: "music", modality_label: "Music generation", engines: ["python"], size_gb: 20, status: "validated", note: "Text-to-music via an OpenAI-style audio gateway." },
  { id: "chatterbox-turbo", name: "Chatterbox Turbo", modality: "voice_clone", modality_label: "Voice / voice clone", engines: ["python"], size_gb: 3, status: "validated", note: "TTS with voice cloning via /v1/audio/speech." },
];

const ENGINE_DIRECTORY = [
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
    summary: "Beginner-friendly local model runner. Pulls Ollama tags and Hugging Face GGUFs over HTTP.",
    quickStart: "ollama serve",
  },
  {
    id: "vllm",
    name: "vLLM / vLLM-Omni",
    modalities: ["chat", "vision", "ocr", "image", "image_edit"],
    defaultPort: 8000,
    installMode: "connect_or_manual",
    summary: "High-speed GPU server for chat, vision, OCR, and Omni image generation.",
    quickStart: "uvx vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000",
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
    id: "comfyui",
    name: "ComfyUI",
    modalities: ["image", "image_edit", "video", "3d"],
    defaultPort: 8188,
    installMode: "connect_or_manual",
    summary: "Visual workflow engine for local images, editing, video, and 3D meshes.",
    quickStart: "python main.py --listen 127.0.0.1 --port 8188",
  },
  {
    id: "strata",
    name: "Strata",
    modalities: ["chat", "vision"],
    defaultPort: 8080,
    installMode: "approved_installer",
    summary: "Runs huge MoE chat models on consumer GPUs and Apple Silicon by streaming experts.",
    quickStart: "strata launch --help",
  },
  {
    id: "maya",
    name: "Project Maya",
    modalities: ["chat", "vision", "ocr"],
    defaultPort: 8080,
    installMode: "approved_installer",
    summary: "Multi-GPU MoE server with built-in model catalog, HTTP pulls, and auto-swapping.",
    quickStart: "./setup.sh",
  },
  {
    id: "exo",
    name: "Exo",
    modalities: ["chat", "vision", "ocr"],
    defaultPort: 52415,
    installMode: "connect_or_manual",
    summary: "Pools multiple Macs or Linux GPU boxes on your network into one local cluster.",
    quickStart: "exo",
  },
];

const VRAM_PRESETS = [8, 12, 16, 24, 32, 48, 80];

const PANE_INVITATION = {
  reason: "No GPU target or running inference engine was detected yet.",
  next_steps: [
    "Click 'Add GPU / Server' to connect a computer on your network or a cloud GPU (no JSON needed).",
    "Or open the Engines tab to install Ollama / Strata / Maya or connect an existing engine URL.",
  ],
};

// Shared refresh hooks so palette commands and settings changes reload all mounted views.
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

function advisoryFitForTarget(model, probedRow) {
  if (!probedRow || !probedRow.target) return model.fit || {};
  if (model.status === "stub") return { verdict: "unavailable", advisory: "No local open engine exists for this modality yet." };
  const t = probedRow.target;
  const availIds = (probedRow.engines || []).filter((e) => e.ok && e.id).map((e) => e.id);
  const nonPhantom = (model.engines || []).filter((e) => e !== "python");
  const matched = nonPhantom.find((e) => availIds.includes(e)) || "";
  if (availIds.length > 0 && !matched && !availIds.includes("openai-compatible")) {
    return {
      verdict: "engine_unavailable",
      engine_available: false,
      engine: "",
      advisory: "No engine that can serve this model answered on " + t.label + " (needs: " + (nonPhantom.join(", ") || "an engine") + ")",
    };
  }
  const vram = Number(t.vram_gb) || 0;
  const ram = Number(t.ram_gb) || 0;
  const budget = vram + Math.max(0, ram - 8);
  const size = Number(model.size_gb) || 0;
  const vramMin = Number(model.vram_gb_min) || 0;
  if (model.engine_managed_fit) {
    if (model.ram_gb_min && ram > 0 && ram < model.ram_gb_min) {
      return { verdict: "needs_ram", advisory: "Declares " + model.ram_gb_min + " GB RAM minimum; " + t.label + " has " + ram + " GB" };
    }
    return { verdict: "engine_managed", advisory: "Engine sizes memory automatically on " + t.label };
  }
  if (!size) return { verdict: "unknown", advisory: "" };
  if (vramMin && vram >= vramMin) return { verdict: "fits_gpu", advisory: "Advisory: fits " + t.label + "'s " + vram + " GB VRAM" };
  if (vramMin && vram === 0) return { verdict: "needs_gpu", advisory: "Set VRAM for " + t.label + " to price GPU fit" };
  if (budget > 0 && size <= budget) return { verdict: "fits_memory", advisory: "Advisory: fits " + t.label + "'s " + budget + " GB memory pool" };
  return { verdict: "too_big", advisory: "Needs ~" + (vramMin || size) + " GB; " + t.label + " reports " + vram + " GB VRAM" };
}

function fitBadgeMeta(verdict) {
  const map = {
    fits_gpu: { label: "Fits your GPU", variant: "success", tone: "good" },
    fits_memory: { label: "Fits memory", variant: "default", tone: "good" },
    engine_managed: { label: "Auto-sized", variant: "default", tone: "good" },
    engine_unavailable: { label: "Needs engine", variant: "warn", tone: "warn" },
    needs_gpu: { label: "Needs GPU", variant: "warn", tone: "warn" },
    needs_ram: { label: "Needs more RAM", variant: "warn", tone: "warn" },
    too_big: { label: "Too large", variant: "destructive", tone: "bad" },
    unavailable: { label: "Coming soon", variant: "muted", tone: "muted" },
    unknown: { label: "Check size", variant: "outline", tone: "muted" },
  };
  return map[verdict] || { label: verdict || "Preview", variant: "outline", tone: "muted" };
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

// --- settings & suite state (plugins.manage + machine.facts) ------------------

function parsePluginRow(row) {
  const fields = (row && row.settings_schema) || [];
  const byKey = {};
  fields.forEach((f) => {
    if (f && f.key) byKey[f.key] = f;
  });
  return { row, fields, byKey };
}

async function readSettingsAndSuite() {
  try {
    const [listRes, presenceRes, factsRes] = await Promise.all([
      host.request("plugins.manage", { action: "list" }),
      host.request("plugins.manage", {
        action: "presence",
        names: SUITE_PACKAGES.map((p) => p.name),
      }).catch(() => null),
      host.request("machine.facts", {}).catch(() => null),
    ]);

    const pluginList = (listRes && Array.isArray(listRes.plugins)) ? listRes.plugins : [];
    const findPkg = (id) => pluginList.find((r) => r && (r.key === id || r.name === id)) || null;
    const coreRow = findPkg("model-forge");
    const imgRow = findPkg("model-forge-image");
    const vidRow = findPkg("model-forge-video");

    let rawTargets = null;
    let rawEngines = null;
    let rawDownloadRoots = null;
    let hfConfigured = false;
    let msConfigured = false;

    if (coreRow && Array.isArray(coreRow.settings_schema)) {
      const { byKey } = parsePluginRow(coreRow);
      rawTargets = byKey.targets_json ? byKey.targets_json.value : null;
      rawEngines = byKey.engines_json ? byKey.engines_json.value : null;
      rawDownloadRoots = byKey.download_roots ? byKey.download_roots.value : null;
      hfConfigured = Boolean(byKey.hf_token && byKey.hf_token.has_value);
      msConfigured = Boolean(byKey.modelscope_token && byKey.modelscope_token.has_value);
    } else if (listRes && listRes.entries) {
      const settings = (listRes.entries["model-forge"] || {}).settings || {};
      rawTargets = settings.targets_json;
      rawEngines = settings.engines_json;
      rawDownloadRoots = settings.download_roots;
    }

    const imgFields = parsePluginRow(imgRow).byKey;
    const vidFields = parsePluginRow(vidRow).byKey;

    const targets = (Array.isArray(rawTargets) ? rawTargets : [])
      .filter((t) => t && t.id && t.base_url)
      .map((t) => ({
        id: String(t.id),
        label: String(t.label || t.id),
        base_url: String(t.base_url),
        kind: String(t.kind || "lan"),
        vram_gb: Number(t.vram_gb) || 0,
        ram_gb: Number(t.ram_gb) || 0,
        note: String(t.note || ""),
        vram_source: String(t.vram_source || "declared"),
        ram_source: String(t.ram_source || "declared"),
        source: "settings",
      }));

    const declared = (Array.isArray(rawEngines) ? rawEngines : [])
      .filter((e) => e && /^https?:\/\//.test(String(e.base_url || "")))
      .map((e) => ({
        label: String(e.label || e.id || e.base_url),
        base_url: String(e.base_url).replace(/\/+$/, ""),
        source: "user-declared",
      }));

    const presenceRows = (presenceRes && Array.isArray(presenceRes.presence)) ? presenceRes.presence : [];
    const suite = SUITE_PACKAGES.map((pkg) => {
      const installed = findPkg(pkg.name);
      const pres = presenceRows.find((r) => r && (r.name === pkg.name || r.catalog_name === pkg.name)) || null;
      const isSelf = pkg.name === "model-forge-desktop";
      const enabled = isSelf
        ? true
        : Boolean(installed ? (installed.enabled !== false && installed.status !== "disabled") : (pres && pres.enabled));
      const present = isSelf || Boolean(installed) || Boolean(pres && pres.installed);
      return {
        ...pkg,
        installed: present,
        enabled,
        version: (installed && installed.version) || (pres && pres.version) || "",
      };
    });

    return {
      targets,
      declared,
      downloadRoots: Array.isArray(rawDownloadRoots) ? rawDownloadRoots : [],
      credentials: { HF_TOKEN: hfConfigured, MODELSCOPE_API_TOKEN: msConfigured },
      mediaSettings: {
        imageBaseUrl: String((imgFields.base_url && imgFields.base_url.value) || ""),
        imageDefaultModel: String((imgFields.default_model && imgFields.default_model.value) || ""),
        videoBaseUrl: String((vidFields.base_url && vidFields.base_url.value) || ""),
        videoDefaultModel: String((vidFields.default_model && vidFields.default_model.value) || ""),
      },
      suite,
      machineFacts: factsRes || null,
      error: "",
    };
  } catch (err) {
    return {
      targets: [],
      declared: [],
      downloadRoots: [],
      credentials: { HF_TOKEN: false, MODELSCOPE_API_TOKEN: false },
      mediaSettings: { imageBaseUrl: "", imageDefaultModel: "", videoBaseUrl: "", videoDefaultModel: "" },
      suite: SUITE_PACKAGES.map((p) => ({ ...p, installed: p.name === "model-forge-desktop", enabled: p.name === "model-forge-desktop", version: "" })),
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

// --- target sources -----------------------------------------------------------

async function readDataDirTargets() {
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
        id: String(t.id),
        label: String(t.label || t.id),
        base_url: String(t.base_url),
        kind: String(t.kind || "lan"),
        vram_gb: Number(t.vram_gb) || 0,
        ram_gb: Number(t.ram_gb) || 0,
        note: String(t.note || ""),
        vram_source: String(t.vram_source || "declared"),
        ram_source: String(t.ram_source || "declared"),
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
    settingsRows ? Promise.resolve({ targets: settingsRows, error: "" }) : readSettingsAndSuite(),
  ]);
  const byId = new Map();
  tool.targets.forEach((t) => byId.set(t.id, t));
  settings.targets.forEach((t) => byId.set(t.id, t));
  let list = [...byId.values()];
  if (!list.some((t) => t.id === "local")) {
    list = [...DEFAULT_TARGETS, ...list];
  }
  return { targets: list, errors: [tool.error, settings.error].filter(Boolean) };
}

// --- the /model-forge cockpit bridge -----------------------------------------

async function readCockpitBridge() {
  const sid = activeChatSessionId();
  if (!sid) {
    return {
      data: null,
      hint: "Open any chat session to sync exact GPU names, full storage inventory, and live fit verdicts (or use the guided controls here right away).",
    };
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

// --- the on-demand resource view (quiet by design) ---------------------------
// Collapsed by default; loads ONCE on expand (and on manual Refresh) — no
// timers, no polling, no auto-refresh.

async function readResourcesBridge() {
  const sid = activeChatSessionId();
  if (!sid) {
    return { data: null, hint: "The resource view needs an open chat session (the /model-forge command bridge); open a chat and expand again." };
  }
  try {
    const res = await host.request("slash.exec", { session_id: sid, command: "/model-forge resources" });
    if (res && res.error) {
      return { data: null, hint: "the /model-forge resource bridge answered with an error: " + (res.error.message || res.error) };
    }
    const raw = String((res && res.output) || "").trim();
    if (!raw) return { data: null, hint: "the /model-forge resource bridge returned no output" };
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") {
      return { data: null, hint: "the /model-forge resource bridge returned an unexpected payload" };
    }
    return { data, hint: "" };
  } catch (err) {
    return { data: null, hint: "the /model-forge resource bridge is unavailable: " + ((err && err.message) || err) };
  }
}

// --- engine probes (all 9 engine fingerprints) -------------------------------

async function probeRoot(root) {
  const tryFetch = async (path) => {
    try {
      const r = await fetch(root + path, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (r.ok) return await r.json();
    } catch (err) { /* offline, CORS-blocked, or timed out */ }
    return null;
  };
  const [props, tags, catalog, stats, version, modelInfo, nodeId, exoState, models] = await Promise.all([
    tryFetch("/props"),
    tryFetch("/api/tags"),
    tryFetch("/api/catalog"),
    tryFetch("/system_stats"),
    tryFetch("/version"),
    tryFetch("/get_model_info"),
    tryFetch("/node_id"),
    tryFetch("/state"),
    tryFetch("/v1/models"),
  ]);
  const engines = [];
  let suggestedVramGb = 0;
  let suggestedRamGb = 0;

  if (stats && (stats.system || stats.devices)) {
    const devices = Array.isArray(stats.devices) ? stats.devices : [];
    const totalVram = devices.reduce((sum, d) => sum + (Number(d && d.vram_total) || 0), 0);
    if (totalVram > 0) suggestedVramGb = Math.round((totalVram / (1024 ** 3)) * 10) / 10;
    const totalRam = Number((stats.system || {}).ram_total) || 0;
    if (totalRam > 0) suggestedRamGb = Math.round((totalRam / (1024 ** 3)) * 10) / 10;
    engines.push({
      id: "comfyui",
      name: "ComfyUI",
      ok: true,
      detail: (stats.system || {})["comfyui-version"] || "running",
      modalities: ["image", "image_edit", "video", "3d"],
      suggested_vram_gb: suggestedVramGb,
      suggested_ram_gb: suggestedRamGb,
    });
  }
  if (version && typeof version.version !== "undefined") {
    const served = (models && Array.isArray(models.data)) ? models.data.map((m) => m.id).filter(Boolean) : [];
    engines.push({
      id: "vllm",
      name: "vLLM",
      ok: true,
      detail: "vllm " + version.version + (served.length ? " · " + served.slice(0, 3).join(", ") : ""),
      models_loaded: served,
      modalities: ["chat", "vision", "ocr", "image", "image_edit"],
    });
  }
  if (modelInfo && typeof modelInfo === "object" && !Array.isArray(modelInfo)) {
    const model = modelInfo.model_path || modelInfo.model || "";
    engines.push({
      id: "sglang",
      name: "SGLang",
      ok: true,
      detail: model ? "model " + model : "running",
      models_loaded: model ? [model] : [],
      modalities: ["chat", "vision", "image", "image_edit"],
    });
  }
  if (props && (props.build_info || props.model_path)) {
    const model = props.model_path || props.model || "";
    engines.push({
      id: "llamacpp",
      name: "llama.cpp (llama-server)",
      ok: true,
      detail: model ? "serving " + model : "running",
      models_loaded: model ? [model] : [],
      modalities: ["chat", "vision"],
    });
  } else if (tags && Array.isArray(tags.models)) {
    if (catalog && Array.isArray(catalog.models)) {
      const names = catalog.models.map((m) => m.name || "").filter(Boolean);
      engines.push({
        id: "maya",
        name: "Project Maya",
        ok: true,
        detail: "catalog: " + names.slice(0, 3).join(", "),
        models_loaded: names,
        modalities: ["chat", "vision", "ocr"],
      });
    } else {
      const names = tags.models.map((m) => m.name || "").filter(Boolean);
      engines.push({
        id: "ollama",
        name: "Ollama",
        ok: true,
        detail: names.length ? "models: " + names.slice(0, 3).join(", ") : "running (no models pulled yet)",
        models: names.length,
        models_loaded: names,
        modalities: ["chat", "vision", "ocr"],
      });
    }
  }
  const isExo = (typeof nodeId === "string" && nodeId.trim().length > 0) ||
    (exoState && typeof exoState === "object" && !Array.isArray(exoState) &&
      "instances" in exoState && ("topology" in exoState || "nodes" in exoState || "deployments" in exoState));
  if (isExo) {
    const served = (models && Array.isArray(models.data)) ? models.data.map((m) => m.id).filter(Boolean) : [];
    engines.push({
      id: "exo",
      name: "Exo",
      ok: true,
      detail: served.length ? "models: " + served.slice(0, 3).join(", ") : "cluster running",
      models_loaded: served,
      modalities: ["chat", "vision", "ocr"],
    });
  }
  if (models && Array.isArray(models.data)) {
    const named = engines.some((e) => e.ok);
    if (!named) {
      const served = models.data.map((m) => m.id).filter(Boolean);
      engines.push({
        id: "openai-compatible",
        name: "OpenAI-compatible",
        ok: true,
        detail: served.slice(0, 6).join(", ") + " — engine identity unknown",
        models: served.length,
        models_loaded: served,
        modalities: ["chat", "vision", "image", "video"],
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
    roots.map(async (root) => ({ root, engines: await probeRoot(root) })),
  );
  const engines = [];
  perRoot.forEach(({ root, engines: es }) => {
    es.forEach((e) => engines.push({ ...e, root }));
  });
  return { target, roots, engines };
}

async function probeDeclared(declared) {
  return Promise.all(declared.map(async (row) => {
    const engines = await probeRoot(rootUrl(row.base_url));
    return { row, engines };
  }));
}

async function loadFullState() {
  const [settings, bridge] = await Promise.all([
    readSettingsAndSuite(),
    readCockpitBridge(),
  ]);
  const { targets, errors: targetErrs } = await readTargets(settings.targets);
  const [probed, declaredProbed] = await Promise.all([
    Promise.all(targets.map(probeTarget)),
    probeDeclared(settings.declared),
  ]);
  const declared = settings.declared.map((row, i) => ({
    ...row,
    engines: (declaredProbed[i] && declaredProbed[i].engines) || [],
  }));
  return {
    probed,
    declared,
    settingsTargets: settings.targets,
    downloadRoots: settings.downloadRoots,
    credentials: (bridge.data && bridge.data.pull_credentials) || settings.credentials,
    mediaSettings: settings.mediaSettings,
    suite: settings.suite,
    machineFacts: settings.machineFacts,
    bridge: bridge.data,
    bridgeHint: bridge.hint,
    errors: [settings.error, ...targetErrs].filter(Boolean),
  };
}

// --- shared UI primitives -----------------------------------------------------

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

function Card(props) {
  return jsx("div", {
    className: cn(
      "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/40 p-3.5",
      props.className,
    ),
    children: props.children,
  });
}

function EngineRow(props) {
  const e = props.engine;
  return jsx(Row, {
    children: [
      jsx(StatusDot, { key: "dot", tone: e.ok ? "good" : "muted" }),
      jsx("span", {
        key: "n",
        style: {
          color: e.ok ? "var(--ui-accent)" : "var(--ui-text-quaternary)",
          minWidth: "130px",
          fontWeight: e.ok ? 500 : 400,
        },
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
  return jsxs(Card, {
    className: "my-2",
    children: [
      jsx("div", {
        className: "mb-1 text-xs font-semibold text-(--ui-text-primary)",
        children: "Point Model Forge at your compute",
      }),
      jsx("div", {
        className: "mb-2 text-xs text-(--ui-text-secondary)",
        children: inv.reason,
      }),
      ...(inv.next_steps || []).map((step, i) =>
        jsx("div", {
          key: "s" + i,
          className: "py-0.5 pl-2.5 text-xs leading-relaxed text-(--ui-text-secondary)",
          children: (i + 1) + ". " + step,
        })),
      props.onAddGpu ? jsx("div", {
        className: "mt-3 flex flex-wrap gap-2",
        children: [
          jsx(Button, {
            key: "add",
            size: "sm",
            onClick: props.onAddGpu,
            children: "Add GPU or Server",
          }),
          props.onOpenEngines ? jsx(Button, {
            key: "eng",
            size: "sm",
            variant: "outline",
            onClick: props.onOpenEngines,
            children: "Set up an Engine",
          }) : null,
        ],
      }) : null,
    ],
  });
}

function TargetSection(props) {
  const { probed, bridge, machineFacts } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "GPU & compute targets" })];
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
  } else if (machineFacts && machineFacts.machine) {
    const m = machineFacts.machine;
    const gpuLabel = m.gpu_class ? String(m.gpu_class).toUpperCase() + " GPU" : "no discrete GPU reported";
    children.push(jsx("div", {
      key: "gpu",
      style: { color: "var(--ui-text-secondary)", fontSize: "12px", padding: "2px 0" },
      children: "This machine: " + gpuLabel +
        (m.ram_gb ? " · " + m.ram_gb + " GB RAM" : "") +
        (m.os_family ? " · " + m.os_family : ""),
    }));
  } else {
    children.push(jsx(Hint, {
      key: "gpu",
      children: "GPU hardware for this machine comes from the model_forge_status probe.",
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
    children: "Declared engine endpoints (user-declared):",
  }));
  if (!declared.length) {
    children.push(jsx(Hint, {
      key: "dn",
      children: "None declared. Connect any running engine HTTP API from the Engines tab.",
    }));
  }
  declared.forEach((d, i) => {
    children.push(jsx(Row, {
      key: "d-" + i,
      children: [
        jsx("span", { key: "n", style: { color: "var(--ui-accent)", minWidth: "130px" }, children: d.label }),
        jsx("span", { key: "u", style: { color: "var(--ui-text-secondary)", fontSize: "12px" }, children: d.base_url }),
        jsx("span", { key: "s", style: { color: "var(--ui-text-quaternary)", fontSize: "11px" }, children: d.source }),
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
        (bridgeHint || "Open the full Model Forge view or a chat session to sync."),
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
        jsx("span", { key: "n", style: { minWidth: "150px", color: "var(--ui-text-secondary)" }, children: m.name }),
        jsx("span", { key: "mod", style: { color: "var(--ui-text-quaternary)", fontSize: "11px", minWidth: "70px" }, children: m.modality_label || m.modality }),
        jsx("span", { key: "f", style: { color: "var(--ui-accent)", fontSize: "12px", minWidth: "110px" }, children: fit.verdict || m.status || "" }),
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

function MemoryBar(props) {
  const row = props.row;
  const label = String(row.kind || "");
  if (row.status !== "ok" || row.used_gb == null || row.total_gb == null) {
    return jsx("div", {
      style: {
        display: "flex", gap: "8px", padding: "2px 0",
        color: "var(--ui-text-quaternary)", fontSize: "11px",
      },
      children: [
        jsx("span", { key: "k", style: { minWidth: "42px" }, children: label }),
        jsx("span", { key: "d", children: "unavailable — " + (row.detail || "no reading") }),
      ],
    });
  }
  const pct = Math.max(0, Math.min(100, Number(row.used_pct) || 0));
  return jsx("div", {
    style: { display: "flex", alignItems: "center", gap: "8px", padding: "2px 0" },
    children: [
      jsx("span", { key: "k", style: { minWidth: "42px", color: "var(--ui-text-secondary)", fontSize: "11px" }, children: label }),
      jsx("div", {
        key: "track",
        style: {
          width: "110px", height: "8px", borderRadius: "4px",
          border: "1px solid var(--ui-stroke-secondary)", overflow: "hidden",
        },
        children: jsx("div", { style: { width: pct + "%", height: "100%", background: "var(--ui-accent)" } }),
      }),
      jsx("span", {
        key: "d",
        style: { color: "var(--ui-text-secondary)", fontSize: "11px" },
        children: row.used_gb + "/" + row.total_gb + " GB (" + Math.round(pct) + "%)" +
          (row.booked_now_gb != null ? " · booked_now " + row.booked_now_gb + " GB" : "") +
          (row.free_gb != null ? " · free " + row.free_gb + " GB" : ""),
      }),
    ],
  });
}

function ResourceView() {
  // Collapsed by default; the snapshot loads once on expand (and on the
  // manual Refresh click) — quiet by design.
  const [open, setOpen] = useState(false);
  const [state, setState] = useState({ phase: "idle", data: null, hint: "" });
  const load = () => {
    setState({ phase: "loading", data: null, hint: "" });
    readResourcesBridge().then((res) => {
      setState(res.data
        ? { phase: "ready", data: res.data, hint: "" }
        : { phase: "error", data: null, hint: res.hint });
    });
  };
  const toggle = () => {
    if (!open && state.phase === "idle") load();
    setOpen(!open);
  };
  const children = [
    jsx("button", {
      key: "toggle",
      type: "button",
      onClick: toggle,
      className: "flex items-center gap-1.5 py-1 text-xs font-semibold text-(--ui-text-secondary) hover:text-(--ui-text-primary)",
      children: [
        jsx(DisclosureCaret, { key: "c", open }),
        jsx("span", { key: "l", children: "Resource view (on-demand)" }),
      ],
    }),
  ];
  if (!open) {
    children.push(jsx(Hint, {
      key: "closed",
      children: "Memory use (VRAM/RAM) and the resident model blocks — loaded once when you open it, never a persistent display. In chat: /model-forge storage resources.",
    }));
    return jsx("div", { children });
  }
  if (state.phase === "loading") {
    children.push(jsx(Hint, { key: "l", children: "Loading the resource snapshot…" }));
  } else if (state.phase === "error") {
    children.push(jsx(Hint, {
      key: "e",
      children: "Resource snapshot unavailable: " + (state.hint || "no data") + " Nothing was changed — Refresh to retry.",
    }));
  } else if (state.data) {
    (state.data.targets || []).forEach((t) => {
      children.push(jsx("div", {
        key: "t-" + t.target,
        style: { margin: "6px 0 2px", fontWeight: 600, fontSize: "11px" },
        children: (t.label || t.target) + " · " + (t.kind || ""),
      }));
      (t.memory || []).forEach((row, i) => {
        children.push(jsx(MemoryBar, { key: "m-" + t.target + "-" + i, row }));
      });
      const residents = t.residents || [];
      children.push(jsx(Hint, { key: "rh-" + t.target, children: "resident (" + residents.length + "):" }));
      residents.forEach((r, i) => {
        const size = r.size_gb != null
          ? r.size_gb + " GB (" + r.size_label + ")"
          : String(r.size_label || "size unknown");
        children.push(jsx("div", {
          key: "r-" + t.target + "-" + i,
          style: { color: "var(--ui-text-secondary)", fontSize: "11px", padding: "1px 0 1px 10px" },
          children: r.model_id + " · size " + size + " · engine=" + r.engine + " · lane=" + r.lane,
        }));
      });
      (t.queued_loads || []).forEach((q, i) => {
        children.push(jsx("div", {
          key: "q-" + t.target + "-" + i,
          style: { color: "var(--ui-text-secondary)", fontSize: "11px", padding: "1px 0 1px 10px" },
          children: "queued load: " + q.model_id + " · " + q.state,
        }));
      });
      if (t.swap_stats) {
        children.push(jsx(Hint, {
          key: "ss-" + t.target,
          children: "swap stats (measured): count=" + t.swap_stats.count +
            " · last=" + t.swap_stats.last_total_seconds + "s" +
            " · average=" + t.swap_stats.average_total_seconds + "s",
        }));
      }
    });
    if (state.data.note) {
      children.push(jsx(Hint, { key: "n", children: state.data.note }));
    }
  }
  children.push(jsx("div", {
    key: "refresh",
    onClick: load,
    style: {
      cursor: "pointer", padding: "3px 0", fontSize: "11px",
      color: "var(--ui-accent)", width: "max-content",
    },
    children: "Refresh (manual — nothing polls)",
  }));
  return jsx("div", { children });
}

function StorageSection(props) {
  const { bridge, bridgeHint, onTriggerAction } = props;
  const children = [jsx(SectionTitle, { key: "t", children: "Installed model storage" })];
  const storage = bridge && bridge.storage;
  const entries = (storage && storage.entries) || [];
  if (!entries.length) {
    children.push(jsx(Hint, {
      key: "h",
      children: "The installed-models list (names, sizes, modification times) comes from model_forge_storage. " +
        (bridgeHint || "Open a chat session so this view can load it."),
    }));
    children.push(jsx(ResourceView, { key: "resource-view" }));
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
        jsx("span", { key: "n", style: { minWidth: "150px", color: "var(--ui-text-secondary)" }, children: e.name }),
        jsx("span", { key: "z", style: { color: "var(--ui-text-secondary)", fontSize: "12px", minWidth: "70px" }, children: fmtBytes(e.size_bytes) }),
        jsx("span", { key: "m", style: { color: "var(--ui-text-quaternary)", fontSize: "11px", minWidth: "150px" }, children: e.last_modified || "" }),
        jsx("span", { key: "r", style: { color: "var(--ui-text-quaternary)", fontSize: "11px", flex: 1 }, children: (e.root_label || e.root || "") + (e.type ? " · " + e.type : "") }),
        onTriggerAction ? jsx(Button, {
          key: "del",
          size: "xs",
          variant: "ghost",
          onClick: () => onTriggerAction(
            "Delete " + e.name,
            "Call model_forge_storage with action=\"delete\" and entry=" + JSON.stringify(e.id || e.path) + " so I can review and approve deleting this model.",
          ),
          children: "Delete…",
        }) : null,
      ],
    }));
  });
  children.push(jsx(Hint, {
    key: "del",
    children: bridge.delete_note ||
      "Read-only listing. Deletion routes through model_forge_storage action=delete where the human-approval gate shows the exact path and size first.",
  }));
  children.push(jsx(ResourceView, { key: "resource-view" }));
  return children;
}

function ResidencySection(props) {
  // B2: broker weight-residency rows (pulled / resident / booked / unbooked /
  // budget) — the same honest data model_forge_broker action=status reports,
  // read from the cockpit bridge (cockpit_data().residency). Read-only.
  const { bridge, bridgeHint } = props;
  const res = bridge && bridge.residency;
  const children = [jsx(SectionTitle, { key: "t", children: "Broker weight residency" })];
  if (!res || res.ok === false) {
    children.push(jsx(Hint, {
      key: "h",
      children: "Weight residency (pulled / resident / booked / budget) comes from model_forge_broker. " +
        (res && res.error ? res.error + " " : "") +
        (bridgeHint || "In chat: /model-forge broker."),
    }));
    return children;
  }
  const budget = res.budget || {};
  children.push(jsx(Hint, {
    key: "b",
    children: "budget: " + (budget.nominal_gb != null ? budget.nominal_gb + " GB nominal" : "undeclared") +
      " · booked_now " + (budget.booked_now_gb != null ? budget.booked_now_gb + " GB" : "?") +
      (budget.headroom_gb != null ? " · headroom " + budget.headroom_gb + " GB" : ""),
  }));
  const pulled = res.pulled || [];
  children.push(jsx(Hint, { key: "ph", children: "pulled (" + pulled.length + "):" }));
  pulled.forEach((r, i) => {
    children.push(jsx(Row, {
      key: "p-" + i,
      children: [
        jsx("span", { key: "n", style: { minWidth: "150px", color: "var(--ui-text-secondary)" }, children: r.model_id }),
        jsx("span", { key: "s", style: { color: "var(--ui-text-secondary)", fontSize: "12px", minWidth: "70px" }, children: r.size_bytes != null ? fmtBytes(r.size_bytes) : "size?" }),
        jsx("span", { key: "p", style: { color: "var(--ui-text-quaternary)", fontSize: "11px" }, children: r.path || "" }),
      ],
    }));
  });
  const resident = res.resident || [];
  children.push(jsx(Hint, { key: "rh", children: "resident (" + resident.length + "):" }));
  resident.forEach((r, i) => {
    children.push(jsx("div", {
      key: "r-" + i,
      style: { color: "var(--ui-text-secondary)", fontSize: "11px", padding: "1px 0 1px 10px" },
      children: r.model_id + " · engine=" + r.engine + " · lane=" + r.lane,
    }));
  });
  const booked = res.booked || [];
  children.push(jsx(Hint, { key: "bh", children: "booked (" + booked.length + ") — measured vs published:" }));
  booked.forEach((r, i) => {
    children.push(jsx("div", {
      key: "bk-" + i,
      style: { color: "var(--ui-text-secondary)", fontSize: "11px", padding: "1px 0 1px 10px" },
      children: r.model_id + " · " + (r.vram_gb != null ? r.vram_gb + " GB" : "?") + " · " + (r.label || r.source || ""),
    }));
  });
  const unbooked = res.unbooked || [];
  if (unbooked.length) {
    children.push(jsx(Hint, { key: "uh", children: "unbooked (" + unbooked.length + ") — run bench after first load:" }));
    unbooked.slice(0, 12).forEach((r, i) => {
      children.push(jsx("div", {
        key: "u-" + i,
        style: { color: "var(--ui-text-quaternary)", fontSize: "11px", padding: "1px 0 1px 10px" },
        children: r.model_id + " · " + (r.modality || ""),
      }));
    });
    if (unbooked.length > 12) {
      children.push(jsx(Hint, { key: "um", children: "…and " + (unbooked.length - 12) + " more (see /model-forge broker)" }));
    }
  }
  if (res.note) children.push(jsx(Hint, { key: "n", children: res.note }));
  return children;
}

// --- compact side pane (dockable cockpit) ------------------------------------

function Cockpit(props) {
  const ctx = props.ctx;
  const [state, setState] = useState(null);
  const load = () => {
    loadFullState().then(setState).catch((err) => {
      setState({
        probed: [], declared: [], settingsTargets: [], downloadRoots: [],
        credentials: { HF_TOKEN: false, MODELSCOPE_API_TOKEN: false },
        mediaSettings: { imageBaseUrl: "", imageDefaultModel: "", videoBaseUrl: "", videoDefaultModel: "" },
        suite: [], machineFacts: null, bridge: null, bridgeHint: "",
        errors: ["cockpit load failed: " + ((err && err.message) || err)],
      });
    });
  };
  useEffect(() => {
    load();
    refreshHooks.add(load);
    const disposer = ctx.onEvent("plugin.model-forge.settings.changed", load);
    return () => {
      refreshHooks.delete(load);
      disposer();
    };
  }, []);
  if (!state) {
    return jsx("div", {
      style: { color: "var(--ui-text-secondary)", padding: "10px", fontSize: "12px" },
      children: "Loading Model Forge…",
    });
  }
  const engineSection = EnginesSection(state);
  const anyEngine = engineSection.anyDetected;
  const anyTarget = state.probed.some((r) => r.target.id !== "local");
  const localHw = state.bridge && (state.bridge.targets || []).find((t) => t.id === "local");
  const anyGpu = Boolean(
    (localHw && localHw.hardware && ((localHw.hardware.gpus || []).length || localHw.hardware.total_vram_gb > 0)) ||
    (state.machineFacts && state.machineFacts.machine && state.machineFacts.machine.gpu_class),
  );
  const nothingDetected = !anyTarget && !anyEngine && !anyGpu;
  const invitation = (state.bridge && state.bridge.invitation) ||
    (nothingDetected ? PANE_INVITATION : null);

  const children = [
    jsxs("div", {
      key: "toolbar",
      className: "mb-2 flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-(--ui-stroke-tertiary)",
      children: [
        jsxs("div", {
          className: "flex items-center gap-1.5",
          children: [
            jsx(Badge, {
              variant: state.credentials && state.credentials.HF_TOKEN ? "success" : "muted",
              children: state.credentials && state.credentials.HF_TOKEN ? "HF Connected" : "HF Optional",
            }),
            jsx(Badge, {
              variant: anyEngine ? "default" : "muted",
              children: anyEngine ? "Engines Ready" : "No Engine Yet",
            }),
          ],
        }),
        jsxs("div", {
          className: "flex items-center gap-1.5",
          children: [
            jsx(Button, {
              size: "xs",
              variant: "default",
              onClick: () => host.navigate(MODEL_FORGE_ROUTE),
              children: "Open Full View",
            }),
            jsx(Button, {
              size: "xs",
              variant: "ghost",
              onClick: load,
              children: "Refresh",
            }),
          ],
        }),
      ],
    }),
  ];
  state.errors.forEach((msg, i) => {
    children.push(jsx("div", {
      key: "err-" + i,
      style: { color: "var(--ui-text-quaternary)", fontSize: "11px", padding: "2px 0" },
      children: msg,
    }));
  });
  children.push(...TargetSection(state));
  if (invitation) {
    children.push(jsx(Invitation, {
      key: "inv",
      invitation,
      onAddGpu: () => {
        host.navigate(MODEL_FORGE_ROUTE);
        if (openAddGpuHook) openAddGpuHook();
      },
      onOpenEngines: () => host.navigate(MODEL_FORGE_ROUTE),
    }));
  }
  children.push(...engineSection.children);
  children.push(...ModelsSection({ bridge: state.bridge, bridgeHint: state.bridgeHint }));
  children.push(...StorageSection({ bridge: state.bridge, bridgeHint: state.bridgeHint }));
  children.push(...ResidencySection({ bridge: state.bridge, bridgeHint: state.bridgeHint }));
  return jsx("div", {
    style: { padding: "8px", overflowY: "auto", height: "100%" },
    children,
  });
}

// --- Guided Dialogs (HF Token, Add/Edit GPU Target, Connect Engine, Run in Chat) ---

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
                  children: connected ? "Token saved (value hidden)" : "Not set (public downloads still work)",
                }),
              ],
            }),
            jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsx("label", {
                  className: "text-xs font-medium text-(--ui-text-secondary)",
                  children: connected ? "Replace Hugging Face token" : "Hugging Face access token (Read)",
                }),
                jsx(Input, {
                  type: "password",
                  value: token,
                  onChange: (e) => setToken(e.target.value),
                  placeholder: "hf_...",
                  disabled: busy,
                }),
                jsx("span", {
                  className: "text-[11px] text-(--ui-text-quaternary)",
                  children: "Create a free Read token at huggingface.co/settings/tokens.",
                }),
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
                connected ? jsx(Button, {
                  size: "sm",
                  variant: "destructive",
                  disabled: busy,
                  onClick: handleDisconnect,
                  children: "Remove Token",
                }) : null,
                jsx(Button, {
                  size: "sm",
                  variant: "ghost",
                  onClick: () => {
                    onOpenChange(false);
                    host.navigate("/settings?tab=plugins&agent=model-forge");
                  },
                  children: "ModelScope & Advanced…",
                }),
              ],
            }),
            jsxs("div", {
              className: "flex items-center gap-2",
              children: [
                jsx(Button, { size: "sm", variant: "outline", onClick: () => onOpenChange(false), children: "Cancel" }),
                jsx(Button, {
                  size: "sm",
                  disabled: busy || !token.trim(),
                  onClick: handleSave,
                  children: busy ? "Saving…" : "Save Token",
                }),
              ],
            }),
          ],
        }),
      ],
    }),
  });
}

function AddComputeDialog(props) {
  const { open, onOpenChange, initialTarget, existingTargets, existingEngines, defaultMode, onSaved } = props;
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
      setError("Enter a full http:// or https:// address, e.g. http://gpu-box.example.com:8080");
      return;
    }
    setTesting(true);
    try {
      const found = await probeRoot(url);
      const comfy = found.find((e) => e.suggested_vram_gb > 0);
      if (comfy) {
        if (!vramGb) {
          setVramGb(String(comfy.suggested_vram_gb));
          setVramSource("suggested");
        }
        if (!ramGb && comfy.suggested_ram_gb) {
          setRamGb(String(comfy.suggested_ram_gb));
        }
      }
      setProbeResult({ url, engines: found });
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
              children: "Connect another computer on your home/office network, a rented cloud GPU, or any running inference server. No JSON editing needed.",
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
                jsx(Input, {
                  value: label,
                  onChange: (e) => setLabel(e.target.value),
                  placeholder: mode === "cloud" ? "Cloud A100 Box" : mode === "engine" ? "Workstation Ollama" : "Gaming PC GPU",
                }),
              ],
            }),
            jsxs("div", {
              className: "grid gap-1.5",
              children: [
                jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: "Server address (URL)" }),
                jsxs("div", {
                  className: "flex gap-2",
                  children: [
                    jsx(Input, {
                      className: "flex-1",
                      value: baseUrl,
                      onChange: (e) => setBaseUrl(e.target.value),
                      placeholder: mode === "cloud" ? "https://gpu.example.com:8080" : "http://gpu-box.example.com:11434",
                    }),
                    jsx(Button, {
                      size: "sm",
                      variant: "outline",
                      disabled: testing || !baseUrl.trim(),
                      onClick: handleTest,
                      children: testing ? "Testing…" : "Test Connection",
                    }),
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
                    jsx("span", { className: "font-medium", children: e.name }),
                    jsx("span", { className: "text-(--ui-text-quaternary)", children: e.detail }),
                  ],
                })),
              ],
            }) : null,
            mode !== "engine" ? jsxs("div", {
              className: "grid gap-2",
              children: [
                jsxs("div", {
                  className: "grid gap-1.5",
                  children: [
                    jsxs("div", {
                      className: "flex items-center justify-between",
                      children: [
                        jsx("label", { className: "text-xs font-medium text-(--ui-text-secondary)", children: "GPU VRAM (GB) — used for model fit badges" }),
                        vramSource === "suggested" ? jsx(Badge, { variant: "success", size: "xs", children: "Suggested from engine" }) : null,
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
                        jsx(Input, {
                          type: "number",
                          value: vramGb,
                          onChange: (e) => { setVramGb(e.target.value); setVramSource("declared"); },
                          placeholder: "Custom VRAM (GB)",
                        }),
                        jsx(Input, {
                          type: "number",
                          value: ramGb,
                          onChange: (e) => setRamGb(e.target.value),
                          placeholder: "System RAM (GB, optional)",
                        }),
                      ],
                    }),
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
            jsx(Button, {
              size: "sm",
              disabled: saving || !baseUrl.trim(),
              onClick: handleSave,
              children: saving ? "Saving…" : "Save",
            }),
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
              children: "This action runs through Hermes' tool approval gate so you can watch live progress and confirm any config write or install.",
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
            jsx("div", {
              className: "text-[11px] text-(--ui-text-quaternary)",
              children: "Copy the prompt below and paste it into any Hermes chat — or open a fresh chat directly.",
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

// --- Full Native Workspace Page (/model-forge) -------------------------------

function ModelForgePage(props) {
  const ctx = props.ctx;
  const [state, setState] = useState(null);
  const [tab, setTab] = useState("models");
  const [selectedTarget, setSelectedTarget] = useState("local");
  const [modality, setModality] = useState("all");
  const [search, setSearch] = useState("");
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
    loadFullState().then((next) => {
      setState(next);
      setImgUrlDraft(next.mediaSettings.imageBaseUrl || "");
      setImgModelDraft(next.mediaSettings.imageDefaultModel || "");
      setVidUrlDraft(next.mediaSettings.videoBaseUrl || "");
      setVidModelDraft(next.mediaSettings.videoDefaultModel || "");
    }).catch((err) => {
      setState({
        probed: [], declared: [], settingsTargets: [], downloadRoots: [],
        credentials: { HF_TOKEN: false, MODELSCOPE_API_TOKEN: false },
        mediaSettings: { imageBaseUrl: "", imageDefaultModel: "", videoBaseUrl: "", videoDefaultModel: "" },
        suite: [], machineFacts: null, bridge: null, bridgeHint: "",
        errors: ["Model Forge load failed: " + ((err && err.message) || err)],
      });
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

  const triggerAgentAction = (title, prompt) => {
    const sid = activeChatSessionId();
    try {
      if (host.composer && typeof host.composer.submit === "function" && sid) {
        const sent = host.composer.submit(sid, prompt);
        if (sent) {
          host.notify({ kind: "info", message: title + " — sent to your active chat." });
          return;
        }
      }
    } catch (err) { /* fall through to prompt dialog */ }
    if (ctx && ctx.os && typeof ctx.os.writeClipboard === "function") {
      ctx.os.writeClipboard(prompt).catch(() => null);
    }
    setPendingAction({ title, prompt });
  };

  const runningEngines = useMemo(() => {
    if (!state) return [];
    const rows = [];
    (state.probed || []).forEach((r) => {
      (r.engines || []).filter((e) => e.ok).forEach((e) => {
        rows.push({ ...e, targetId: r.target.id, targetLabel: r.target.label, source: "auto-detected" });
      });
    });
    (state.declared || []).forEach((d) => {
      (d.engines || []).filter((e) => e.ok).forEach((e) => {
        rows.push({ ...e, root: d.base_url, targetId: "declared", targetLabel: d.label, source: "user-declared" });
      });
    });
    return rows;
  }, [state]);

  const modelRows = useMemo(() => {
    const raw = (state && state.bridge && Array.isArray(state.bridge.models) && state.bridge.models.length)
      ? state.bridge.models
      : PREVIEW_MODELS;
    const selectedProbe = (state && state.probed || []).find((r) => r.target.id === selectedTarget) || null;
    const priced = selectedTarget !== "local" && selectedProbe
      ? raw.map((m) => ({ ...m, fit: advisoryFitForTarget(m, selectedProbe) }))
      : raw;
    const q = search.trim().toLowerCase();
    return priced.filter((m) => {
      if (modality !== "all" && m.modality !== modality) return false;
      if (!q) return true;
      const blob = [m.id, m.name, m.modality, m.modality_label, m.note, ...(m.engines || [])].join(" ").toLowerCase();
      return blob.includes(q);
    });
  }, [state, selectedTarget, modality, search]);

  if (!state) {
    return jsx("div", {
      className: "flex h-full items-center justify-center text-sm text-(--ui-text-secondary)",
      children: "Detecting your GPU and local inference engines…",
    });
  }

  const suiteReadyCount = (state.suite || []).filter((p) => p.installed && p.enabled).length;
  const localBridgeTarget = (state.bridge && state.bridge.targets || []).find((t) => t.id === "local");
  const localHw = localBridgeTarget && localBridgeTarget.hardware;
  const localGpuSummary = localHw
    ? ((localHw.gpus || []).map((g) => (g.name || "GPU") + (g.vram_gb ? " (" + g.vram_gb + " GB)" : "")).join(", ") || "No local GPU")
    : (state.machineFacts && state.machineFacts.machine && state.machineFacts.machine.gpu_class
      ? String(state.machineFacts.machine.gpu_class).toUpperCase() + " GPU · " + (state.machineFacts.machine.ram_gb || "?") + " GB RAM"
      : "No local GPU detected");
  const remoteCount = (state.probed || []).filter((r) => r.target.id !== "local").length;
  const hfSet = Boolean(state.credentials && state.credentials.HF_TOKEN);
  const activeWiring = (state.bridge && state.bridge.active_wiring) || null;
  const runningEngineIds = new Set(runningEngines.map((e) => e.id).filter(Boolean));
  const targetOptions = (state.probed || []).map((r) => ({ id: r.target.id, label: r.target.label }));

  const handleConfirmRemove = async () => {
    if (!removeConfirm) return;
    if (removeConfirm.kind === "target") {
      const next = (state.settingsTargets || []).filter((t) => t.id !== removeConfirm.id);
      await savePluginSettings("model-forge", { targets_json: next });
      host.notify({ kind: "info", message: "Removed target: " + removeConfirm.label });
    } else if (removeConfirm.kind === "engine") {
      const next = (state.declared || [])
        .filter((d) => rootUrl(d.base_url) !== rootUrl(removeConfirm.base_url))
        .map((d) => ({ label: d.label, base_url: d.base_url }));
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
        await savePluginSettings("model-forge-image", {
          base_url: imgUrlDraft.trim(),
          default_model: imgModelDraft.trim(),
        });
        host.notify({ kind: "success", message: "Saved Model Forge Image endpoint." });
      } else {
        await savePluginSettings("model-forge-video", {
          base_url: vidUrlDraft.trim(),
          default_model: vidModelDraft.trim(),
        });
        host.notify({ kind: "success", message: "Saved Model Forge Video endpoint." });
      }
      load();
    } catch (err) {
      host.notifyError(err, "Could not save media backend settings.");
    } finally {
      setSavingMedia(false);
    }
  };

  const renderModelsTab = () => {
    const isPreviewOnly = !(state.bridge && state.bridge.models && state.bridge.models.length);
    const liveModels = [];
    const seenLive = new Set();
    runningEngines.forEach((eng) => {
      (eng.models_loaded || []).forEach((mid) => {
        const key = eng.targetId + "::" + eng.id + "::" + mid;
        if (mid && !seenLive.has(key)) {
          seenLive.add(key);
          liveModels.push({
            modelId: mid,
            engineId: eng.id,
            engineName: eng.name,
            targetId: eng.targetId === "declared" ? "local" : eng.targetId,
            targetLabel: eng.targetLabel || eng.targetId,
          });
        }
      });
    });
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        isPreviewOnly ? jsxs(Card, {
          className: "flex flex-wrap items-center justify-between gap-2",
          children: [
            jsxs("div", {
              className: "text-xs text-(--ui-text-secondary)",
              children: [
                jsx("span", { className: "font-medium text-(--ui-text-primary)", children: "Catalog preview mode. " }),
                "Open any Hermes chat to sync live GPU fit badges, installed weights, and one-click chat pulls.",
              ],
            }),
            jsx(Button, {
              size: "xs",
              onClick: () => { if (typeof host.newChat === "function") host.newChat(); else host.navigate("/"); },
              children: "Open Chat to Sync",
            }),
          ],
        }) : null,

        // Live models already detected on running engines — 1-click "Use in Hermes"
        liveModels.length > 0 ? jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between gap-2",
              children: [
                jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Ready to use right now (" + liveModels.length + " detected on your engines)" }),
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
                      "Call model_forge_activate with model_id=" + JSON.stringify(lm.modelId) + ", target=" + JSON.stringify(lm.targetId) + ", and apply=true so I can approve piping it into Hermes.",
                    ),
                    children: "Use in Hermes",
                  }),
                ],
              })),
            }),
          ],
        }) : null,

        // Pull ANY Hugging Face or Ollama model bar
        jsxs(Card, {
          className: "grid gap-2.5",
          children: [
            jsxs("div", {
              className: "flex flex-wrap items-center justify-between gap-2",
              children: [
                jsxs("div", {
                  children: [
                    jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Pull any model (Hugging Face or Ollama)" }),
                    jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Paste any Hugging Face repo (e.g. unsloth/Qwen3-8B-GGUF) or Ollama tag (e.g. qwen3:8b). Model Forge vets the weights and picks the right engine." }),
                  ],
                }),
                jsx(Button, {
                  size: "xs",
                  variant: "ghost",
                  onClick: () => triggerAgentAction(
                    "Search Hugging Face",
                    "Call model_forge_browse with query=" + JSON.stringify(customModelId.trim() || search.trim() || "qwen") + " and target=" + JSON.stringify(selectedTarget) + " to show matching Hugging Face models with fit verdicts.",
                  ),
                  children: "Search Hugging Face…",
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
                  placeholder: "owner/repo or Ollama tag (e.g. qwen3:8b, black-forest-labs/FLUX.2-klein-4B)",
                }),
                jsx(SegmentedControl, {
                  value: customEngine,
                  onChange: setCustomEngine,
                  sizing: "content",
                  options: [
                    { id: "auto", label: "Auto engine" },
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
                      "Call model_forge_install with model_id=" + JSON.stringify(mid) + ", target=" + JSON.stringify(selectedTarget) + engPart + " and report the download progress.",
                    );
                  },
                  children: "Pull Model",
                }),
              ],
            }),
          ],
        }),

        // Filter bar: modality chips + search
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
            jsx(SearchField, {
              variant: "box",
              containerClassName: "w-56",
              placeholder: "Filter models…",
              value: search,
              onChange: setSearch,
            }),
          ],
        }),

        // Model cards grid
        modelRows.length === 0
          ? jsx(EmptyState, {
              title: "No models match this filter",
              description: "Try clearing your search or paste any Hugging Face repo in the box above to pull it directly.",
            })
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
                        (fit.advisory || m.note) ? jsx("div", {
                          className: "text-xs leading-relaxed text-(--ui-text-secondary)",
                          children: fit.advisory || m.note,
                        }) : null,
                      ],
                    }),
                    jsxs("div", {
                      className: "flex flex-wrap items-center gap-1.5 pt-1",
                      children: [
                        isCoreGguf
                          ? jsx(Button, {
                              size: "xs",
                              onClick: () => host.navigate("/settings?tab=providers"),
                              children: "Open Hermes Local Models",
                            })
                          : jsx(Button, {
                              size: "xs",
                              variant: matchedRunning ? "default" : "secondary",
                              disabled: m.status === "stub",
                              onClick: () => triggerAgentAction(
                                "Pull " + m.name,
                                "Call model_forge_install with model_id=" + JSON.stringify(m.id) + " and target=" + JSON.stringify(selectedTarget) + ".",
                              ),
                              children: "Pull Model",
                            }),
                        canActivate ? jsx(Button, {
                          size: "xs",
                          variant: "outline",
                          onClick: () => triggerAgentAction(
                            "Wire " + m.name + " into Hermes",
                            "Call model_forge_activate with model_id=" + JSON.stringify(m.id) + ", target=" + JSON.stringify(selectedTarget) + ", and apply=true so I can approve wiring it natively into Hermes.",
                          ),
                          children: "Use in Hermes",
                        }) : null,
                      ],
                    }),
                  ],
                });
              }),
            }),
        jsx(Hint, {
          children: "Fit badges are advisory estimates from published footprints; each inference engine does authoritative memory sizing at launch.",
        }),
      ],
    });
  };

  const renderComputeTab = () => {
    const remoteRows = (state.probed || []).filter((r) => r.target.id !== "local");
    return jsxs("div", {
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
            state.machineFacts && state.machineFacts.machine ? jsxs("div", {
              className: "flex flex-wrap gap-3 text-[11px] text-(--ui-text-tertiary)",
              children: [
                state.machineFacts.machine.cpu_model ? jsx("span", { children: "CPU: " + state.machineFacts.machine.cpu_model }) : null,
                state.machineFacts.machine.ram_gb ? jsx("span", { children: "RAM: " + state.machineFacts.machine.ram_gb + " GB" }) : null,
                state.machineFacts.machine.os_family ? jsx("span", { children: "OS: " + state.machineFacts.machine.os_family + " " + (state.machineFacts.machine.os_release || "") }) : null,
              ],
            }) : null,
          ],
        }),

        jsxs("div", {
          className: "flex items-center justify-between",
          children: [
            jsx("div", { className: "text-xs font-semibold uppercase tracking-wide text-(--ui-text-secondary)", children: "Network & Cloud GPUs (" + remoteRows.length + ")" }),
            jsx(Button, {
              size: "sm",
              onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); },
              children: "Add GPU / Cloud Server",
            }),
          ],
        }),

        remoteRows.length === 0
          ? jsxs(Card, {
              className: "text-center py-6",
              children: [
                jsx("div", { className: "text-sm font-medium text-(--ui-text-primary)", children: "No remote or cloud GPUs added yet" }),
                jsx("div", { className: "mt-1 text-xs text-(--ui-text-secondary)", children: "Have a gaming PC on your Wi-Fi or a rented cloud GPU? Point Model Forge at it in 10 seconds." }),
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
              children: remoteRows.map((r) => {
                const t = r.target;
                const live = (r.engines || []).filter((e) => e.ok);
                const editable = t.source === "settings";
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
                                jsx("span", { className: "text-sm font-semibold text-(--ui-text-primary)", children: t.label }),
                                jsx(Badge, { variant: "outline", size: "xs", children: t.kind }),
                                t.vram_gb ? jsx(Badge, { variant: "default", size: "xs", children: t.vram_gb + " GB VRAM" }) : null,
                                t.ram_gb ? jsx(Badge, { variant: "muted", size: "xs", children: t.ram_gb + " GB RAM" }) : null,
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
                              onClick: () => { setEditingTarget(t); setAddGpuMode(t.kind || "lan"); setAddGpuOpen(true); },
                              children: editable ? "Edit" : "Override",
                            }),
                            editable ? jsx(Button, {
                              size: "xs",
                              variant: "ghost",
                              onClick: () => setRemoveConfirm({ kind: "target", id: t.id, label: t.label }),
                              children: "Remove",
                            }) : jsx(Button, {
                              size: "xs",
                              variant: "ghost",
                              onClick: () => triggerAgentAction(
                                "Remove target " + t.label,
                                "Call model_forge_target_add with action=\"remove\" and id=" + JSON.stringify(t.id) + ".",
                              ),
                              children: "Remove…",
                            }),
                          ],
                        }),
                      ],
                    }),
                    live.length
                      ? jsx("div", {
                          className: "flex flex-wrap gap-1.5",
                          children: live.map((e, i) => jsx(Badge, { key: i, variant: "success", size: "xs", children: e.name + (e.detail ? " · " + e.detail : "") })),
                        })
                      : jsx(Hint, { children: "No engine answered at " + t.base_url + " during the last check." }),
                  ],
                });
              }),
            }),
      ],
    });
  };

  const renderEnginesTab = () => {
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        jsxs("div", {
          className: "flex flex-wrap items-center justify-between gap-2",
          children: [
            jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: "Every engine works two ways: install it (or open Hermes Local Models) or connect an already-running server URL." }),
            jsx(Button, {
              size: "sm",
              onClick: () => { setEditingTarget(null); setAddGpuMode("engine"); setAddGpuOpen(true); },
              children: "Connect Engine URL",
            }),
          ],
        }),
        jsx("div", {
          className: "grid gap-2.5 md:grid-cols-2",
          children: ENGINE_DIRECTORY.map((eng) => {
            const liveRows = runningEngines.filter((r) => r.id === eng.id);
            const live = liveRows[0] || null;
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
                            jsx(StatusDot, { tone: live ? "good" : "muted" }),
                            jsx("span", { className: "text-sm font-semibold text-(--ui-text-primary)", children: eng.name }),
                          ],
                        }),
                        jsx(Badge, { variant: live ? "success" : "muted", size: "xs", children: live ? "Running" : "Not detected" }),
                      ],
                    }),
                    jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: eng.summary }),
                    jsx("div", { className: "text-[11px] text-(--ui-text-quaternary)", children: "Modalities: " + eng.modalities.join(", ") + " · default port :" + eng.defaultPort }),
                    live ? jsx("div", {
                      className: "text-[11px] text-(--ui-accent)",
                      children: "Active at " + (live.root || "") + (live.detail ? " — " + live.detail : ""),
                    }) : null,
                  ],
                }),
                jsxs("div", {
                  className: "flex flex-wrap items-center gap-1.5",
                  children: [
                    eng.installMode === "core_managed"
                      ? jsx(Button, { size: "xs", onClick: () => host.navigate("/settings?tab=providers"), children: "Open Local Models" })
                      : eng.installMode === "approved_installer"
                        ? jsx(Button, {
                            size: "xs",
                            variant: live ? "secondary" : "default",
                            onClick: () => triggerAgentAction(
                              "Install " + eng.name,
                              "Call model_forge_install with action=\"engine_install\", engine=" + JSON.stringify(eng.id) + ", and execute=true so I can inspect the pinned vendor installer command and approve it.",
                            ),
                            children: live ? "Reinstall / Update…" : "Install " + eng.name,
                          })
                        : jsx(Button, {
                            size: "xs",
                            variant: "secondary",
                            onClick: () => setEngineCmdDialog(eng),
                            children: "Quick-Start Command",
                          }),
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

        // Declared engine endpoints list
        (state.declared || []).length > 0 ? jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsx("div", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Connected Engine URLs (" + state.declared.length + ")" }),
            ...state.declared.map((d, i) => jsxs("div", {
              key: "dec-" + i,
              className: "flex flex-wrap items-center justify-between gap-2 border-t border-(--ui-stroke-tertiary) pt-2",
              children: [
                jsxs("div", {
                  children: [
                    jsxs("div", {
                      className: "flex items-center gap-2 text-xs font-medium text-(--ui-text-primary)",
                      children: [
                        jsx(StatusDot, { tone: (d.engines || []).length ? "good" : "muted" }),
                        d.label,
                        jsx("span", { className: "font-mono text-[11px] font-normal text-(--ui-text-quaternary)", children: d.base_url }),
                      ],
                    }),
                    jsx("div", {
                      className: "text-[11px] text-(--ui-text-tertiary)",
                      children: (d.engines || []).length
                        ? d.engines.map((e) => e.name + ": " + e.detail).join(" · ")
                        : "Not answering right now",
                    }),
                  ],
                }),
                jsx(Button, {
                  size: "xs",
                  variant: "ghost",
                  onClick: () => setRemoveConfirm({ kind: "engine", label: d.label, base_url: d.base_url }),
                  children: "Remove",
                }),
              ],
            })),
          ],
        }) : null,
      ],
    });
  };

  const renderWiringTab = () => {
    const mediaCandidates = runningEngines.filter((e) =>
      (e.id === "comfyui" || e.id === "vllm" || e.id === "sglang" || e.id === "openai-compatible") &&
      (e.modalities || []).some((m) => m === "image" || m === "video") && e.root,
    );
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        // 1. Model Forge Suite checklist
        jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex flex-wrap items-center justify-between gap-2",
              children: [
                jsxs("div", {
                  children: [
                    jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Model Forge Plugin Suite (" + suiteReadyCount + "/" + SUITE_PACKAGES.length + " active)" }),
                    jsx("div", { className: "text-xs text-(--ui-text-secondary)", children: "Install and enable all 5 suite packages from the Plugin Store for full chat, vision, image, and video piping." }),
                  ],
                }),
                jsx(Button, {
                  size: "sm",
                  variant: "outline",
                  onClick: () => host.navigate("/capabilities?tab=plugins"),
                  children: "Open Plugin Store",
                }),
              ],
            }),
            jsx("div", {
              className: "grid gap-1 pt-1",
              children: (state.suite || []).map((pkg) => jsxs("div", {
                key: pkg.name,
                className: "flex items-center justify-between gap-2 rounded border border-(--ui-stroke-tertiary) px-2.5 py-1.5 text-xs",
                children: [
                  jsxs("div", {
                    className: "flex items-center gap-2 min-w-0",
                    children: [
                      jsx(StatusDot, { tone: pkg.installed && pkg.enabled ? "good" : pkg.installed ? "warn" : "muted" }),
                      jsx("span", { className: "font-medium text-(--ui-text-primary)", children: pkg.label }),
                      jsx("span", { className: "truncate text-[11px] text-(--ui-text-tertiary)", children: pkg.role }),
                    ],
                  }),
                  jsx(Badge, {
                    variant: pkg.installed && pkg.enabled ? "success" : pkg.installed ? "warn" : "muted",
                    size: "xs",
                    children: pkg.installed && pkg.enabled ? (pkg.version ? "v" + pkg.version : "Active") : pkg.installed ? "Disabled" : "Not installed",
                  }),
                ],
              })),
            }),
          ],
        }),

        // 2. Pull credentials
        jsxs(Card, {
          className: "grid gap-1",
          children: [
            jsx("div", { className: "mb-1 text-sm font-semibold text-(--ui-text-primary)", children: "Model Hub Credentials" }),
            jsx(ListRow, {
              title: "Hugging Face Token",
              description: hfSet ? "Saved in Hermes credentials (sent only to huggingface.co)." : "Optional. Add a Read token for gated repos and faster downloads.",
              action: jsxs("div", {
                className: "flex items-center gap-2",
                children: [
                  jsx(Badge, { variant: hfSet ? "success" : "muted", children: hfSet ? "Connected" : "Not set" }),
                  jsx(Button, { size: "sm", variant: hfSet ? "outline" : "default", onClick: () => setHfOpen(true), children: hfSet ? "Manage" : "Add Token" }),
                ],
              }),
            }),
            jsx(ListRow, {
              title: "ModelScope Token & Download Folders",
              description: "Configure optional ModelScope credentials or extra download roots in Hermes Plugin Settings.",
              action: jsx(Button, {
                size: "sm",
                variant: "outline",
                onClick: () => host.navigate("/settings?tab=plugins&agent=model-forge"),
                children: "Open Plugin Settings",
              }),
            }),
          ],
        }),

        // 3. Active Chat & /model Wiring
        jsxs(Card, {
          className: "grid gap-2",
          children: [
            jsxs("div", {
              className: "flex items-center justify-between gap-2",
              children: [
                jsx("div", { className: "text-sm font-semibold text-(--ui-text-primary)", children: "Chat & /model Piping" }),
                jsx(Button, { size: "xs", variant: "outline", onClick: () => host.navigate("/settings?tab=providers"), children: "Providers Settings" }),
              ],
            }),
            activeWiring && activeWiring.ok ? jsxs("div", {
              className: "grid gap-1 text-xs text-(--ui-text-secondary)",
              children: [
                jsx("div", { children: "Active default chat model: " + (activeWiring.chat.model || "not set") + (activeWiring.chat.provider ? " (" + activeWiring.chat.provider + ")" : "") }),
                activeWiring.vision && activeWiring.vision.model ? jsx("div", { children: "Auxiliary vision lane: " + activeWiring.vision.model + " @ " + activeWiring.vision.base_url }) : null,
                (activeWiring.custom_providers || []).length ? jsx("div", {
                  className: "text-[11px] text-(--ui-text-tertiary)",
                  children: "Registered /model endpoints: " + activeWiring.custom_providers.map((p) => p.name + " (" + (p.models || []).length + ")").join(", "),
                }) : null,
              ],
            }) : jsx(Hint, { children: "Click 'Use in Hermes' on any model in the Models tab to pipe it straight into Hermes' native /model switcher." }),
          ],
        }),

        // 4. Image & Video Backend Piping (direct GUI save via plugins.manage settings!)
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
                jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Routes Hermes' native image_generate tool to either a raw ComfyUI URL (e.g. http://127.0.0.1:8188 — direct workflows) or an OpenAI-compatible /v1 images API." }),
                jsx(Input, { value: imgUrlDraft, onChange: (e) => setImgUrlDraft(e.target.value), placeholder: "http://127.0.0.1:8188 or http://gpu-box.example.com:8080/v1" }),
                jsx(Input, { value: imgModelDraft, onChange: (e) => setImgModelDraft(e.target.value), placeholder: "Default model or workflow id (blank = auto)" }),
                mediaCandidates.length > 0 ? jsxs("div", {
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
                      onClick: () => triggerAgentAction(
                        "Set Hermes Image Provider",
                        "Call model_forge_activate with model_id=" + JSON.stringify(imgModelDraft.trim() || "qwen-image-2.1") + ", surface=\"image\", target=" + JSON.stringify(selectedTarget) + ", and apply=true.",
                      ),
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
                jsx("div", { className: "text-[11px] text-(--ui-text-tertiary)", children: "Routes Hermes' native video_generate tool to either a raw ComfyUI URL (e.g. http://127.0.0.1:8188 — direct workflows) or an OpenAI-compatible /v1 video API." }),
                jsx(Input, { value: vidUrlDraft, onChange: (e) => setVidUrlDraft(e.target.value), placeholder: "http://127.0.0.1:8188 or http://gpu-box.example.com:8080/v1" }),
                jsx(Input, { value: vidModelDraft, onChange: (e) => setVidModelDraft(e.target.value), placeholder: "Default model or workflow id (blank = auto)" }),
                mediaCandidates.length > 0 ? jsxs("div", {
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
                      onClick: () => triggerAgentAction(
                        "Set Hermes Video Provider",
                        "Call model_forge_activate with model_id=" + JSON.stringify(vidModelDraft.trim() || "ltx-2.5") + ", surface=\"video\", target=" + JSON.stringify(selectedTarget) + ", and apply=true.",
                      ),
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
    return jsxs("div", {
      className: "grid gap-4",
      children: [
        jsx(Card, {
          children: StorageSection({
            bridge: state.bridge,
            bridgeHint: state.bridgeHint,
            onTriggerAction: triggerAgentAction,
          }),
        }),
        jsx(Card, {
          children: ResidencySection({
            bridge: state.bridge,
            bridgeHint: state.bridgeHint,
          }),
        }),
      ],
    });
  };

  return jsxs("div", {
    className: "mx-auto flex h-full max-w-5xl flex-col gap-4 overflow-y-auto px-6 py-5",
    children: [
      // Page header
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
                  jsx(Badge, { variant: "muted", size: "xs", children: "Local AI" }),
                ],
              }),
              jsx("p", {
                className: "mt-0.5 text-xs text-(--ui-text-secondary)",
                children: "Find your GPU, pull any model across any modality, and pipe it natively into Hermes — no cloud account required.",
              }),
            ],
          }),
          jsxs("div", {
            className: "flex flex-wrap items-center gap-2",
            children: [
              targetOptions.length > 1 ? jsx(SegmentedControl, {
                value: selectedTarget,
                onChange: setSelectedTarget,
                sizing: "content",
                options: targetOptions,
              }) : null,
              jsx(Button, {
                size: "sm",
                variant: "outline",
                onClick: () => { setEditingTarget(null); setAddGpuMode("lan"); setAddGpuOpen(true); },
                children: "Add GPU / Server",
              }),
              jsx(Button, {
                size: "sm",
                variant: "ghost",
                onClick: load,
                children: "Refresh",
              }),
            ],
          }),
        ],
      }),

      // 4-Step Quick Readiness Strip
      jsxs("div", {
        className: "grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4",
        children: [
          jsxs("button", {
            type: "button",
            onClick: () => setTab("wiring"),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left transition-colors hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", {
                className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)",
                children: [
                  "1. Plugin Suite",
                  jsx(StatusDot, { tone: suiteReadyCount === SUITE_PACKAGES.length ? "good" : "warn" }),
                ],
              }),
              jsx("div", {
                className: "mt-1 text-[11px] text-(--ui-text-secondary)",
                children: suiteReadyCount + " of " + SUITE_PACKAGES.length + " packages active",
              }),
            ],
          }),
          jsxs("button", {
            type: "button",
            onClick: () => setTab("compute"),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left transition-colors hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", {
                className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)",
                children: [
                  "2. GPU & Compute",
                  jsx(StatusDot, { tone: (localHw && localHw.total_vram_gb > 0) || remoteCount > 0 ? "good" : "warn" }),
                ],
              }),
              jsx("div", {
                className: "mt-1 truncate text-[11px] text-(--ui-text-secondary)",
                children: remoteCount > 0 ? localGpuSummary + " + " + remoteCount + " remote" : localGpuSummary,
              }),
            ],
          }),
          jsxs("button", {
            type: "button",
            onClick: () => setHfOpen(true),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left transition-colors hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", {
                className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)",
                children: [
                  "3. Hugging Face",
                  jsx(StatusDot, { tone: hfSet ? "good" : "muted" }),
                ],
              }),
              jsx("div", {
                className: "mt-1 text-[11px] text-(--ui-text-secondary)",
                children: hfSet ? "Token connected" : "Add token for gated models",
              }),
            ],
          }),
          jsxs("button", {
            type: "button",
            onClick: () => setTab("engines"),
            className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/50 p-3 text-left transition-colors hover:bg-(--chrome-action-hover)",
            children: [
              jsxs("div", {
                className: "flex items-center justify-between text-xs font-medium text-(--ui-text-primary)",
                children: [
                  "4. Inference Engines",
                  jsx(StatusDot, { tone: runningEngines.length > 0 ? "good" : "warn" }),
                ],
              }),
              jsx("div", {
                className: "mt-1 text-[11px] text-(--ui-text-secondary)",
                children: runningEngines.length > 0
                  ? runningEngines.length + " running (" + runningEngines.slice(0, 2).map((e) => e.name).join(", ") + ")"
                  : "Click to install or connect",
              }),
            ],
          }),
        ],
      }),

      // Errors (honest, never swallowed)
      ...(state.errors || []).map((msg, i) => jsx("div", {
        key: "perr-" + i,
        className: "rounded border border-(--ui-stroke-secondary) px-3 py-1.5 text-xs text-(--ui-text-secondary)",
        children: msg,
      })),

      // Navigation tabs
      jsx("div", {
        className: "flex flex-wrap items-center justify-between gap-2",
        children: jsx(SegmentedControl, {
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
      }),

      jsx(Separator, {}),

      tab === "models" ? renderModelsTab() : null,
      tab === "compute" ? renderComputeTab() : null,
      tab === "engines" ? renderEnginesTab() : null,
      tab === "wiring" ? renderWiringTab() : null,
      tab === "storage" ? renderStorageTab() : null,

      // Dialogs
      jsx(HfCredentialsDialog, {
        open: hfOpen,
        onOpenChange: setHfOpen,
        connected: hfSet,
        onSaved: load,
      }),
      jsx(AddComputeDialog, {
        open: addGpuOpen,
        onOpenChange: setAddGpuOpen,
        initialTarget: editingTarget,
        existingTargets: state.settingsTargets,
        existingEngines: state.declared,
        defaultMode: addGpuMode,
        onSaved: load,
      }),
      jsx(ActionPromptDialog, {
        action: pendingAction,
        onClose: () => setPendingAction(null),
      }),
      jsx(ConfirmDialog, {
        open: Boolean(removeConfirm),
        onClose: () => setRemoveConfirm(null),
        onConfirm: handleConfirmRemove,
        destructive: true,
        confirmLabel: "Remove",
        title: removeConfirm ? "Remove " + removeConfirm.label + "?" : "Remove?",
        description: "This only removes the saved address from Model Forge settings. Nothing on the remote machine is touched or deleted.",
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
                jsx("div", {
                  className: "rounded border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-2.5 font-mono text-xs text-(--ui-text-primary) break-all",
                  children: engineCmdDialog.quickStart,
                }),
                jsx("div", {
                  className: "text-[11px] text-(--ui-text-tertiary)",
                  children: "Once it starts on port :" + engineCmdDialog.defaultPort + ", click Refresh and Model Forge detects it automatically.",
                }),
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

// --- Settings -> Plugins -> Model Forge pages --------------------------------

function SettingsOverviewPage() {
  const [state, setState] = useState(null);
  const [hfOpen, setHfOpen] = useState(false);
  const load = () => { readSettingsAndSuite().then(setState).catch(() => null); };
  useEffect(() => { load(); }, []);
  const hfSet = Boolean(state && state.credentials && state.credentials.HF_TOKEN);
  return jsxs("div", {
    className: "grid gap-3",
    children: [
      jsx(ListRow, {
        title: "Open Model Forge Workspace",
        description: "Browse models by modality, test GPUs, manage engines, and wire local models into Hermes.",
        action: jsx(Button, { size: "sm", onClick: () => host.navigate(MODEL_FORGE_ROUTE), children: "Open Model Forge" }),
      }),
      jsx(ListRow, {
        title: "Hugging Face Token",
        description: hfSet ? "Connected (stored in Hermes credentials; sent only to huggingface.co)." : "Add a Read token to unlock gated models and full-speed downloads.",
        action: jsx(Button, { size: "sm", variant: hfSet ? "outline" : "default", onClick: () => setHfOpen(true), children: hfSet ? "Manage Token" : "Add HF Token" }),
      }),
      jsx(ListRow, {
        title: "Model Forge Plugin Suite",
        description: "Manage all 5 companion packages (Core, Desktop, Providers, Image, Video) in the Plugin Store.",
        action: jsx(Button, { size: "sm", variant: "outline", onClick: () => host.navigate("/capabilities?tab=plugins"), children: "Plugin Store" }),
      }),
      jsx(ListRow, {
        title: "Raw Agent Settings (ModelScope & Download Roots)",
        description: "Open the schema form for ModelScope credentials and extra download directories.",
        action: jsx(Button, { size: "sm", variant: "ghost", onClick: () => host.navigate("/settings?tab=plugins&agent=model-forge"), children: "Open Schema…" }),
      }),
      jsx(HfCredentialsDialog, { open: hfOpen, onOpenChange: setHfOpen, connected: hfSet, onSaved: load }),
    ],
  });
}

function SettingsComputeSubpage() {
  const [state, setState] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const load = () => { readSettingsAndSuite().then(setState).catch(() => null); };
  useEffect(() => { load(); }, []);
  const targets = (state && state.targets) || [];
  return jsxs("div", {
    className: "grid gap-3",
    children: [
      jsx(ListRow, {
        title: "Remote & Cloud GPU Targets",
        description: "Add another computer on your network or a cloud GPU box without editing JSON.",
        action: jsx(Button, { size: "sm", onClick: () => { setEditing(null); setAddOpen(true); }, children: "Add GPU Target" }),
      }),
      ...targets.map((t) => jsx(ListRow, {
        key: t.id,
        title: t.label + " (" + t.kind + ")",
        description: t.base_url + (t.vram_gb ? " · " + t.vram_gb + " GB VRAM" : ""),
        action: jsx(Button, { size: "xs", variant: "outline", onClick: () => { setEditing(t); setAddOpen(true); }, children: "Edit" }),
      })),
      jsx(AddComputeDialog, {
        open: addOpen,
        onOpenChange: setAddOpen,
        initialTarget: editing,
        existingTargets: targets,
        existingEngines: (state && state.declared) || [],
        defaultMode: "lan",
        onSaved: load,
      }),
    ],
  });
}

export default {
  id: "model-forge-desktop",
  name: "Model Forge",
  register(ctx) {
    // 1. Full native workspace page + sidebar navigation row
    ctx.register({
      id: "model-forge-page",
      area: ROUTES_AREA,
      data: { path: MODEL_FORGE_ROUTE },
      render: () => jsx(ModelForgePage, { ctx }),
    });
    ctx.register({
      id: "model-forge-nav",
      area: SIDEBAR_NAV_AREA,
      data: { path: MODEL_FORGE_ROUTE, label: "Model Forge", codicon: "layers" },
    });

    // 2. Native Settings -> Plugins -> Model Forge entry (feature-detected)
    ctx.registerSettingsPage?.({
      id: "settings",
      title: "Model Forge",
      icon: "layers",
      render: () => jsx(SettingsOverviewPage, {}),
      children: [
        { id: "compute", title: "Compute & GPUs", render: () => jsx(SettingsComputeSubpage, {}) },
      ],
    });

    // 3. Compact dockable side pane (backward compatible)
    ctx.register({
      id: "model-forge-desktop-pane",
      area: "panes",
      title: "Model Forge",
      data: { placement: "right" },
      render: () => jsx(Cockpit, { ctx }),
    });

    // 4. Command palette shortcuts
    ctx.register({
      id: "model-forge-desktop-open",
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
      id: "model-forge-desktop-refresh",
      area: PALETTE_AREA,
      title: "Model Forge: refresh cockpit",
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
