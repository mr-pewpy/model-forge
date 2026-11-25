// Automated headless smoke test for model-forge-desktop/desktop/plugin.js.
// Evaluates the real ESM plugin module against a lightweight SDK + React mock
// (mirroring Hermes Desktop's runtime-loader pipeline) and exercises:
//   1. register(ctx): full workspace route, sidebar nav, settings page + subpage,
//      dockable side pane, command palette actions
//   2. probeRoot: all 9 engine fingerprints (llama.cpp, Ollama, Maya, ComfyUI +
//      VRAM suggestion, vLLM, SGLang, Exo /node_id & /state, OpenAI-compatible)
//   3. ModelForgePage: renders cleanly with populated data AND pre-chat empty
//      data across all 5 tabs (models, compute, engines, wiring, storage)
//   4. HfCredentialsDialog, AddComputeDialog, Cockpit, Settings pages: render
//      cleanly with zero ReferenceErrors, and verify save_key / plugins.manage
//      settings RPC payloads.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL: " + msg);
    process.exit(1);
  }
}

const pluginPath = path.join(__dirname, "..", "model-forge-desktop", "desktop", "plugin.js");
const rawSrc = fs.readFileSync(pluginPath, "utf8");

// Rewrite the 3 allowed ESM imports into sandbox parameter bindings so we run
// the real plugin.js verbatim inside vm without bundling.
let body = rawSrc
  .replace(/import\s*\{([\s\S]*?)\}\s*from\s*["']@hermes\/plugin-sdk["']\s*;/, "const {$1} = __SDK__;")
  .replace(/import\s*\{([\s\S]*?)\}\s*from\s*["']react\/jsx-runtime["']\s*;/, "const {$1} = __JSX__;")
  .replace(/import\s*\{([\s\S]*?)\}\s*from\s*["']react["']\s*;/, "const {$1} = __REACT__;")
  .replace(/export\s+default\s+/, "const __DEFAULT_EXPORT__ = ");

body += "\nmodule.exports = { plugin: __DEFAULT_EXPORT__, probeRoot, advisoryFitForTarget, loadFullState, ModelForgePage, Cockpit, HfCredentialsDialog, AddComputeDialog, SettingsOverviewPage, SettingsComputeSubpage, saveHfToken, removeHfToken, savePluginSettings };\n";

// Minimal synchronous React hook harness (records state slots per render).
let hookState = [];
let hookIdx = 0;
function resetHooks(initialSlots = []) {
  hookState = initialSlots.slice();
  hookIdx = 0;
}
const mockReact = {
  useState(init) {
    const i = hookIdx++;
    if (!(i in hookState)) {
      hookState[i] = typeof init === "function" ? init() : init;
    }
    return [hookState[i], (v) => { hookState[i] = typeof v === "function" ? v(hookState[i]) : v; }];
  },
  useEffect() { /* no-op in sync render smoke */ },
  useMemo(fn) { return fn(); },
};

const h = (type, props) => ({ type: typeof type === "function" ? type.name || "Fn" : String(type), props: props || {} });
const mockJsx = { jsx: h, jsxs: h };

const rpcCalls = [];
const notifications = [];
const navigated = [];
let mockSessionId = "sess-live-1";

const mockHost = {
  state: {
    focusedSessionId: { get: () => mockSessionId },
    activeSessionId: { get: () => mockSessionId },
  },
  composer: {
    submit: () => true,
  },
  notify: (n) => notifications.push(n),
  notifyError: (e, m) => notifications.push({ kind: "error", message: m, error: e }),
  navigate: (p) => navigated.push(p),
  newChat: () => navigated.push("NEW_CHAT"),
  request: async (method, params) => {
    rpcCalls.push({ method, params });
    if (method === "plugins.manage" && params.action === "list") {
      return {
        plugins: [
          {
            key: "model-forge",
            name: "model-forge",
            version: "1.2.0",
            enabled: true,
            settings_schema: [
              { key: "targets_json", type: "list", value: [{ id: "gpu-box", label: "GPU Box", base_url: "http://gpu-box.example.com:8080", kind: "lan", vram_gb: 24, ram_gb: 64 }] },
              { key: "engines_json", type: "list", value: [{ label: "Lab Ollama", base_url: "http://192.0.2.10:11434" }] },
              { key: "download_roots", type: "list", value: [] },
              { key: "hf_token", type: "secret", env: "HF_TOKEN", has_value: true },
              { key: "modelscope_token", type: "secret", env: "MODELSCOPE_API_TOKEN", has_value: false },
            ],
          },
          { key: "model-forge-providers", name: "model-forge-providers", version: "1.2.0", enabled: true },
          {
            key: "model-forge-image", name: "model-forge-image", version: "1.2.0", enabled: true,
            settings_schema: [
              { key: "base_url", value: "http://gpu-box.example.com:8080/v1" },
              { key: "default_model", value: "qwen-image-2.1" },
            ],
          },
          {
            key: "model-forge-video", name: "model-forge-video", version: "1.2.0", enabled: true,
            settings_schema: [
              { key: "base_url", value: "" },
              { key: "default_model", value: "" },
            ],
          },
        ],
      };
    }
    if (method === "plugins.manage" && params.action === "presence") {
      return { presence: [] };
    }
    if (method === "plugins.manage" && params.action === "settings") {
      return { ok: true, key: params.key, written: params.values };
    }
    if (method === "machine.facts") {
      return {
        machine: { os_family: "linux", os_release: "6.12", cpu_model: "AMD EPYC", ram_gb: 64, gpu_class: "nvidia" },
        has_nvidia_gpu: true,
      };
    }
    if (method === "shell.exec") {
      return { stdout: "[]", stderr: "", code: 0 };
    }
    if (method === "slash.exec") {
      return {
        output: JSON.stringify({
          ok: true,
          targets: [{
            id: "local", label: "This machine", kind: "local", base_url: "http://127.0.0.1:8080",
            hardware: { platform: "linux", gpus: [{ name: "RTX 4090", vram_gb: 24 }], total_vram_gb: 24, ram_gb: 64 },
            engines: [],
          }],
          models: [{
            id: "qwen-image-2.1", name: "Qwen-Image 2.1", modality: "image", modality_label: "Image generation",
            engines: ["vllm", "comfyui"], size_gb: 25, vram_gb_min: 20, status: "validated",
            fit: { verdict: "fits_gpu", engine_available: true, engine: "vllm", advisory: "" },
          }],
          storage: { entries: [{ id: "downloads:m", name: "m.safetensors", path: "/x/m", size_bytes: 5e9, last_modified: "2026-11-25" }] },
          residency: { ok: true, pulled: [], resident: [], booked: [], unbooked: [], budget: { nominal_gb: 24, booked_now_gb: 0, headroom_gb: 24 } },
          pull_credentials: { HF_TOKEN: true, MODELSCOPE_API_TOKEN: false },
          active_wiring: { ok: true, chat: { model: "qwen3:8b", provider: "custom" }, vision: {}, image: { provider: "model-forge" }, video: {}, custom_providers: [] },
        }),
      };
    }
    if (method === "model.save_key" || method === "model.disconnect") {
      return { ok: true };
    }
    return {};
  },
};

const uiStub = (name) => {
  const fn = (props) => ({ type: name, props });
  Object.defineProperty(fn, "name", { value: name });
  return fn;
};

const mockSdk = {
  Badge: uiStub("Badge"),
  Button: uiStub("Button"),
  Codicon: uiStub("Codicon"),
  ConfirmDialog: uiStub("ConfirmDialog"),
  CopyButton: uiStub("CopyButton"),
  Dialog: uiStub("Dialog"),
  DialogContent: uiStub("DialogContent"),
  DialogDescription: uiStub("DialogDescription"),
  DialogFooter: uiStub("DialogFooter"),
  DialogHeader: uiStub("DialogHeader"),
  DialogTitle: uiStub("DialogTitle"),
  DisclosureCaret: uiStub("DisclosureCaret"),
  EmptyState: uiStub("EmptyState"),
  Input: uiStub("Input"),
  ListRow: uiStub("ListRow"),
  PALETTE_AREA: "palette",
  ROUTES_AREA: "routes",
  SIDEBAR_NAV_AREA: "sidebar.nav",
  SearchField: uiStub("SearchField"),
  SegmentedControl: uiStub("SegmentedControl"),
  Separator: uiStub("Separator"),
  StatusDot: uiStub("StatusDot"),
  cn: (...xs) => xs.filter(Boolean).join(" "),
  host: mockHost,
};

let fetchResponder = async () => ({ ok: false });
const sandbox = {
  module: { exports: {} },
  __SDK__: mockSdk,
  __JSX__: mockJsx,
  __REACT__: mockReact,
  fetch: (url, opts) => fetchResponder(url, opts),
  AbortSignal: { timeout: () => undefined },
  console,
  Map, Set, Promise, Array, Object, Number, String, Boolean, Math, JSON, Error, RegExp,
};

vm.runInNewContext(body, sandbox, { filename: "plugin.js" });
const mod = sandbox.module.exports;

async function main() {
  // 1. Registration check
  assert(mod.plugin && mod.plugin.id === "model-forge-desktop", "plugin id mismatch");
  const registered = [];
  const settingsPages = [];
  const fakeCtx = {
    register: (c) => { registered.push(c); return () => {}; },
    registerSettingsPage: (p) => { settingsPages.push(p); return () => {}; },
    onEvent: () => () => {},
    os: { writeClipboard: async () => true },
  };
  mod.plugin.register(fakeCtx);
  assert(registered.some((c) => c.area === "routes" && c.data && c.data.path === "/model-forge"), "missing /model-forge route");
  assert(registered.some((c) => c.area === "sidebar.nav" && c.data && c.data.label === "Model Forge"), "missing sidebar nav row");
  assert(registered.some((c) => c.area === "panes" && c.id === "model-forge-desktop-pane"), "missing dockable side pane");
  assert(registered.filter((c) => c.area === "palette").length >= 2, "missing palette commands");
  assert(settingsPages.length === 1 && settingsPages[0].children.length === 1, "missing Settings -> Plugins page + Compute subpage");
  console.log("register(ctx): route + sidebar + settings page + pane + palette — OK");

  // 2. Engine probe fingerprints (including Exo & ComfyUI VRAM suggestion)
  const probeCases = [
    ["ComfyUI + VRAM", { "/system_stats": { system: { "comfyui-version": "0.3.1" }, devices: [{ vram_total: 24 * 1024 ** 3 }] } }, "comfyui", 24],
    ["vLLM", { "/version": { version: "0.8.1" }, "/v1/models": { data: [{ id: "Qwen/Qwen3-8B" }] } }, "vllm", 0],
    ["SGLang", { "/get_model_info": { model_path: "FLUX.2-klein" } }, "sglang", 0],
    ["llama.cpp", { "/props": { build_info: "b4000", model_path: "qwen.gguf" } }, "llamacpp", 0],
    ["Ollama", { "/api/tags": { models: [{ name: "qwen3:8b" }] } }, "ollama", 0],
    ["Project Maya", { "/api/tags": { models: [{ name: "glm" }] }, "/api/catalog": { models: [{ name: "glm-maya-s" }] } }, "maya", 0],
    ["Exo (/node_id)", { "/node_id": "node-abc", "/v1/models": { data: [{ id: "mlx/qwen" }] } }, "exo", 0],
    ["Exo (/state)", { "/state": { instances: {}, topology: {} } }, "exo", 0],
    ["Generic OpenAI", { "/v1/models": { data: [{ id: "custom-model" }] } }, "openai-compatible", 0],
  ];
  for (const [label, map, expectedId, expectedVram] of probeCases) {
    fetchResponder = async (url) => {
      const p = Object.keys(map).find((k) => url.endsWith(k));
      if (!p) return { ok: false };
      return { ok: true, json: async () => map[p] };
    };
    const found = await mod.probeRoot("http://127.0.0.1:9999");
    assert(found.some((e) => e.id === expectedId), "probeRoot failed for " + label);
    if (expectedVram > 0) {
      assert(found[0].suggested_vram_gb === expectedVram, "ComfyUI VRAM suggestion mismatch");
    }
  }
  console.log("probeRoot: all 9 engine fingerprints (incl. Exo + ComfyUI VRAM) — OK");

  // 3. loadFullState + ModelForgePage across all 5 tabs (populated & pre-chat)
  fetchResponder = async () => ({ ok: false });
  const populated = await mod.loadFullState();
  assert(populated.suite.length === 5, "expected 5 suite packages");
  assert(populated.credentials.HF_TOKEN === true, "expected HF_TOKEN true");
  assert(populated.probed.length === 2, "expected local + gpu-box targets");

  for (const tabId of ["models", "compute", "engines", "wiring", "storage"]) {
    resetHooks([populated, tabId]);
    const tree = mod.ModelForgePage({ ctx: fakeCtx });
    assert(tree && tree.props && Array.isArray(tree.props.children), "ModelForgePage tab " + tabId + " returned invalid tree");
    const serialized = JSON.stringify(tree);
    assert(serialized.includes("Model Forge"), "tab " + tabId + " missing title");
  }
  console.log("ModelForgePage: all 5 tabs render cleanly with populated state — OK");

  // Pre-chat empty state (no session open) must still render cleanly with machine.facts + preview catalog
  mockSessionId = null;
  const preChat = await mod.loadFullState();
  assert(preChat.bridge === null, "pre-chat bridge should be null");
  assert(preChat.machineFacts && preChat.machineFacts.machine.gpu_class === "nvidia", "machine.facts should still load pre-chat");
  resetHooks([preChat, "models"]);
  const preTree = mod.ModelForgePage({ ctx: fakeCtx });
  assert(JSON.stringify(preTree).includes("Catalog preview mode"), "pre-chat should show friendly preview banner");
  console.log("ModelForgePage: pre-chat mode (no session open) renders cleanly — OK");

  // 4. Compact side pane + dialogs + settings pages
  resetHooks([populated]);
  const paneTree = mod.Cockpit({ ctx: fakeCtx });
  assert(JSON.stringify(paneTree).includes("Open Full View"), "side pane missing Open Full View button");

  resetHooks([]);
  assert(mod.HfCredentialsDialog({ open: true, onOpenChange: () => {}, connected: false }), "HfCredentialsDialog failed");
  resetHooks([]);
  assert(mod.AddComputeDialog({ open: true, onOpenChange: () => {}, existingTargets: [], existingEngines: [], defaultMode: "cloud" }), "AddComputeDialog failed");
  resetHooks([populated]);
  assert(mod.SettingsOverviewPage(), "SettingsOverviewPage failed");
  resetHooks([populated]);
  assert(mod.SettingsComputeSubpage(), "SettingsComputeSubpage failed");
  console.log("Cockpit + Dialogs + Settings pages render cleanly — OK");

  // 5. Credential & settings write RPCs
  rpcCalls.length = 0;
  await mod.saveHfToken("hf_test_example_token");
  assert(rpcCalls.some((c) => c.method === "model.save_key" && c.params.slug === "huggingface"), "saveHfToken must call model.save_key");
  await mod.removeHfToken();
  assert(rpcCalls.some((c) => c.method === "model.disconnect" && c.params.slug === "huggingface"), "removeHfToken must call model.disconnect");
  await mod.savePluginSettings("model-forge", { targets_json: [] });
  assert(rpcCalls.some((c) => c.method === "plugins.manage" && c.params.action === "settings" && c.params.key === "model-forge"), "savePluginSettings must call plugins.manage settings");
  console.log("RPC write doors (model.save_key, model.disconnect, plugins.manage settings) — OK");

  // 6. Per-target advisory fit recomputation
  const remoteA100 = { target: { id: "cloud", label: "Cloud A100", vram_gb: 80, ram_gb: 128 }, engines: [{ id: "vllm", ok: true }] };
  const fit = mod.advisoryFitForTarget({ id: "qwen-image-2.1", engines: ["vllm", "comfyui"], size_gb: 25, vram_gb_min: 20 }, remoteA100);
  assert(fit.verdict === "fits_gpu", "expected fits_gpu on 80 GB cloud target, got " + fit.verdict);
  console.log("Per-target advisory fit calculation — OK");

  console.log("ALL DESKTOP GUI SMOKE CHECKS PASSED");
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
