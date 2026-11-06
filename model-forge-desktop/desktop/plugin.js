// Model Forge — cockpit pane.
//
// Targets + engines + live model counts for every compute target. Targets
// come from the model-forge plugin's settings (Desktop Settings -> Plugins ->
// Model Forge -> "Compute targets"), falling back to a local default; engine
// probes run directly against each target's HTTP API (no generation, just
// /v1/models and friends). A blocked fetch degrades to a hint row.

import { host, PALETTE_AREA } from "@hermes/plugin-sdk";
import { jsx } from "react/jsx-runtime";
import { useEffect, useState } from "react";

const DEFAULT_TARGETS = [
  { id: "local", label: "This machine", base_url: "http://127.0.0.1:8080", kind: "local" },
];

async function readTargets() {
  try {
    const res = await host.request("plugins.manage", { action: "list" });
    const entries = (res && (res.entries || res.plugins)) || {};
    const entry = entries["model-forge"] || {};
    const rows = (entry.settings || {}).targets_json || [];
    const targets = rows
      .filter((t) => t && t.id && t.base_url)
      .map((t) => ({
        id: t.id,
        label: t.label || t.id,
        base_url: t.base_url,
        kind: t.kind || "lan",
        vram_gb: t.vram_gb || 0,
        ram_gb: t.ram_gb || 0,
      }));
    return [...DEFAULT_TARGETS.filter(
      (d) => !targets.some((t) => t.id === d.id)), ...targets];
  } catch (err) {
    return DEFAULT_TARGETS;
  }
}

async function probeTarget(target) {
  const root = String(target.base_url || "").replace(/\/+$/, "").replace(/\/v1$/, "");
  const engines = [];
  const tryFetch = async (path) => {
    try {
      const r = await fetch(root + path, { signal: AbortSignal.timeout(2500) });
      if (r.ok) return await r.json();
    } catch (err) { /* offline or CORS-blocked */ }
    return null;
  };
  const models = await tryFetch("/v1/models");
  const tags = await tryFetch("/api/tags");
  const stats = await tryFetch("/system_stats");
  if (stats && (stats.system || stats.devices)) {
    engines.push({
      name: "ComfyUI", ok: true,
      detail: (stats.system || {})["comfyui-version"] || "running",
    });
  }
  if (models && Array.isArray(models.data)) {
    engines.push({
      name: "OpenAI-compatible", ok: true,
      detail: models.data.slice(0, 6).map((m) => m.id).join(", "),
      models: models.data.length,
    });
  }
  if (tags && Array.isArray(tags.models)) {
    engines.push({
      name: "Ollama", ok: true,
      detail: tags.models.slice(0, 3).map((m) => m.name).join(", "),
      models: tags.models.length,
    });
  }
  if (engines.length === 0) {
    engines.push({
      name: "none detected", ok: false,
      detail: "no engine answered at " + root +
        " (or the app blocked the probe) — use the model_forge_status tool",
    });
  }
  return { target, engines };
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
  const load = () => {
    readTargets().then(async (targets) => {
      const probed = await Promise.all(targets.map(probeTarget));
      setRows(probed);
    });
  };
  useEffect(() => {
    load();
    const disposer = ctx.onEvent("plugin.model-forge.settings.changed", load);
    return disposer;
  }, []);
  if (!rows) {
    return jsx("div", {
      style: { color: "var(--ui-text-secondary)", padding: "8px" },
      children: "Probing compute targets…",
    });
  }
  const children = [];
  rows.forEach((r) => {
    children.push(jsx("div", {
      key: "t-" + r.target.id,
      style: { margin: "10px 0 4px", fontWeight: 600 },
      children: (r.target.label || r.target.id) +
        " · " + (r.target.kind || "") +
        (r.target.vram_gb ? " · " + r.target.vram_gb + " GB VRAM" : "") +
        (r.target.ram_gb ? " · " + r.target.ram_gb + " GB RAM" : ""),
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
        host.notify({ kind: "info", message: "Model Forge cockpit refreshes automatically." });
      },
    });
  },
};
