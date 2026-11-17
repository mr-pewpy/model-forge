// Model Forge — dashboard cockpit tab.
//
// The same read-only cockpit data the /model-forge command and the Desktop
// cockpit pane render: compute targets (hardware + detected engines),
// declared engine endpoints, model fit rows (the model_forge_browse fit
// data, advisory), and installed model storage (the model_forge_storage
// list shape) with a collapsed 'Resource view' (the on-demand memory
// snapshot — loaded only when the view is expanded). Served by
// dashboard/plugin_api.py at /api/plugins/model-forge/cockpit (and
// /api/plugins/model-forge/resources for the resource view).
//
// Read-only by design: deletion is a tool verb that rides the human-approval
// gate, so this tab renders the listing and names the delete verb honestly
// instead of offering an in-tab delete. Every failure state (loading, API
// error, timeout) is rendered as a stated reason — never silently defaulted.
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const React = SDK.React;
  const h = React.createElement;
  const C = SDK.components || {};

  const API_URL = "/api/plugins/model-forge/cockpit";
  // The on-demand resource snapshot's own route — fetched only when the
  // collapsed Resource view is expanded (and on its manual refresh click).
  // Nothing polls it: quiet by design.
  const RESOURCES_URL = "/api/plugins/model-forge/resources";
  // register() is called with the manifest.json name verbatim (same name
  // check the dashboard loader does).

  function fmtBytes(n) {
    const bytes = Number(n) || 0;
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + " GB";
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
    return bytes + " B";
  }

  function Section(props) {
    return h("div", { style: { marginTop: "18px" } },
      h("div", {
        style: {
          fontWeight: 600, fontSize: "12px", textTransform: "uppercase",
          letterSpacing: "0.04em", marginBottom: "6px",
          color: "var(--color-muted-foreground)",
        },
      }, props.title),
      props.children);
  }

  function Line(props) {
    return h("div", {
      style: {
        padding: "3px 0", fontSize: "12px", lineHeight: "1.5",
        borderBottom: "1px solid var(--color-border)",
        color: props.muted ? "var(--color-muted-foreground)"
                           : "var(--color-foreground)",
      },
    }, props.children);
  }

  function TargetsSection(props) {
    const rows = props.data.targets || [];
    return h(Section, { title: "GPU & compute targets (" + rows.length + ")" },
      rows.map(function (t) {
        const hw = t.hardware || {};
        const gpus = (hw.gpus || []).map(function (g) {
          return (g.name || "GPU") + (g.vram_gb ? " (" + g.vram_gb + " GB)" : "");
        }).join(", ") || "no GPU detected";
        return h("div", { key: "t-" + t.id },
          h(Line, null,
            (t.label || t.id) + " · " + (t.kind || "") + " · " + (t.base_url || "")),
          h(Line, { muted: true },
            "hardware: " + (hw.platform || "unknown") + " · GPUs: " + gpus +
            " · VRAM " + (hw.total_vram_gb || 0) + " GB · RAM " +
            (hw.ram_gb || 0) + " GB"),
          (t.engines || []).map(function (e, j) {
            return h(Line, { key: "e-" + j },
              "engine " + e.id + " (" + e.name + "): available=" +
              String(Boolean(e.available)) + " · " + (e.detail || "no detail reported") +
              " · models_loaded: " + ((e.models_loaded || []).join(", ") || "none") +
              " · source=" + (e.source || "auto-detected"));
          }),
          (t.engines || []).length === 0
            ? h(Line, { muted: true, key: "none" },
                "engines: none detected (no engine answered the probes)")
            : null);
      }),
      (props.data.declared_engines || []).length
        ? h(Section, { title: "Declared engine endpoints" },
            (props.data.declared_engines || []).map(function (d, i) {
              return h(Line, { key: "d-" + i },
                (d.label || d.base_url) + " — " + d.base_url +
                " · source=" + (d.source || "user-declared"));
            }))
        : h(Line, { muted: true },
            "No declared engine endpoints — the bring-your-own-engine path is " +
            "the 'Declared engine endpoints' plugin setting (engines_json)."));
  }

  function ModelsSection(props) {
    const rows = props.data.models || [];
    return h(Section, { title: "Models — browse fit data (" + rows.length + ")" },
      h(Line, { muted: true },
        "The model_forge_browse fit data (advisory; engines do authoritative " +
        "memory sizing at launch)."),
      rows.map(function (m, i) {
        const fit = m.fit || {};
        return h("div", { key: "m-" + i },
          h(Line, null,
            m.name + " · " + (m.modality_label || m.modality) +
            (fit.verdict ? " · fit=" + fit.verdict : "") +
            (m.size_gb ? " · ~" + m.size_gb + " GB" : "") +
            " · engines: " + ((m.engines || []).join(", ") || "none listed") +
            " · status=" + (m.status || "unknown")),
          fit.advisory ? h(Line, { muted: true }, fit.advisory) : null,
          m.note ? h(Line, { muted: true }, "note: " + m.note) : null);
      }));
  }

  function MemoryBar(props) {
    const row = props.row;
    const label = String(row.kind || "");
    if (row.status !== "ok" || row.used_gb == null || row.total_gb == null) {
      return h(Line, { muted: true },
        label + " unavailable — " + (row.detail || "no reading"));
    }
    const pct = Math.max(0, Math.min(100, Number(row.used_pct) || 0));
    return h("div", {
      style: {
        display: "flex", alignItems: "center", gap: "8px", padding: "2px 0",
        fontSize: "12px",
      },
    },
      h("span", { style: { minWidth: "42px" } }, label),
      h("div", {
        style: {
          width: "110px", height: "8px", borderRadius: "4px",
          border: "1px solid var(--color-border)", overflow: "hidden",
        },
      }, h("div", {
        style: { width: pct + "%", height: "100%",
                 background: "var(--color-primary, currentColor)" },
      })),
      h("span", null,
        row.used_gb + "/" + row.total_gb + " GB (" + Math.round(pct) + "%)" +
        (row.booked_now_gb != null ? " · booked_now " + row.booked_now_gb + " GB" : "") +
        (row.free_gb != null ? " · free " + row.free_gb + " GB" : "")));
  }

  function ResourceView() {
    // Collapsed by default; the snapshot loads once on expand (and on the
    // manual Refresh click) — no timers, no polling, no animation.
    const [open, setOpen] = React.useState(false);
    const [state, setState] = React.useState({ phase: "idle", data: null, error: "" });
    const load = function () {
      setState({ phase: "loading", data: null, error: "" });
      SDK.fetchJSON(RESOURCES_URL)
        .then(function (data) {
          if (!data || typeof data !== "object") {
            setState({ phase: "error", data: null,
                       error: "the resource API returned an unexpected payload" });
          } else if (data.ok === false) {
            setState({ phase: "error", data: null,
                       error: data.error || "the resource API reported a failure" });
          } else {
            setState({ phase: "ready", data: data, error: "" });
          }
        })
        .catch(function (err) {
          setState({ phase: "error", data: null,
                     error: "the resource API is unavailable: " +
                            ((err && err.message) || err) });
        });
    };
    const toggle = function () {
      if (!open && state.phase === "idle") load();
      setOpen(!open);
    };
    const children = [
      h("div", {
        key: "toggle",
        onClick: toggle,
        style: {
          cursor: "pointer", padding: "4px 0 2px", fontWeight: 600,
          fontSize: "12px", color: "var(--color-foreground)",
        },
      }, (open ? "▾ " : "▸ ") + "Resource view (on-demand)"),
    ];
    if (!open) {
      children.push(h(Line, { key: "closed", muted: true },
        "Memory use (VRAM/RAM) and the resident model blocks — loaded once " +
        "when you open it, never a persistent display. In chat: " +
        "/model-forge storage resources."));
      return h("div", null, children);
    }
    if (state.phase === "loading") {
      children.push(h(Line, { key: "l", muted: true },
        "Loading the resource snapshot…"));
    } else if (state.phase === "error") {
      children.push(h(Line, { key: "e", muted: true },
        "Resource snapshot unavailable: " + state.error +
        " Nothing was changed — Refresh to retry."));
    } else if (state.data) {
      (state.data.targets || []).forEach(function (t) {
        children.push(h("div", {
          key: "t-" + t.target,
          style: { margin: "6px 0 2px", fontWeight: 600, fontSize: "12px" },
        }, (t.label || t.target) + " · " + (t.kind || "")));
        (t.memory || []).forEach(function (row, i) {
          children.push(h(MemoryBar, { key: "m-" + t.target + "-" + i, row: row }));
        });
        const residents = t.residents || [];
        children.push(h(Line, { key: "rh-" + t.target, muted: true },
          "resident (" + residents.length + "):"));
        residents.forEach(function (r, i) {
          const size = r.size_gb != null
            ? r.size_gb + " GB (" + r.size_label + ")"
            : String(r.size_label || "size unknown");
          children.push(h(Line, { key: "r-" + t.target + "-" + i },
            r.model_id + " · size " + size + " · engine=" + r.engine +
            " · lane=" + r.lane));
        });
        (t.queued_loads || []).forEach(function (q, i) {
          children.push(h(Line, { key: "q-" + t.target + "-" + i, muted: true },
            "queued load: " + q.model_id + " · " + q.state));
        });
        if (t.swap_stats) {
          children.push(h(Line, { key: "ss-" + t.target, muted: true },
            "swap stats (measured): count=" + t.swap_stats.count +
            " · last=" + t.swap_stats.last_total_seconds + "s" +
            " · average=" + t.swap_stats.average_total_seconds + "s"));
        }
      });
      if (state.data.note) {
        children.push(h(Line, { key: "n", muted: true }, state.data.note));
      }
    }
    children.push(h("div", {
      key: "refresh",
      onClick: load,
      style: {
        cursor: "pointer", padding: "3px 0", fontSize: "12px", width: "max-content",
        color: "var(--color-primary, currentColor)",
      },
    }, "Refresh (manual — nothing polls)"));
    return h("div", null, children);
  }

  function StorageSection(props) {
    const storage = props.data.storage || {};
    const entries = storage.entries || [];
    return h(Section, { title: "Installed model storage (" + entries.length + ")" },
      h(Line, { muted: true }, storage.note ||
        "Sizes are on-disk sizes; modification times are filesystem times."),
      entries.map(function (e, i) {
        return h(Line, { key: "s-" + i },
          e.name + " · " + fmtBytes(e.size_bytes) + " · " +
          (e.last_modified || "") + " · " + (e.root_label || e.root || "") +
          (e.type ? " · " + e.type : ""));
      }),
      h(Line, { muted: true }, props.data.delete_note ||
        "Read-only listing. To delete an entry, run the model_forge_storage " +
        "tool with action=delete and the entry id — the human-approval gate " +
        "names the exact path and size first. This tab never deletes files."),
      // The on-demand resource view: collapsed by default, loads its
      // snapshot only when expanded (and on its manual Refresh) — quiet.
      h(ResourceView, { key: "resource-view" }));
  }

  function Invitation(props) {
    const inv = props.invitation;
    return h("div", {
      style: {
        margin: "10px 0", padding: "10px",
        border: "1px solid var(--color-border)", borderRadius: "6px",
        fontSize: "12px",
      },
    },
      h("div", { style: { fontWeight: 600, marginBottom: "4px" } },
        "Point Model Forge at your compute"),
      h("div", { style: { color: "var(--color-muted-foreground)" } }, inv.reason),
      (inv.next_steps || []).map(function (step, i) {
        return h("div", {
          key: "s" + i,
          style: { padding: "2px 0 2px 10px", lineHeight: "1.5" },
        }, (i + 1) + ". " + step);
      }));
  }

  function ModelForgePage() {
    const [state, setState] = React.useState({ phase: "loading", data: null, error: "" });
    React.useEffect(function () {
      let live = true;
      SDK.fetchJSON(API_URL)
        .then(function (data) {
          if (!live) return;
          if (!data || typeof data !== "object") {
            setState({ phase: "error", data: null,
                       error: "the cockpit API returned an unexpected payload" });
          } else if (data.ok === false) {
            setState({ phase: "error", data: null,
                       error: data.error || "the cockpit API reported a failure" });
          } else {
            setState({ phase: "ready", data: data, error: "" });
          }
        })
        .catch(function (err) {
          if (!live) return;
          setState({ phase: "error", data: null,
                     error: "the cockpit API is unavailable: " +
                            ((err && err.message) || err) });
        });
      return function () { live = false; };
    }, []);

    const body = state.phase === "loading"
      ? h("div", {
          style: { color: "var(--color-muted-foreground)", padding: "12px" },
        }, "Loading the Model Forge cockpit…")
      : state.phase === "error"
        ? h("div", {
            style: { color: "var(--color-destructive)", padding: "12px" },
          }, "Model Forge cockpit could not load: " + state.error +
             " Nothing was changed — refresh to retry.")
        : h("div", null,
            (state.data.invitation ? h(Invitation, { invitation: state.data.invitation }) : null),
            h(TargetsSection, { data: state.data }),
            h(ModelsSection, { data: state.data }),
            h(StorageSection, { data: state.data }));

    return h(C.Card || "div", { style: { margin: "12px" } },
      h(C.CardHeader || "div", null,
        h(C.CardTitle || "h2", null, "Model Forge"),
        h("div", {
          style: { fontSize: "12px", color: "var(--color-muted-foreground)" },
        }, "Read-only cockpit: compute targets, engines, model fit data, " +
           "installed model storage — the same data as /model-forge in chat.")),
      h(C.CardContent || "div", null, body));
  }

  window.__HERMES_PLUGINS__.register("model-forge", ModelForgePage);
})();
