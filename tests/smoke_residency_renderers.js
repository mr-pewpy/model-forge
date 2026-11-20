// Smoke: run both shipped ResidencySection renderers against the real
// _broker_status data shape and assert they render nodes without throwing.
const fs = require("fs");
const vm = require("vm");

// Real cockpit_residency() / _broker_status shape (verified field-by-field).
const realData = {
  residency: {
    ok: true, action: "status", target: "local",
    pulled: [{ model_id: "model-a", path: "/w/model-a.bin", size_bytes: 7, source: "" }],
    resident: [{ model_id: "chat-a", engine: "vllm", lane: "chat" }],
    booked: [{ model_id: "chat-a", vram_gb: 6.0, source: "measured", label: "measured peak (bench)" }],
    unbooked: [{ model_id: "gen-x", modality: "image", vram_gb: null, label: "n/a", hint: "run bench" }],
    budget: { nominal_gb: 24.0, booked_now_gb: 6.0, headroom_gb: 18.0 },
    unbooked_residents: [], swap_stats: { count: 0, last_total_seconds: null, average_total_seconds: null },
    active_scopes: [], note: "Read-only report.",
  },
};

function makeMocks() {
  const h = (type, props, ...children) => ({ type, props: props || {}, children });
  return {
    h, jsx: h,
    Section: "Section", Line: "Line", Hint: "Hint", Row: "Row", SectionTitle: "SectionTitle",
    fmtBytes: (n) => (n == null ? "?" : n + "B"),
  };
}

function extract(src, name, nextName) {
  const start = src.indexOf("function " + name);
  const end = src.indexOf("function " + nextName, start);
  return src.slice(start, end);
}

function runRenderer(src, name, nextName, argName, mocks, data) {
  const body = extract(src, name, nextName);
  const keys = Object.keys(mocks).join(",");
  const fn = new Function(keys, body + "\nreturn " + name + ";");
  const comp = fn(...Object.keys(mocks).map((k) => mocks[k]));
  const out = comp(data);
  return out;
}

function assert(cond, msg) { if (!cond) { console.error("FAIL: " + msg); process.exit(1); } }

// Dashboard (React.createElement / h).
const dash = fs.readFileSync("model-forge/dashboard/dist/index.js", "utf8");
const dashOut = runRenderer(dash, "ResidencySection", "Invitation", "props", makeMocks(), { data: realData });
assert(dashOut, "dashboard ResidencySection returned nothing");
const dashStr = JSON.stringify(dashOut);
assert(dashStr.includes("Broker weight residency"), "dashboard: missing title");
assert(dashStr.includes("model-a"), "dashboard: missing pulled model");
assert(dashStr.includes("chat-a"), "dashboard: missing resident model");
assert(dashStr.includes("measured peak"), "dashboard: missing booked label");
assert(dashStr.includes("headroom"), "dashboard: missing budget headroom");
console.log("dashboard ResidencySection: rendered " +
  (dashOut.children ? dashOut.children.length : 0) + " nodes — OK");

// Desktop pane (jsx + SectionTitle/Hint/Row).
const pane = fs.readFileSync("model-forge-desktop/desktop/plugin.js", "utf8");
const paneOut = runRenderer(pane, "ResidencySection", "Cockpit", "props", makeMocks(), { bridge: realData, bridgeHint: "" });
assert(Array.isArray(paneOut) && paneOut.length, "pane ResidencySection returned no children");
const paneStr = JSON.stringify(paneOut);
assert(paneStr.includes("Broker weight residency"), "pane: missing title");
assert(paneStr.includes("model-a"), "pane: missing pulled model");
assert(paneStr.includes("chat-a"), "pane: missing resident model");
assert(paneStr.includes("measured peak"), "pane: missing booked label");
console.log("pane ResidencySection: rendered " + paneOut.length + " children — OK");

// Empty/error path must also render honestly (no throw).
const empty = runRenderer(dash, "ResidencySection", "Invitation", "props", makeMocks(), { data: {} });
assert(empty, "dashboard empty path returned nothing");
const err = runRenderer(dash, "ResidencySection", "Invitation", "props", makeMocks(), { data: { residency: { ok: false, error: "stack down" } } });
assert(JSON.stringify(err).includes("stack down"), "error path not surfaced");
console.log("empty + error paths render honestly — OK");
console.log("ALL RESIDENCY RENDERER SMOKE CHECKS PASSED");
