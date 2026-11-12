# Model Forge — hackathon scope

**Deadline: 5PM EST, Tue Nov 11, 2026** (Hermes Plugin Hackathon).
Entry = catalog PR + demo video on X (@NousResearch tag) + form.

Judges score **usefulness, reliability, polish, creativity** and favor
plugins that do "more than a prompt or a skill could". Model Forge qualifies:
it is a hardware-aware model catalog with engine detection, download
orchestration, and approval-gated activation — runtime behavior no skill can
emulate. (Honest scoping: it never installs or starts engines; it pulls
model files and emits exact run commands.)

## Storyboard (the demo, ~90s)

One gaming PC, no cloud, chained modality unlocks:

1. `model_forge_status` → "you have a 24 GB GPU, here's what runs"
2. chat model picked → Hermes chats on it
3. want an image → image model pulled (engine already warm) → generated
4. edit it → image-edit model pulled → edited
5. 3D-print it → Hunyuan3D pulls → GLB out
6. promo video → i2v model pulls → video out
7. voice + soundtrack → TTS/voice clone + music/SFX pulls → alive

World model = honest stub slot in the UI ("engine wanted") — show the seam,
don't fake the capability.

## Build order

- **D0 (done):** repo, plugin skeleton, capability registry (11 modalities),
  hardware probe, llama.cpp/Ollama/Strata detection, `model_forge_status` +
  `model_forge_browse` tools, tests.
- **D1 (done):** Strata adapter full (detect + `strata launch` argv builder),
  Maya adapter (detect via /api/catalog + Models-page API: catalog/loaded/
  pull/pull_status/cancel/delete), `models_loaded` surfaced in status,
  `model-forge-providers` package (strata + maya as keyless local Hermes
  providers, LM Studio profile pattern).
- **D2 (done):** compute targets (local/lan/cloud) with declared budgets,
  target-aware status/browse, `model_forge_target_add`; keyless local
  providers (strata, maya).
- **D3 (done):** engine adapters v2 — ComfyUI, vLLM(-Omni via images-route
  probe), SGLang(+Diffusion probe), honest generic OpenAI-compatible
  detector; `model_forge_install` + `model_forge_activate` (plan-only unless
  apply=true); HF/ModelScope pull credentials via Desktop Settings →
  Plugins (secret-scope backed, host-scoped in code).
- **D4 (done):** image-gen + video-gen provider packages (any OpenAI-
  compatible images API / video job API; Settings → Plugins endpoint
  fields); pull depth — credential-aware streaming downloader with resume +
  progress jobs (`job_id` polling), local file pulls + run-on-target
  instructions for remote; activate emits media_setup wiring commands;
  video-route probing on generic endpoints.
- **D5 (done):** `model_forge_generate_media` (3d/music/sfx — verified live:
  image→GLB through a media gateway's Hunyuan3D lane, glTF 2.0 confirmed); TTS/
  voice-clone command provider script (`tts/model_forge_tts.py`); cockpit
  desktop pane (`model-forge-desktop/`) — targets with live engine probes;
  `targets_json` setting (Desktop JSON editor feeds tools + pane); route
  probing hardened (openapi-first — some gateways 404 wrong-method GETs and
  empty POSTs on routes that exist).
- **Post-D5 hardening (done):** live discovery layer (`discovery.py`) —
  curated catalog + unvetted live HuggingFace freshness in
  `model_forge_browse` (`fresh=false` opt-out, 1-hour cache, fail-soft
  offline, see `docs/model-discovery.md`); hardware probe v2 (AMD via
  rocm-smi/sysfs, Apple Silicon unified memory, Windows RAM fallback);
  download hardening (sha256 + resume + destination confinement via
  `download_roots`, symlinks refused); human-approval policy hook on
  `model_forge_target_add` and `model_forge_activate apply=true`.
- **D6:** end-to-end rehearsal of the storyboard on real hardware; demo video
  recording (≤ 90s).
- **D7 (Nov 11):** catalog PR (`plugin-catalog/model-forge.yaml`, SHA-pinned),
  `hermes plugins validate` green, form submitted, video posted on X.

## Non-goals for the hackathon

- Multi-host orchestration (distributed serving territory — future plugin)
- Rebuilding core's GGUF pricing/catalog (we complement it)
- World model inference (no local open engine with a stable API yet)

## Submission checklist

- [ ] Public GitHub repo (canonical public home)
- [ ] `hermes plugins validate <path> --install-deps` green
- [ ] README disclosure lines match reality
- [ ] plugin-catalog entry: full 40-char SHA pin, quoted version,
      `requires_hermes` SemVer floor
- [ ] Demo video ≤ 90s on X tagging @NousResearch
- [ ] Typeform links submitted

## Risk register

- **Scope:** 6 modalities fully wired beats 12 half-wired. Cut audio before
  cutting the 3D step (3D-print → video is the memorable beat).
- **Catalog rules:** engine installs are shell commands — disclose; no
  bundled binaries (engine installers run from upstream sources, pinned).
- **GPU contention during demo:** pre-stage the two biggest models before
  recording; the video shows pulls for the smaller ones.
