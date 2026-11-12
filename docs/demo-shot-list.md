# Model Forge — Demo Video Shot List (≤ 90 s)

One GPU workstation, zero cloud, one Hermes. Show the plugin working —
live calls only, no mockups. Screen-capture the desktop app + terminal.

| # | Time | Shot | On screen | Narration beat |
|---|------|------|-----------|----------------|
| 1 | 0:00–0:08 | Title + the ask | Desktop with Model Forge cockpit pane open (targets + live engines) | "One GPU box. Every kind of model. Watch." |
| 2 | 0:08–0:20 | Find the hardware | Agent calls `model_forge_status` → remote GPU target, hardware budgets, engines + loaded models | "Hermes finds the GPU — even when it lives in another machine." |
| 3 | 0:20–0:32 | Pick + chat | `model_forge_browse` (chat rows, fit verdicts) → chat answer on the local model | "Pick a model by what fits. Chat runs on it." |
| 4 | 0:32–0:46 | Make an image | `image_generate` → real image appears (cuts to result while it renders) | "An image? The plugin pulls the model and generates." |
| 5 | 0:46–0:56 | Edit it | one-line edit request → edited image | "Edit it. Same stack, edit model." |
| 6 | 0:56–1:08 | 3D-print it | `model_forge_generate_media` 3d → GLB lands (show file/mesh) | "Turn it into a printable 3D model." |
| 7 | 1:08–1:22 | Promo video | i2v job submitted → timelapse of result | "Promo video from the render. All local." |
| 8 | 1:22–1:30 | Close | cockpit pane again + "open source, hackathon entry" card | "Model Forge. Local models for everything." |

## Recording rules
- Real speeds, cuts only BETWEEN beats (never fake progress bars).
- Pre-stage nothing except the engines being warm; every model pull in the
  video must be a real pull.
- Keep the agent's tool calls visible — the tool bridge IS the demo.
- If the i2v beat (the slowest, ~15 min real) is too slow for a cutaway,
  show the job poll + jump cut to the finished clip; say "a few minutes".

## Verified timings (2026-11-06 rehearsal, GPU workstation)
- image (qwen-image-2.1): ~110 s · (moss): ~198 s
- 3D (hunyuan3d): ~55 s submit→done
- i2v (h3): ~15 min (jump-cut candidate)
- status/browse/targets: < 1 s each

## Ship checklist
- [ ] Record screen (desktop + one terminal), 90 s hard cap
- [ ] Post on X tagging @NousResearch
- [ ] Catalog PR: plugin-catalog/model-forge.yaml, 40-char SHA pin
- [ ] Typeform: both links
