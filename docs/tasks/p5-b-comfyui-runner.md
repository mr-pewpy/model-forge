# P5b — ComfyUI workflow runner (task file)

Roadmap: docs/broker-roadmap.md "P5 — broker consolidation", workstream (b).
Porting source (READ-ONLY): /home/hermes/work/weights-broker —
broker/main.py ComfyBackend._inject (line 213), GenWorker (1181),
_openai_image_params (1330), _normalize_image_refs (1298); config.yaml
`workflows:` (line 229) + workflows/*.json (copy VERBATIM).
NEVER `git reset --hard` in that repo.

## B1 — engines/comfyui.py workflow runner

- Template registry: the 7 workflow configs (h3-t2v, h3-i2v, h3-r2v,
  qwen-image-2.1, -edit, -cutout, moss-image) with template path,
  output_kind (image/video/3d), param_map — copied from config.yaml
  `workflows:`; templates carried as package files verbatim from
  workflows/*.json.
- Port `_inject` behavior: explicit param_map ({node, field} dicts) first,
  then class-based fallback for seed/steps/prompt only; size → width/height
  (_openai_image_params); image_url → image.
- /prompt submit with 405 cold-boot retry (~6 tries, 10 s sleep when "405"
  in the error — a cold ai-dock boot fails submit 100% without this);
  /history polling; output collection by output_kind (image ui images,
  video SaveVideo, 3d ui["3d"] from SaveGLB); data-URL / http image refs →
  POST /upload/image normalized to the NESTED {"upload": {"path": ...}}
  shape (the broker bug at pitfall 16: the path is val["upload"]["path"]);
  explicit cache free (POST /free) between back-to-back jobs; alive() probe
  (container-up != service-up).
- Stdlib-only + injectable seams (repo convention: urllib-based default,
  fakes in tests; no third-party deps — the test venv has none).
- Owner files: model-forge/model_forge/engines/comfyui.py,
  model-forge/model_forge/workflows/ (new pkg dir), tests/test_comfyui_runner.py
  (new).

## B2 — wiring

- model_forge_generate_media + Hermes image/video providers
  (model-forge-image/, model-forge-video/) run lanes through the runner;
  old gen id aliases resolve via model_forge/aliases.py (C1's seam).
- Owner files: model-forge/model_forge/tools.py (media path only),
  model-forge/model_forge/media.py, model-forge-image/, model-forge-video/,
  tests/test_media_wiring.py (new).

## Acceptance (integration)

All 7 gen lanes run end-to-end through model-forge with the same output
shapes as today (images via POST /v1/images/generations semantics, video via
the job API); old gen id aliases resolve.
