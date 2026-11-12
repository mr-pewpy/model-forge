# Model discovery, fit verdicts, and hardware probes

How `model_forge_browse` and `model_forge_status` decide what to show. Every
statement here matches `model_forge/registry.py`, `model_forge/discovery.py`,
and `model_forge/hardware.py`.

## Two discovery layers

`model_forge_browse` returns two blocks:

1. **`models` — the curated catalog** (`registry.py`, the `MODELS` tuple).
   Hand-maintained entries with approximate download size, minimum VRAM, and
   an advisory fit verdict per compute target. Each entry carries a
   hand-assigned maturity label: `validated`, `known`, or `stub` (a stub is
   a slot with no working engine yet).
2. **`fresh` — the live HuggingFace layer** (`discovery.py`). Newest public
   models per modality lane, pulled from `huggingface.co/api/models` and
   sorted by last-modified date.

## What "fresh" means

Fresh rows are **unvetted discovery results**, deliberately distinct from
curated entries:

- newest-first by `last_modified` (undated rows last, id as tiebreak);
- every row carries `id`, `last_modified` (YYYY-MM-DD), `downloads`,
  `pipeline_tag`, a `url`, and `status: "fresh"`;
- **no fit estimate, no engine mapping, not installable** — install resolves
  curated registry ids only;
- deduplicated against the curated catalog (a curated source never appears
  twice);
- results are cached on disk (plugin data dir `cache/`) for **one hour**;
- lanes come from `MODALITY_TAGS` in `discovery.py`. Known lane quirks:
  music and sfx share `text-to-audio` (HuggingFace has no music-vs-sfx
  pipeline tags), and `world_model` probes a `world-model` tag that returns
  nothing today — an honest empty lane.

### Opting out

Pass `fresh=false` to `model_forge_browse`: curated catalog only, no network
call (`live_discovery: "disabled"`). Without the argument the live layer
runs and every failure (no network, no plugin storage, malformed response)
fails soft to `live_discovery: "unavailable"` with an explanatory note — the
curated catalog keeps working offline.

## Adding a curated entry

Append a `ModelEntry(...)` to `MODELS` in `model_forge/registry.py`. Browse,
install, and activate pick it up automatically. Fields:

| Field | Meaning |
|-------|---------|
| `id` | Stable id used by `model_forge_install` / `model_forge_activate` |
| `name`, `modality` | Display name; one of the `MODALITIES` tuple |
| `engines` | Engine ids that can serve it (first entry is the default) |
| `source` | HuggingFace repo id or engine-native id |
| `size_gb` | Approximate download size (0 = unknown) |
| `vram_gb_min` | Minimum VRAM for a comfortable run (0 = not VRAM-bound) |
| `ram_gb_min` | Minimum system RAM where relevant |
| `engine_managed_fit` | True when the engine sizes itself (Strata/Maya) |
| `status` | `validated` \| `known` \| `stub` |
| `engine_models` | Map of engine id → engine-native model id (e.g. Maya setup ids, Strata `family:pack`) |
| `note` | Free-form honesty note shown in browse |

For file-pull installs (ComfyUI/vLLM/SGLang/Python engines) the user must
still pass `engine_model=` with the exact file path inside the source repo —
the registry does not store or guess filenames.

## How the fit verdict works

`fit_verdict(entry, hardware)` in `registry.py` returns, in rule order:

| Verdict | Rule |
|---------|------|
| `unavailable` | entry `status` is `stub` |
| `engine_managed` | `engine_managed_fit` is set (Strata/Maya size at launch) |
| `unknown` | `size_gb` is 0 |
| `fits_gpu` | `vram_gb_min` set and detected VRAM ≥ `vram_gb_min` |
| `needs_gpu` | `vram_gb_min` set but no VRAM detected (0 GB) |
| `fits_memory` | `size_gb` ≤ budget |
| `too_big` | none of the above |

The budget is `total_vram_gb + max(0, ram_gb − 8)` — VRAM plus the RAM an
engine may stream weights from (8 GB reserved for the system).

Verdicts are **advisory**. They are estimates from published footprints; the
engine that serves a model does the authoritative memory sizing at launch.
For remote targets the hardware profile comes from the `vram_gb` / `ram_gb`
budgets you declared when registering the target.

## Hardware probes

`probe_hardware()` in `hardware.py`, per platform — every probe is
fail-soft (a missing tool or unreadable file degrades to "no data", never an
error):

| What | How | Fallback |
|------|-----|----------|
| NVIDIA GPUs | `nvidia-smi` CSV query (name + VRAM) | none — empty GPU list |
| AMD GPUs | `rocm-smi --showmeminfo vram` **and** amdgpu sysfs `/sys/class/drm/cardN/device/mem_info_vram_total` (vendor `0x1002`) — both probes run (the code concatenates them; a card visible to both can be listed twice) | — |
| Apple Silicon | unified memory: VRAM budget = total RAM; GPU labeled `Apple Silicon (unified memory)`, platform `darwin-arm64` | — |
| RAM | POSIX `os.sysconf` | Windows `GlobalMemoryStatusEx` (ctypes); 0.0 when both fail |

NVIDIA and AMD probes can both run on one machine (mixed rigs are summed
into `total_vram_gb`). Apple Silicon's total is RAM plus any discrete GPUs
the other probes find.
