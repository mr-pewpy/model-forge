"""Tool schemas — what the LLM sees."""

STATUS = {
    "name": "model_forge_status",
    "description": (
        "Report every compute target (this machine, a remote host, a cloud GPU) "
        "with its GPUs, VRAM, RAM, and which local inference engines are "
        "running there (llama.cpp, Ollama, Strata, Maya, ...) and what models "
        "they have loaded — plus the user-declared engine endpoints (Settings "
        "-> Plugins -> Model Forge) and the installed model-storage listing "
        "(names, sizes, modification times). When hardware detection finds no "
        "GPU and no compute targets exist, the response carries an invitation "
        "naming the exact next steps (model_forge_target_add, or declare an "
        "engine endpoint in settings) instead of a bare empty state. Use when "
        "the user asks what runs locally, what their hardware can handle, or "
        "where their GPUs are."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

BROWSE = {
    "name": "model_forge_browse",
    "description": (
        "Browse local AI models by modality (chat, vision, ocr, image, "
        "image_edit, video, 3d, music, sfx, voice_clone, world_model) with an "
        "advisory hardware fit verdict for a compute target. When hardware "
        "detection finds no GPU and no compute targets exist, the response "
        "carries an invitation naming the exact next steps "
        "(model_forge_target_add, or declare an engine endpoint in settings). "
        "Use when the user wants to run something locally and asks what model "
        "to use, what fits their GPU, or what a modality needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "modality": {
                "type": "string",
                "description": (
                    "Filter by modality, e.g. 'image', 'video', '3d'. "
                    "Omit for all modalities."
                ),
                "enum": [
                    "chat", "vision", "ocr", "image", "image_edit", "video",
                    "3d", "music", "sfx", "voice_clone", "world_model",
                ],
            },
            "engine": {
                "type": "string",
                "description": (
                    "Filter by inference engine id, e.g. 'llamacpp', 'strata', "
                    "'maya', 'comfyui', 'vllm'. Omit for all engines."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "Compute target id to price against (default 'local'). "
                    "See model_forge_status for the list."
                ),
            },
            "fresh": {
                "type": "boolean",
                "description": (
                    "Include the live HuggingFace freshness section (newest "
                    "models per modality, sorted by last-modified date). "
                    "Default true; false — or the strings 'false', '0', "
                    "'no' — means curated catalog only, no HuggingFace "
                    "call (target engine detection still runs)."
                ),
            },
        },
    },
}

TARGET_ADD = {
    "name": "model_forge_target_add",
    "description": (
        "Register a compute target: a machine with GPUs where inference can "
        "run — this machine, a remote server, a rented cloud GPU "
        "box. Persists so future sessions know about it. Use when the user "
        "says their GPU lives somewhere else or wants to add a server."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Short unique id, e.g. 'gpu-box' or 'lambda-a100'.",
            },
            "label": {
                "type": "string",
                "description": "Human label, e.g. 'GPU workstation'.",
            },
            "base_url": {
                "type": "string",
                "description": "Engine root URL, e.g. http://gpu-box.example.com:8080",
            },
            "kind": {
                "type": "string",
                "description": "Advisory label for the machine kind.",
                "enum": ["local", "lan", "cloud"],
            },
            "vram_gb": {
                "type": "number",
                "description": "Total GPU VRAM on that machine (for fit verdicts).",
            },
            "ram_gb": {
                "type": "number",
                "description": "System RAM on that machine (for fit verdicts).",
            },
            "note": {
                "type": "string",
                "description": "Free-form note, e.g. 'NVIDIA A100, 80 GB'.",
            },
        },
        "required": ["id", "base_url"],
    },
}

INSTALL = {
    "name": "model_forge_install",
    "description": (
        "Pull/download a model on a compute target. Maya targets download "
        "over HTTP with progress; Strata targets get the exact `strata "
        "launch` command to run there; GGUF chat models defer to Hermes' "
        "built-in Local Models. Or provision an inference engine with "
        "action='engine_install': the plan shows the vendor's exact "
        "official install command (Ollama, Strata, Project Maya) and it "
        "runs as a background job only after the human-approval gate "
        "approves it; llama.cpp is never installed — it defers to Hermes "
        "core's managed runtime. Use when the user wants a model from "
        "model_forge_browse downloaded, or an engine installed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["model_install", "engine_install"],
                "description": (
                    "'model_install' (default) pulls a model. "
                    "'engine_install' provisions an inference engine: plan "
                    "first (the vendor's exact install command), then "
                    "execute=true runs it after the human-approval gate "
                    "approves — denial or timeout runs nothing."
                ),
            },
            "model_id": {
                "type": "string",
                "description": "Registry model id from model_forge_browse "
                               "(required for model_install unless polling "
                               "with job_id).",
            },
            "target": {
                "type": "string",
                "description": "Compute target id (default 'local').",
            },
            "engine": {
                "type": "string",
                "description": (
                    "For model_install: which engine to install for when the "
                    "model supports several (e.g. 'vllm' or 'comfyui' for "
                    "qwen-image-2.1); default auto-selects from the engines "
                    "detected on the target and must be one of the model's "
                    "engines. For action=engine_install: the engine id to "
                    "provision ('ollama', 'strata' or 'maya' — their "
                    "vendors' official installers; 'llamacpp' returns "
                    "guidance to Hermes core's managed runtime)."
                ),
            },
            "execute": {
                "type": "boolean",
                "description": (
                    "For action=engine_install: true runs the planned "
                    "vendor-official install command as a background job "
                    "(progress via job_id, hard timeout) — the "
                    "human-approval gate shows the exact command first and "
                    "denial or timeout runs nothing. false/omitted = plan "
                    "only."
                ),
            },
            "engine_model": {
                "type": "string",
                "description": (
                    "Engine-native model id when the registry has none: a "
                    "Maya setup id (e.g. glm-maya-s), for Strata "
                    "<family>:<pack> (e.g. deepseek41:ist-das-2.00bpw), or "
                    "for engine file pulls a file path inside the source "
                    "repo (e.g. diffusion_models/your-weights.safetensors). "
                    "For file pulls, omit engine_model and Model Forge "
                    "resolves the primary weights file from the repo's "
                    "file list."
                ),
            },
            "dest": {
                "type": "string",
                "description": "Absolute destination path for a file download "
                               "(default: the plugin's downloads dir).",
            },
            "sha256": {
                "type": "string",
                "description": "Optional sha256 to verify a downloaded file.",
            },
            "job_id": {
                "type": "string",
                "description": "Poll a running download or engine-install "
                               "job instead of starting an install.",
            },
        },
        "required": [],
    },
}

GENERATE_MEDIA = {
    "name": "model_forge_generate_media",
    "description": (
        "Generate media through a compute target. image/video run through "
        "the local ComfyUI workflow runner when one is available on a local "
        "target (image lanes run synchronously and return data: URLs; video "
        "lanes submit a job and return filesystem asset paths, polled with "
        "job_id). On a target without a runner, image falls back to the "
        "target's media-gateway OpenAI images route "
        "(/v1/images/generations) and a remote video request is refused with "
        "a pointer to the Model Forge facade video job API (POST /v1/videos). "
        "3d/music/sfx ride the media-gateway endpoint only (lanes: "
        "/v1/videos/generations for 3d jobs, /v1/audio/music, /v1/audio/sfx); "
        "3d submits a job (poll with job_id) while music/sfx run "
        "synchronously. The GPU lane is claimed around the job and released "
        "when the job completes or fails. Known limit: the ComfyUI-native job "
        "flow for 3d/music/sfx is not implemented — a plain ComfyUI endpoint "
        "does not serve those lanes; they need a media gateway. Use when the "
        "user wants an image, video, 3D model, soundtrack, or sound effect "
        "generated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "modality": {
                "type": "string",
                "description": "What to generate.",
                "enum": ["3d", "music", "sfx", "image", "video"],
            },
            "model": {
                "type": "string",
                "description": "Engine lane/model id, e.g. 'gen/3d' on a "
                               "media gateway target, or a ComfyUI workflow "
                               "lane (e.g. 'gen/krea2' for image, "
                               "'gen/h3-t2v' for video).",
            },
            "prompt": {
                "type": "string",
                "description": "Text prompt (required for image, video, "
                               "music, sfx; optional for 3d).",
            },
            "image": {
                "type": "string",
                "description": "Source image: URL or local file path (for 3d "
                               "image-to-GLB, and img2img/edit lanes).",
            },
            "target": {
                "type": "string",
                "description": "Compute target id (default 'local').",
            },
            "job_id": {
                "type": "string",
                "description": "Poll a submitted 3d or local video job "
                               "instead of starting.",
            },
        },
        "required": ["modality"],
    },
}

ACTIVATE = {
    "name": "model_forge_activate",
    "description": (
        "Connect a running chat/vision model as THIS Hermes' default model "
        "on the next session. Safe by default: without apply=true it only "
        "shows what would change. apply=true (behind the human-approval "
        "gate) writes the model config AND registers the matching "
        "custom_providers entry so the model is selectable via /model "
        "afterwards. Use when the user says 'use this model', 'run hermes "
        "on it', or 'make it my model'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_id": {
                "type": "string",
                "description": "Registry model id from model_forge_browse.",
            },
            "target": {
                "type": "string",
                "description": "Compute target id (default 'local').",
            },
            "engine_model": {
                "type": "string",
                "description": (
                    "Engine-native served model id to write into the config "
                    "when the registry default is not what the engine serves."
                ),
            },
            "apply": {
                "type": "boolean",
                "description": (
                    "true = actually write model.* config and the "
                    "custom_providers entry (confirm with the user first). "
                    "false/omitted = plan only."
                ),
            },
        },
        "required": ["model_id"],
    },
}

STORAGE = {
    "name": "model_forge_storage",
    "description": (
        "Model weights on this machine and the on-demand resource view. "
        "'list' (default) reports each entry in the Model Forge downloads "
        "directory plus the engine-managed model directories (the Ollama "
        "models dir, the Hugging Face hub cache): name, absolute path, size "
        "in bytes, and filesystem modification time. 'delete' removes one "
        "listed entry — the user approves the exact path and size first, "
        "and only an entry id or exact path from a previous list is "
        "accepted (arbitrary paths are refused). 'resources' (explicit, "
        "on-demand) takes a read-only memory snapshot per compute target — "
        "VRAM and RAM as used / booked_now / free in GB with percentages "
        "(used from the hardware probe, booked_now = the live claims of the "
        "models resident right now), resident model blocks (model id, size, "
        "engine, lane), queued loads, and measured swap stats where present "
        "— with honest 'unavailable' rows when a probe fails. The snapshot "
        "runs only when asked: nothing polls or updates on its own, and "
        "action=list stays lean. Use when the user asks what model files "
        "are on disk, how much space they take, wants unwanted weights "
        "deleted, or asks what is using memory right now."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "delete", "resources"],
                "description": (
                    "'list' (default) inventories model storage; 'delete' "
                    "removes one listed entry (asks for the user's approval "
                    "first); 'resources' takes the on-demand read-only "
                    "memory snapshot (VRAM/RAM usage, resident model "
                    "blocks) — shown only when asked."
                ),
            },
            "entry": {
                "type": "string",
                "description": (
                    "For action=delete: the entry id returned by "
                    "action=list, or that entry's exact absolute path. "
                    "Anything else is refused."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "For action=resources: one compute target id (default: "
                    "every registered compute target)."
                ),
            },
        },
        "required": [],
    },
}

BROKER = {
    "name": "model_forge_broker",
    "description": (
        "Weight residency and workflow scopes on a compute target: what "
        "weights are on disk and what is loaded in memory now, and the "
        "claim/release verbs that let one GPU lane serve several models in "
        "turn. 'status' (read-only) reports pulled weights with sizes, what "
        "is resident, booked ceilings, budget usage, and measured swap "
        "stats. 'claim' takes the GPU lane for a modality or model: it runs "
        "the eviction policy, steps down what must go, pre-warms the "
        "requested model, and returns a scope id with honest progress states "
        "(queued / loading / ready). 'release' ends a scope by id and "
        "restores exactly what was resident at claim time. 'swap' is the "
        "one-call exchange verb: it swaps the resident weights for one "
        "incoming model in strict phases (PREFLIGHT / EVICT / LOAD / READY) "
        "with measured timings — never estimates — and the pre-flight "
        "guarantee that nothing is evicted before the preflight passes "
        "(a failed preflight leaves residency untouched). 'plan' is a dry "
        "run of claim — the release/load steps with reasons, nothing "
        "executed. 'bench' measures what you run: a short VRAM measurement "
        "window on a loaded (or about-to-run) model records its real "
        "observed peak and books it as the model's nominal ceiling — "
        "measured peaks replace published estimates and are never "
        "downgraded. A claim or swap that would START an engine server "
        "process (the vLLM and llama.cpp lifecycles) requires human "
        "approval before it runs — the server binds 127.0.0.1 only, never "
        "all interfaces; a claim that just takes a lane over an "
        "already-resident model starts nothing and needs no approval. Use "
        "when the user wants models swapped on one GPU "
        "(chat -> image generation -> back to chat as one workflow), wants "
        "one model exchanged for another in one call with measured timings, "
        "asks what is loaded and what it costs, wants to preview a swap, or "
        "wants a model's true VRAM footprint measured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "claim", "release", "plan", "swap",
                         "bench"],
                "description": (
                    "'status' = read-only residency summary; 'claim' = take "
                    "the GPU lane for a modality or model and return a scope "
                    "id (starting a server process, e.g. vLLM or llama.cpp, "
                    "requires human approval); 'release' = end a scope and "
                    "restore what was resident at claim time (if that "
                    "restore would restart a vLLM/llama.cpp server, the "
                    "release requires human approval too); 'plan' = dry "
                    "run of claim (nothing executed); 'swap' = exchange the "
                    "resident weights for one incoming model in one measured "
                    "call (starting a server process requires human "
                    "approval; pass dry_run=true for the preflight plan rows "
                    "only — nothing is changed); 'bench' = measure a model's "
                    "VRAM peak over a short window and book it as its "
                    "nominal ceiling (pass model_id)."
                ),
            },
            "model_id": {
                "type": "string",
                "description": (
                    "Catalog model id from model_forge_browse to claim the "
                    "lane for (claim/plan), the incoming model to exchange "
                    "the resident weights for (swap), or the model id to "
                    "measure (bench — load it first, or exercise it during "
                    "the bench window)."
                ),
            },
            "modality": {
                "type": "string",
                "description": (
                    "Claim the lane for a whole modality instead of one "
                    "model: its first curated model is used (pass model_id "
                    "to pick a specific one)."
                ),
            },
            "scope_id": {
                "type": "string",
                "description": (
                    "For action=release: the scope id returned by claim. "
                    "For action=claim: optional name for the scope (one is "
                    "generated when omitted)."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "Compute target id (default 'local'). See "
                    "model_forge_status for the list."
                ),
            },
            "release": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "For action=swap: explicit outgoing model ids to step "
                    "down (a comma-separated string is also accepted). "
                    "Default: the eviction policy's own set. "
                    "Image-generation swaps always evict GPU residents "
                    "deterministically, so their explicit list is honored "
                    "as part of the full eviction."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "For action=swap: true returns the preflight plan rows "
                    "with an explicit 'nothing was changed' note and zero "
                    "side effects. false/omitted = execute the swap."
                ),
            },
        },
        "required": ["action"],
    },
}
