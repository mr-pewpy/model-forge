"""Tool schemas — what the LLM sees."""

STATUS = {
    "name": "model_forge_status",
    "description": (
        "Report every compute target (this machine, a remote host, a cloud GPU) "
        "with its GPUs, VRAM, RAM, and which local inference engines are "
        "running there (llama.cpp, Ollama, Strata, Maya, ...) and what models "
        "they have loaded. Use when the user asks what runs locally, what "
        "their hardware can handle, or where their GPUs are."
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
        "advisory hardware fit verdict for a compute target. Use when the "
        "user wants to run something locally and asks what model to use, "
        "what fits their GPU, or what a modality needs."
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
        "Generate 3D models, music, or sound effects through a "
        "user-configured OpenAI-compatible media gateway endpoint on a "
        "compute target (lanes: /v1/videos/generations for 3d jobs, "
        "/v1/audio/music, /v1/audio/sfx). 3d: image to GLB (submit job, "
        "then poll with job_id). music/sfx: prompt to audio; the call "
        "blocks while the audio renders and the result is saved to the "
        "plugin media directory. The GPU lane is claimed around the job and "
        "released when the job completes or fails. Known limit: the "
        "ComfyUI-native job flow is not implemented — a plain ComfyUI "
        "endpoint does not serve these lanes. Use when the user wants a 3D "
        "model, soundtrack, or sound effect generated through a media "
        "gateway."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "modality": {
                "type": "string",
                "description": "What to generate.",
                "enum": ["3d", "music", "sfx"],
            },
            "model": {
                "type": "string",
                "description": "Engine lane/model id, e.g. 'gen/3d' on a "
                               "media gateway target.",
            },
            "prompt": {
                "type": "string",
                "description": "Text prompt (required for music/sfx; "
                               "optional for 3d).",
            },
            "image": {
                "type": "string",
                "description": "Source image for 3d: URL or local file path.",
            },
            "target": {
                "type": "string",
                "description": "Compute target id (default 'local').",
            },
            "job_id": {
                "type": "string",
                "description": "Poll a submitted 3d job instead of starting.",
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
        "shows what would change. Use when the user says 'use this model', "
        "'run hermes on it', or 'make it my model'."
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
                    "true = actually write model.* config (confirm with the "
                    "user first). false/omitted = plan only."
                ),
            },
        },
        "required": ["model_id"],
    },
}

STORAGE = {
    "name": "model_forge_storage",
    "description": (
        "List and delete model weights on this machine: the Model Forge "
        "downloads directory plus the engine-managed model directories "
        "(the Ollama models dir, the Hugging Face hub cache). 'list' "
        "(default) reports each entry's name, absolute path, size in "
        "bytes, and filesystem modification time. 'delete' removes one "
        "listed entry — the user approves the exact path and size first, "
        "and only an entry id or exact path from a previous list is "
        "accepted (arbitrary paths are refused). Use when the user asks "
        "what model files are on disk, how much space they take, or wants "
        "unwanted weights deleted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "delete"],
                "description": (
                    "'list' (default) inventories model storage; 'delete' "
                    "removes one listed entry (asks for the user's approval "
                    "first)."
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
        "is resident, booked ceilings, and budget usage. 'claim' takes the "
        "GPU lane for a modality or model: it runs the eviction policy, "
        "steps down what must go, pre-warms the requested model, and "
        "returns a scope id with honest progress states (queued / loading / "
        "ready). 'release' ends a scope by id and restores exactly what was "
        "resident at claim time. 'plan' is a dry run of claim — the "
        "release/load steps with reasons, nothing executed. Use when the "
        "user wants models swapped on one GPU (chat -> image generation -> "
        "back to chat as one workflow), asks what is loaded and what it "
        "costs, or wants to preview a swap."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "claim", "release", "plan"],
                "description": (
                    "'status' = read-only residency summary; 'claim' = take "
                    "the GPU lane for a modality or model and return a scope "
                    "id; 'release' = end a scope and restore what was "
                    "resident at claim time; 'plan' = dry run of claim "
                    "(nothing executed)."
                ),
            },
            "model_id": {
                "type": "string",
                "description": (
                    "Catalog model id from model_forge_browse to claim the "
                    "lane for (claim/plan)."
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
        },
        "required": ["action"],
    },
}
