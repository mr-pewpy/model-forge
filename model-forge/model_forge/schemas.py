"""Tool schemas — what the LLM sees."""

STATUS = {
    "name": "model_forge_status",
    "description": (
        "Report every compute target (this machine, a LAN box, a cloud GPU) "
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
        },
    },
}

TARGET_ADD = {
    "name": "model_forge_target_add",
    "description": (
        "Register a compute target: a machine with GPUs where inference can "
        "run — this machine, an LXC on a Proxmox node, a rented cloud GPU "
        "box. Persists so future sessions know about it. Use when the user "
        "says their GPU lives somewhere else or wants to add a server."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Short unique id, e.g. 'ai-lxc' or 'lambda-a100'.",
            },
            "label": {
                "type": "string",
                "description": "Human label, e.g. 'AI LXC on Proxmox'.",
            },
            "base_url": {
                "type": "string",
                "description": "Engine root URL, e.g. http://192.168.1.62:8080",
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
                "description": "Free-form note, e.g. '2x RTX 3090'.",
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
        "built-in Local Models. Use when the user wants a model from "
        "model_forge_browse downloaded."
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
                    "Engine-native model id when the registry has none: a "
                    "Maya setup id (e.g. glm-maya-s) or, for Strata, "
                    "<family>:<pack> (e.g. deepseek41:ist-das-2.00bpw)."
                ),
            },
        },
        "required": ["model_id"],
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
