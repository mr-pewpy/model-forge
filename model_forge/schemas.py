"""Tool schemas — what the LLM sees."""

STATUS = {
    "name": "model_forge_status",
    "description": (
        "Report this machine's accelerators (GPUs, VRAM, RAM), which local "
        "inference engines are running (llama.cpp, Ollama, Strata, ...), and "
        "what Model Forge can serve. Use when the user asks what models or "
        "engines run locally, what their hardware can handle, or before "
        "choosing a local model."
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
        "advisory hardware fit verdict for THIS machine. Use when the user "
        "wants to run something locally and asks what model to use, what fits "
        "their GPU, or what a modality needs."
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
        },
    },
}
