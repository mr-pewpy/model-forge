#!/usr/bin/env python3
"""Model Forge TTS / voice-clone command provider.

Config-driven TTS provider (no Python plugin needed) — wire it in config.yaml:

  tts:
    provider: model_forge
    providers:
      model_forge:
        command: ["python3", "<this file>", "--endpoint", "http://<target>:<port>",
                  "--voice", "default"]

Speaks any OpenAI-style /v1/audio/speech endpoint. For voice cloning, pass
--reference <audio file> and the engine clones that voice (engines that
support it, e.g. Chatterbox-style backends).

Usage: model_forge_tts.py --endpoint URL [--voice NAME] [--reference FILE]
       --in TEXT_FILE --out OUT.mp3   (stdin text + stdout bytes also work)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="engine root, e.g. http://gpu-box.example.com:11434")
    ap.add_argument("--voice", default="default")
    ap.add_argument("--reference", default="", help="voice-clone reference audio file")
    ap.add_argument("--in", dest="infile", default="-")
    ap.add_argument("--out", dest="outfile", default="-")
    args = ap.parse_args()

    text = (open(args.infile, encoding="utf-8").read() if args.infile != "-"
            else sys.stdin.read())
    if not text.strip():
        print("no text", file=sys.stderr)
        return 1

    body: dict = {"model": "tts-1", "voice": args.voice, "input": text}
    if args.reference:
        with open(args.reference, "rb") as f:
            body["referenceAudio"] = base64.b64encode(f.read()).decode()

    root = args.endpoint.rstrip("/")
    req = urllib.request.Request(
        root + "/v1/audio/speech", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            audio = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"tts failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.outfile != "-":
        with open(args.outfile, "wb") as f:
            f.write(audio)
    else:
        sys.stdout.buffer.write(audio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
