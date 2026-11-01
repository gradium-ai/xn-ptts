#!/usr/bin/env python3
"""Download the pinned 31m Q8 checkpoint and 17 distinct English benchmark voices.

Run with: uv run --no-project --with huggingface_hub scripts/prepare-voice-bench.py DIR
Keep DIR outside the repository. No model or voice weights belong in Git.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODEL = "gradium/phonon-31m-en"
MODEL_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"
VOICES = "gradium/phonon-voices"
VOICE_REVISION = "da6c9ec0d47be8df5c87c730a3aa97187f3229c3"
MODEL_FILES = ["config.json", "tokenizer.json", "default-voice.safetensors", "model.q8.gguf"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    snapshot_download(
        MODEL, revision=MODEL_REVISION, allow_patterns=MODEL_FILES,
        local_dir=root / "phonon-31m-en",
    )
    catalog = Path(hf_hub_download(VOICES, "voices.jsonl", revision=VOICE_REVISION))
    english = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    selected = [voice for voice in english if voice.get("language") == "en"][:17]
    assert len(selected) == 17, "benchmark needs 17 distinct English voices"
    for voice in selected:
        hf_hub_download(
            VOICES, voice["filename"], revision=VOICE_REVISION, local_dir=root / "voices",
        )
    (root / "voices" / "voices.jsonl").write_text(
        "".join(json.dumps(voice) + "\n" for voice in selected)
    )
    paths = [root / "phonon-31m-en" / name for name in MODEL_FILES]
    paths += [root / "voices" / voice["filename"] for voice in selected]
    paths.append(root / "voices" / "voices.jsonl")
    manifest = {
        "model": MODEL, "model_revision": MODEL_REVISION,
        "voices": VOICES, "voice_revision": VOICE_REVISION,
        "voice_names": [voice["name"] for voice in selected],
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
