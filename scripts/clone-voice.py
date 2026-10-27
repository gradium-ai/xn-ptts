# /// script
# requires-python = ">=3.9"
# dependencies = [
#    "requests",
# ]
# ///
"""Clone a voice with the Gradium API and download it as a Phonon voice file.

Uploads a recording of the speaker (10 seconds of clear speech are used, from --start on),
waits for Gradium to make the voice, and writes its voice file. The voice is saved in your
Gradium account: download-voice.py fetches it again by the ID this prints. Needs a Gradium API
key in GRADIUM_API_KEY. Clone only voices you have the speaker's consent to use.

    uv run scripts/clone-voice.py speaker.wav --lang en -o voice.safetensors
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.gradium.ai/api"
# Every Phonon checkpoint uses the same voice files, so any model name the API knows will do.
MODEL = "default"
# Seconds to wait for each request.
TIMEOUT = 60


def check(r: requests.Response, *args, **kwargs) -> None:
    """Exits with the API's own message when a request fails."""
    if not r.ok:
        try:
            detail = r.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = r.text
        sys.exit(f"Gradium API error {r.status_code}: {detail}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="a recording of the speaker (wav, mp3, flac, ...)")
    p.add_argument("--lang", required=True, choices=["en", "fr", "de", "es", "pt"], help="the recording's language")
    p.add_argument("--name", help="the voice's name in your account (default: the audio file's name)")
    p.add_argument("--start", type=float, default=0.0, help="where the speech starts in the recording, in seconds")
    p.add_argument("-o", "--out", type=Path, help="output file (default: VOICE_ID.safetensors)")
    args = p.parse_args()

    key = os.environ.get("GRADIUM_API_KEY")
    if not key:
        sys.exit("set GRADIUM_API_KEY to your Gradium API key")
    s = requests.Session()
    s.headers["x-api-key"] = key
    s.hooks["response"] = check

    with args.audio.open("rb") as f:
        form = {"name": args.name or args.audio.stem, "language": args.lang, "start_s": args.start}
        r = s.post(f"{API}/voices/", files={"audio_file": (args.audio.name, f)}, data=form, timeout=TIMEOUT)
    voice_id = r.json().get("uid")
    if not voice_id:
        sys.exit(f"cloning failed: {r.json().get('error')}")
    print(f"created voice {voice_id}")

    for _ in range(60):
        if s.get(f"{API}/voices/embeddings/{voice_id}/{MODEL}", timeout=TIMEOUT).json().get("found"):
            break
        time.sleep(2)
    else:
        sys.exit(f"voice {voice_id} is not ready yet: try download-voice.py later")

    r = s.get(f"{API}/voices/embeddings/{voice_id}/{MODEL}/blob", params={"as_safetensors": "1"}, timeout=TIMEOUT)
    out = args.out or Path(f"{voice_id}.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
