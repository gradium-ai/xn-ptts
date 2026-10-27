# /// script
# requires-python = ">=3.9"
# dependencies = [
#    "requests",
# ]
# ///
"""Download a Gradium voice as a Phonon voice file.

The voice can be one you designed or cloned in Gradium Studio, or cloned with clone-voice.py.
Needs a Gradium API key in GRADIUM_API_KEY.

    uv run scripts/download-voice.py --list
    uv run scripts/download-voice.py VOICE_ID -o voice.safetensors
"""

import argparse
import os
import sys
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
    p.add_argument("voice_id", nargs="?", help="the voice's ID, as --list or Gradium Studio shows it")
    p.add_argument("-o", "--out", type=Path, help="output file (default: VOICE_ID.safetensors)")
    p.add_argument("--list", action="store_true", help="list your voices")
    args = p.parse_args()
    if not args.list and not args.voice_id:
        p.error("give a voice ID, or --list")

    key = os.environ.get("GRADIUM_API_KEY")
    if not key:
        sys.exit("set GRADIUM_API_KEY to your Gradium API key")
    s = requests.Session()
    s.headers["x-api-key"] = key
    s.hooks["response"] = check

    if args.list:
        r = s.get(f"{API}/voices/", params={"limit": 0}, timeout=TIMEOUT)
        for v in r.json():
            print(f"{v['uid']}  {v.get('language') or '--'}  {v['name']}: {v.get('description') or ''}")
        return

    r = s.get(f"{API}/voices/embeddings/{args.voice_id}/{MODEL}/blob", params={"as_safetensors": "1"}, timeout=TIMEOUT)
    out = args.out or Path(f"{args.voice_id}.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
