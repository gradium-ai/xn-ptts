# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.24", "pyarrow"]
# ///
"""Download public English test sets from the Hugging Face Hub, laid out for `wer.py`.

    uv run scripts/fetch_evalsets.py <root> [set ...]

Writes `<root>/<set>/<id>.<ext>` (the audio as published, not re-encoded) and `<id>.txt` (the
reference) for each set named, or every set below without names. Downloads go to a temporary
cache that is deleted afterwards.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ESB = "hf-audio/open-asr-leaderboard"

# name: (repo, files to download, how the rows are stored)
SETS = {
    "voxpopuli_aa": ("ArtificialAnalysis/VoxPopuli-Cleaned-AA", ["*.jsonl", "audio/*"], "jsonl"),
    "earnings22_aa": ("ArtificialAnalysis/Earnings22-Cleaned-AA", ["*.jsonl", "audio/*"], "jsonl"),
    "pipecat": ("pipecat-ai/stt-benchmark-data", ["data/*.parquet"], "parquet"),
    "fleurs_en": ("nithinraok/asr-leaderboard-datasets", ["data/fleurs/en_test.parquet"], "parquet"),
    # The Open ASR Leaderboard's English test sets (ESB).
    "esb_librispeech_clean": (ESB, ["librispeech/test.clean-*.parquet"], "parquet"),
    "esb_librispeech_other": (ESB, ["librispeech/test.other-*.parquet"], "parquet"),
    "esb_ami": (ESB, ["ami/test-*.parquet"], "parquet"),
    "esb_earnings22": (ESB, ["earnings22/test-*.parquet"], "parquet"),
    "esb_voxpopuli": (ESB, ["voxpopuli/test-*.parquet"], "parquet"),
    "esb_tedlium": (ESB, ["tedlium/test-*.parquet"], "parquet"),
}


def extension(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    raise ValueError(f"unknown audio format {data[:4]!r}")


def from_jsonl(snapshot: Path, out: Path) -> int:
    """Artificial Analysis layout: a JSONL of `file_name` and `transcript`, the files in `audio/`."""
    count = 0
    for meta in snapshot.glob("*.jsonl"):
        for line in meta.read_text().splitlines():
            row = json.loads(line)
            audio = snapshot / "audio" / Path(row["file_name"]).name
            shutil.copyfile(audio, out / audio.name)
            (out / audio.name).with_suffix(".txt").write_text(row["transcript"])
            count += 1
    return count


def from_parquet(snapshot: Path, out: Path) -> int:
    """Rows with an `audio` struct of bytes and a transcript column."""
    count = 0
    for path in sorted(snapshot.rglob("*.parquet")):
        table = pq.read_table(path)
        names = table.column_names
        text = next(n for n in ("text", "transcription", "transcript") if n in names)
        key = next((n for n in ("sample_id", "id", "file_name") if n in names), None)
        for i, row in enumerate(table.to_pylist()):
            data = row["audio"]["bytes"]
            if key == "file_name":
                stem = Path(row[key]).stem
            else:
                stem = str(row[key]).replace("/", "_") if key else f"{i:05d}"
            (out / f"{stem}.{extension(data)}").write_bytes(data)
            (out / f"{stem}.txt").write_text(row[text])
            count += 1
    return count


def main() -> None:
    root = Path(sys.argv[1])
    for name in sys.argv[2:] or SETS:
        repo, patterns, kind = SETS[name]
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as cache:
            snapshot = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, cache_dir=cache))
            count = from_jsonl(snapshot, out) if kind == "jsonl" else from_parquet(snapshot, out)
        print(f"{name}: {count} files in {out}", flush=True)


if __name__ == "__main__":
    main()
