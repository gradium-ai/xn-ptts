#!/usr/bin/env python3
"""Check that WAV files say the words they were made from.

    python scripts/check-speech.py --text "What every file should say." DIR...

Every `.wav` under each DIR is transcribed with Whisper and compared word by word with
`--text`. A file passes when it is 24 kHz mono 16-bit, not silent, of a plausible length for the
text, and its transcript is close enough to the text. A model with the wrong tokenizer, broken
weights or a broken kernel still makes speech-like sound, so reading the words is the only check
that catches it. Prints a table, also to the GitHub job summary when there is one, and exits 1 if
any file fails or there are none.

Needs `faster-whisper` (`pip install faster-whisper`), which downloads its model on first use.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()


def word_error_rate(reference: list[str], heard: list[str]) -> float:
    """Word-level edit distance over the length of the reference."""
    previous = list(range(len(heard) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(heard, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1] / max(len(reference), 1)


def read_wav(path: Path) -> tuple[np.ndarray | None, str | None]:
    """The samples in [-1, 1], or what is wrong with the file."""
    try:
        with wave.open(str(path)) as w:
            shape = (w.getframerate(), w.getnchannels(), w.getsampwidth())
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        return None, f"not a WAV file: {e}"
    if shape != (SAMPLE_RATE, 1, 2):
        return None, f"not 24 kHz mono 16-bit: {shape[0]} Hz, {shape[1]} channel(s), {8 * shape[2]}-bit"
    return np.frombuffer(frames, "<i2") / 32768, None


def to_16k(pcm: np.ndarray) -> np.ndarray:
    """Resample to the 16 kHz Whisper takes, in the frequency domain. Done here rather than by
    faster-whisper, whose file decoding breaks with some PyAV releases."""
    n = len(pcm) * 16000 // SAMPLE_RATE
    return (np.fft.irfft(np.fft.rfft(pcm)[: n // 2 + 1], n) * n / len(pcm)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="directories holding the .wav files")
    parser.add_argument("--text", required=True, help="what every file should say")
    parser.add_argument("--max-wer", type=float, default=0.25, help="largest word error rate that passes")
    parser.add_argument("--whisper", default="base.en", help="faster-whisper model name or path")
    args = parser.parse_args()

    files = sorted((p.relative_to(d), p) for d in args.dirs for p in d.rglob("*.wav"))
    if not files:
        print(f"no .wav files under {', '.join(map(str, args.dirs))}", file=sys.stderr)
        return 1

    from faster_whisper import WhisperModel

    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    reference = words(args.text)
    # Speech runs at roughly two to four words a second; these bounds leave room either side
    # and still catch a stream that never stops, or one that stops after a word.
    shortest, longest = len(reference) / 6, len(reference) / 1.2

    rows, failed = [], 0
    for name, path in files:
        pcm, problem = read_wav(path)
        seconds, peak, wer, heard = "-", "-", "-", "-"
        if pcm is not None and len(pcm):
            segments, _ = asr.transcribe(to_16k(pcm), language="en", beam_size=5, condition_on_previous_text=False)
            heard = " ".join(s.text.strip() for s in segments)
            error = word_error_rate(reference, words(heard))
            length, loudest = len(pcm) / SAMPLE_RATE, float(np.abs(pcm).max())
            if loudest < 0.05:
                problem = "silent"
            elif not shortest <= length <= longest:
                problem = f"length outside {shortest:.1f}-{longest:.1f} s"
            elif error > args.max_wer:
                problem = f"word error rate above {args.max_wer:.0%}"
            seconds, peak, wer = f"{length:.2f}", f"{loudest:.2f}", f"{error:.0%}"
        elif pcm is not None:
            problem = "empty"
        failed += problem is not None
        rows.append((str(name), seconds, peak, wer, heard or "-", problem or "ok"))

    header = ("file", "seconds", "peak", "WER", "heard", "result")
    table = "\n".join(
        "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"
        for cells in [header, ("---",) * len(header), *rows]
    )
    summary = f"Expected: {args.text}\n\n{table}\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### Speech check\n\n{summary}\n")
    if failed:
        print(f"{failed} of {len(rows)} files failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
