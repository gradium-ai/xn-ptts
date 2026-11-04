# /// script
# requires-python = ">=3.10"
# dependencies = ["jiwer>=3", "more-itertools", "regex"]
# ///
"""Corpus WER of `transcribe` output, per evaluation set.

    uv run scripts/wer.py transcripts.tsv [more.tsv ...]

Each line of a transcript file is `<audio path>\t<text>`, as `examples/transcribe` prints. The
reference sits beside the audio: `<stem>.transcribe.json` (its `text`) or `<stem>.txt`. A file's
set is the name of its folder, so `<root>/<set>/<id>.wav` gives one WER per `<set>`.

References and hypotheses both go through Whisper's English normalizer (`whisper_norm/`, the
pinned copy the training validation uses; the `whisper-normalizer` package on PyPI is a
modified fork that scores differently), and a set's WER is its errors over its reference words
(corpus WER, not a mean of per-file WERs), as the training validation computes it. Files whose
normalized reference is empty are skipped.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
from whisper_norm.english import EnglishTextNormalizer


def reference(audio: Path) -> str:
    meta = audio.with_suffix(".transcribe.json")
    if meta.exists():
        return json.loads(meta.read_text())["text"]
    return audio.with_suffix(".txt").read_text()


def main() -> None:
    normalize = EnglishTextNormalizer()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # errors, words, files
    for tsv in sys.argv[1:]:
        for line in Path(tsv).read_text().splitlines():
            if "\t" not in line:
                continue
            path, hyp = line.split("\t", 1)
            audio = Path(path)
            ref = normalize(reference(audio))
            if not ref.split():
                continue
            out = jiwer.process_words(ref, normalize(hyp))
            c = counts[audio.parent.name]
            c[0] += out.substitutions + out.deletions + out.insertions
            c[1] += len(ref.split())
            c[2] += 1
    width = max(len(name) for name in counts)
    for name in sorted(counts):
        errors, words, files = counts[name]
        print(f"{name:<{width}}  {100 * errors / words:6.2f}%  ({files} files, {words} words)")
    mean = sum(e / w for e, w, _ in counts.values()) / len(counts)
    print(f"{'mean of sets':<{width}}  {100 * mean:6.2f}%")


if __name__ == "__main__":
    main()
