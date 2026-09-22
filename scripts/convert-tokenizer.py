# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "tokenizers",
#    "sentencepiece",
#    "protobuf",
# ]
# ///
"""Convert a SentencePiece .model file to a HuggingFace tokenizers JSON file."""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Metaspace as MetaspaceDecoder
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import Unigram
from tokenizers.normalizers import Prepend
from tokenizers.pre_tokenizers import Metaspace

# Control and unknown pieces are prefixed with this in the HF vocab. SentencePiece keeps
# them out of the segmentation lattice; HF's Unigram has no such notion, and because this
# vocabulary spells "<" and ">" as byte-fallback pieces there is no alternative path over
# "<s>", so a prompt containing one would encode to the control id. A score penalty cannot
# fix that (HF derives the fallback score from the vocab minimum), but a NUL prefix makes
# the piece unmatchable while every id keeps its position. Nothing in ptts resolves a token
# by name, and the pieces stay registered as special so the decoder still skips them.
SENTINEL = "\u0000"

# Includes inputs that exercise SentencePiece's add_dummy_prefix behavior
# (leading whitespace, whitespace-only) to catch regressions in the prepend logic.
TEST_SENTENCES = [
    "Hello, world!",
    "£",
    "café",
    "日本語",
    "14½-13½",
    "",
    " ",
    "  hello  ",
    " Hello, world!",
]


def convert(model_path: str, output_path: str) -> None:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    control = {i for i in range(sp.get_piece_size()) if sp.is_control(i) or sp.is_unknown(i)}
    vocab = [
        ((SENTINEL if i in control else "") + sp.id_to_piece(i), sp.get_score(i))
        for i in range(sp.get_piece_size())
    ]

    tokenizer = Tokenizer(Unigram(vocab, unk_id=sp.unk_id(), byte_fallback=True))
    # SentencePiece's `add_dummy_prefix` always prepends a space before encoding,
    # so " hello" becomes "  hello" and tokenizes to [▁, ▁hello]. HF's Metaspace
    # `prepend_scheme="always"` only prepends when the input doesn't already start
    # with `▁`, so it would emit just [▁hello] for the same input. Adding the
    # `Prepend` normalizer first restores the unconditional prepend (the normalizer
    # is a no-op on empty strings, preserving SP's `encode("") == []` behavior).
    tokenizer.normalizer = Prepend(prepend="▁")
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="always")
    tokenizer.decoder = DecoderSequence(
        [MetaspaceDecoder(prepend_scheme="always"), ByteFallback(), Fuse()]
    )

    # Register control/unknown tokens as special so the decoder skips them.
    for i in sorted(control):
        tokenizer.add_special_tokens(
            [AddedToken(SENTINEL + sp.id_to_piece(i), special=True)]
        )

    # Sanity check before writing anything: a tokenizer.json that disagrees with the
    # .model yields plausible audio from the wrong ids, which is exactly what `ptts`
    # refuses to do silently -- so a mismatch has to fail, not warn. A pipeline with
    # nobody reading stderr would otherwise upload a broken tokenizer.
    pieces = [sp.id_to_piece(i) for i in sorted(control)]
    checks = [(t, True) for t in TEST_SENTENCES]
    checks += [(t, False) for p in pieces for t in (p, f"Mixed {p} tag")]

    mismatches = 0
    for test, verbose in checks:
        sp_encoded = sp.encode(test, out_type=int)
        hf_encoded = tokenizer.encode(test).ids
        if verbose:
            print(f"SentencePiece: '{test}' -> {sp_encoded} -> '{sp.decode(sp_encoded)}'")
            print(f"HuggingFace:   '{test}' -> {hf_encoded} -> '{tokenizer.decode(hf_encoded)}'")
        if sp_encoded != hf_encoded:
            mismatches += 1
            print(f"WARNING: token ids differ for '{test}'!", file=sys.stderr)
    print(f"Checked {len(pieces)} control pieces as literal input text")

    if mismatches:
        print(
            f"Error: {mismatches}/{len(checks)} test sentences tokenize "
            f"differently; not writing {output_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer.save(output_path)
    print(f"Saved tokenizer to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to the SentencePiece .model file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for tokenizer.json (default: same directory as input)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(model_path.with_name("tokenizer.json"))
    convert(str(model_path), output_path)


if __name__ == "__main__":
    main()
