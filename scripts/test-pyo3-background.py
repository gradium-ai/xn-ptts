# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "numpy",
#    "sphn",
# ]
# ///
"""Test the background (streaming) generation of the `ptts` PyO3 bindings.

Loads a model from a local `config.json` (the model weights, tokenizer and
config are expected to sit next to it), registers a voice from a precomputed
voice safetensors, then:

1. runs a streaming generation via `generate_bt`, iterating over the audio
   chunks as they get decoded and writing the concatenation to a WAV file;
2. runs a second generation that gets cancelled after the first chunk, and
   checks that the receiver stops early.

Build/install the extension first, e.g. from the repo root:

    maturin develop --manifest-path ptts-pyo3/Cargo.toml

Then:

    python scripts/test-pyo3-background.py --config path/to/config.json --voice path/to/voice.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sphn

import ptts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the model config.json file"
    )
    parser.add_argument(
        "--voice", required=True, help="Path to a precomputed voice safetensors file"
    )
    parser.add_argument(
        "--text",
        default="Hello world, this is a test of the pocket text to speech model.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "-o", "--output", default="out-bt.wav", help="Output WAV path (default: out-bt.wav)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Sampling temperature"
    )
    parser.add_argument(
        "--seed", type=int, default=4242424242424242, help="Random seed"
    )
    parser.add_argument(
        "--quant",
        default=None,
        help="Optional CPU quantization, e.g. q8_0, q4k (default: unquantized f32)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    voice_path = Path(args.voice)
    if not voice_path.is_file():
        print(f"Error: voice not found: {voice_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {config_path} (quant={args.quant or 'f32'})")
    model = ptts.load_model(
        temperature=args.temperature,
        config=str(config_path),
        quant=args.quant,
    )
    sample_rate = model.sample_rate()
    print(f"Model loaded, sample rate {sample_rate} Hz")

    voice_name = voice_path.stem
    model.add_voice(voice_name, str(voice_path))
    state = model.get_state_for_voice(voice_name)

    # 1. Streaming generation: iterate over chunks as they get decoded.
    print(f"Generating audio (streaming) for: {args.text!r}")
    start = time.time()
    chunks = []
    recv = state.clone().generate_bt(
        args.text, temperature=args.temperature, seed=args.seed
    )
    for i, chunk in enumerate(recv):
        chunk = np.asarray(chunk, dtype=np.float32)
        elapsed = time.time() - start
        print(
            f"  chunk {i:3d}: {chunk.shape[0]:6d} samples "
            f"({chunk.shape[0] / sample_rate:.3f}s) at t={elapsed:.2f}s"
        )
        chunks.append(chunk)

    if not chunks:
        print("Error: no audio chunks received", file=sys.stderr)
        sys.exit(1)
    pcm = np.concatenate(chunks)
    duration = pcm.shape[0] / sample_rate
    sphn.write_wav(args.output, pcm, sample_rate=sample_rate)
    print(f"Wrote {duration:.2f}s of audio to {args.output} ({len(chunks)} chunks)")

    # 2. Cancellation: stop after the first chunk and drain what remains.
    print("Testing cancellation after the first chunk")
    recv = state.clone().generate_bt(
        args.text, temperature=args.temperature, seed=args.seed
    )
    first = next(recv)
    recv.cancel()
    remaining = sum(1 for _ in recv)
    print(f"  received 1 + {remaining} chunks after cancel")
    if 1 + remaining >= len(chunks):
        print("Error: cancel did not stop the generation early", file=sys.stderr)
        sys.exit(1)
    assert first is not None
    print("Cancellation OK")


if __name__ == "__main__":
    main()
