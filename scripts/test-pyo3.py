# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "numpy",
#    "sphn",
# ]
# ///
"""Smoke-test the `ptts` PyO3 bindings end to end.

Loads a model from a local `config.json` (the model weights, tokenizer and
config are expected to sit next to it), registers a voice from a precomputed
voice safetensors, runs a single text generation and writes the result to a
WAV file.

Build/install the extension first, e.g. from the repo root:

    maturin develop --manifest-path ptts-pyo3/Cargo.toml

Then:

    python scripts/test_pyo3.py --config path/to/config.json --voice path/to/voice.safetensors
"""

import argparse
import sys
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
        "-o", "--output", default="out.wav", help="Output WAV path (default: out.wav)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature"
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
    print(f"Model loaded, sample rate {model.sample_rate()} Hz")

    voice_name = voice_path.stem
    model.add_voice(voice_name, str(voice_path))
    print(f"Registered voice '{voice_name}'. Available voices: {model.voices()}")

    state = model.get_state_for_voice(voice_name)
    print(f"Generating audio for: {args.text!r}")
    pcm = state.generate_audio(args.text, temperature=args.temperature, seed=args.seed)
    pcm = np.asarray(pcm, dtype=np.float32)

    duration = pcm.shape[0] / model.sample_rate()
    sphn.write_wav(args.output, pcm, sample_rate=model.sample_rate())
    print(f"Wrote {duration:.2f}s of audio to {args.output}")


if __name__ == "__main__":
    main()
