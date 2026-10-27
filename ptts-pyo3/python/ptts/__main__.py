"""`ptts` -- synthesize from the command line.

    ptts --lang en "hello world" -o out.wav
    python -m ptts --lang en "hello world" -o out.wav

Every option here maps to one `TTS` argument, so the module doubles as a
worked example of the API.
"""

from __future__ import annotations

import argparse
import sys

from . import TTS, __version__, available_devices, available_quants, build_info, catalog_voices


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # Usage lines should name the spelling the reader actually typed. argparse's own
        # default gets `__main__.py` for `python -m ptts`, which names nothing runnable.
        prog="python -m ptts" if sys.argv[0].endswith("__main__.py") else "ptts",
        description="Generate speech from text using Phonon.",
    )
    p.add_argument("text", nargs="?", help="text to synthesize")
    p.add_argument("-o", "--output", default="out.wav", help="output WAV path (default: out.wav)")
    p.add_argument(
        "-v",
        "--voice",
        help="a voice of the checkpoint or of the voice catalog, by name or catalog voice ID;"
        " defaults to the checkpoint's own",
    )
    p.add_argument(
        "-l",
        "--lang",
        help="required: normalize text as en, fr, de, es, pt, or none to skip",
    )
    p.add_argument(
        "-r",
        "--rewrites",
        help="word rewrites to run: default, all, none, or a comma-separated list such as"
        " numbers,emails (default: default, every rule but phones, times and dates)",
    )
    p.add_argument(
        "-m",
        "--model",
        help="local model directory or config.json, or a Hugging Face repo as hf://owner/name or"
        " hf://owner/name@revision (default: the default model, downloaded on first use)",
    )
    p.add_argument(
        "-d", "--device", help=f"one of {', '.join(['auto', *available_devices()])} (default: auto)"
    )
    p.add_argument(
        "--voice-catalog",
        help="where --voice is looked up when the checkpoint lacks it: hf://owner/name[@revision], or a"
        " local voices.jsonl or folder holding one (default: the default voice catalog)",
    )
    p.add_argument("-q", "--quant", help=f"weight format: {', '.join(available_quants())}")
    p.add_argument("-t", "--temperature", type=float, help="sampling temperature")
    p.add_argument("-s", "--seed", type=int, help="sampling seed, for a reproducible run")
    p.add_argument(
        "-c",
        "--condition",
        action="append",
        metavar="NAME=VALUE",
        help="set one of the checkpoint's conditioners, e.g. duration_delta=0.5; repeatable",
    )
    p.add_argument("--threads", type=int, help="CPU threads for tensor ops")
    p.add_argument("--list-voices", action="store_true", help="list the voices of --voice-catalog, then exit")
    p.add_argument("--build-info", action="store_true", help="print the build configuration, then exit")
    p.add_argument("--version", action="version", version=f"ptts {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.build_info:
        for key, value in build_info().items():
            print(f"{key}: {value}")
        return 0

    if args.list_voices:
        try:
            voices = catalog_voices(args.voice_catalog)
        except Exception as e:  # noqa: BLE001 -- a CLI reports, it does not re-raise
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        for v in voices:
            print("\t".join(v[key] or "-" for key in ("name", "language", "voice_id", "description")))
        return 0

    if not args.text:
        parser.error("nothing to say: pass some text, or --list-voices")

    if args.lang is None:
        # Text is normalized before it is tokenized, and the spoken forms of `@`, `+` and `=`
        # differ per language, so there is nothing safe to guess on the caller's behalf.
        parser.error("--lang is required: one of en, fr, de, es, pt, or none to skip it")

    if args.threads is not None:
        # After the arguments are checked, before the model loads: this sizes a global pool
        # that is built once.
        from . import set_num_threads

        set_num_threads(args.threads)

    kwargs: dict[str, object] = {"lang": args.lang}
    for name in (
        "model",
        "device",
        "quant",
        "voice",
        "seed",
        "temperature",
        "rewrites",
        "voice_catalog",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.condition:
        conditions = {}
        for condition in args.condition:
            name, sep, value = condition.partition("=")
            if not sep:
                parser.error(f"--condition takes NAME=VALUE, got {condition!r}")
            conditions[name] = value
        kwargs["conditions"] = conditions
    try:
        tts = TTS(**kwargs)  # type: ignore[arg-type]
        seconds = tts.save(args.output, args.text)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 -- a CLI reports, it does not re-raise
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"wrote {args.output} ({seconds:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
