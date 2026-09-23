"""`ptts` -- synthesize from the command line.

    ptts --lang en "hello world" -o out.wav
    python -m ptts --lang en "hello world" -o out.wav

Both spellings run `main`: the first through the `ptts` console script the wheel installs,
which is also what `uvx ptts` and `pipx run ptts` invoke.

Deliberately thin: every option here maps to one `TTS` argument, so the module doubles as a
worked example of the API.
"""

from __future__ import annotations

import argparse
import sys

from . import TTS, __version__, available_devices, available_quants, build_info


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        # Usage lines should name the spelling the reader actually typed. argparse's own
        # default gets `__main__.py` for `python -m ptts`, which names nothing runnable.
        prog="python -m ptts" if sys.argv[0].endswith("__main__.py") else "ptts",
        description="Generate speech from text using Pocket TTS.",
    )
    p.add_argument("text", nargs="?", help="text to synthesize")
    p.add_argument("-o", "--output", default="out.wav", help="output WAV path (default: out.wav)")
    p.add_argument("-v", "--voice", help="voice name; defaults to the checkpoint's own")
    # Required, as on `pocket_tts` and `ptts-ws-server`, but enforced below rather than by
    # argparse: `--build-info` is the one flag that never builds a model, and it should not
    # have to name a language it has no use for.
    p.add_argument(
        "-l",
        "--lang",
        help="required: normalize text as en, fr, de, es, pt, or none to skip",
    )
    p.add_argument(
        "-m",
        "--model",
        help="Hugging Face repo id or path to a local config.json (default: the published one)",
    )
    p.add_argument(
        "-d", "--device", help=f"one of {', '.join(['auto', *available_devices()])} (default: auto)"
    )
    p.add_argument("-q", "--quant", help=f"weight format: {', '.join(available_quants())}")
    p.add_argument("-t", "--temperature", type=float, default=0.5, help="sampling temperature")
    p.add_argument("-s", "--seed", type=int, help="sampling seed, for a reproducible run")
    p.add_argument("--threads", type=int, help="CPU threads for tensor ops")
    p.add_argument("--list-voices", action="store_true", help="list the checkpoint's voices, then exit")
    p.add_argument("--build-info", action="store_true", help="print the build configuration, then exit")
    p.add_argument("--version", action="version", version=f"ptts {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.build_info:
        for key, value in build_info().items():
            print(f"{key}: {value}")
        return 0

    if args.threads is not None:
        # Before the model loads: this sizes a global pool that is built once.
        from . import set_num_threads

        set_num_threads(args.threads)

    if not args.list_voices and not args.text:
        _parser().error("nothing to say: pass some text, or --list-voices")

    if args.lang is None:
        # Text is normalized before it is tokenized, and the spoken forms of `@`, `+` and `=`
        # differ per language, so there is nothing safe to guess on the caller's behalf.
        _parser().error("--lang is required: one of en, fr, de, es, pt, or none to skip it")

    kwargs: dict[str, object] = {"temperature": args.temperature, "lang": args.lang}
    for name in ("config", "device", "quant", "voice", "seed"):
        value = getattr(args, {"config": "model"}.get(name, name))
        if value is not None:
            kwargs[name] = value

    # Errors carry their own remedy -- a gated checkpoint says how to authenticate, an unknown
    # voice lists the ones that exist -- so print the message rather than a traceback.
    try:
        tts = TTS(**kwargs)  # type: ignore[arg-type]
        if args.list_voices:
            print("\n".join(tts.voices))
            return 0
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
