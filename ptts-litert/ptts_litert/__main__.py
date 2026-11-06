"""`ptts-litert export | compile | check`. Each command's module explains what it does."""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="ptts-litert", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="write a LiteRT bundle from a checkpoint (needs the export extra)")
    e.add_argument("out", type=Path, help="bundle directory to write")
    e.add_argument(
        "--dir", type=Path, required=True, help="checkpoint directory with config.json and safetensors weights"
    )
    e.add_argument("--weights", default="model.safetensors", help="weights file inside --dir")
    e.add_argument(
        "--voices", type=Path, help="a directory of voice .safetensors files, instead of the checkpoint's own"
    )
    e.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a conditioner, e.g. padding_bonus=0.5; repeatable. Fixed in host.safetensors",
    )
    e.add_argument("--tokenizer", type=Path, help="a tokenizer.json, for a checkpoint that ships none")
    e.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="rows per prefill call; with the longest voice and its frames, it sizes the KV cache",
    )
    e.add_argument("--temperature", type=float, default=0.3, help="sampling temperature recorded for the runtime")

    c = sub.add_parser("compile", help="compile a bundle's model for NPUs (needs the compile extra, Linux x86_64)")
    c.add_argument("bundle", type=Path)
    c.add_argument("targets", nargs="+", metavar="VENDOR:SOC", help="e.g. qualcomm:SM8750 mediatek:MT6991")

    k = sub.add_parser("check", help="run a bundle on LiteRT's CPU runtime against dump_reference output")
    k.add_argument("bundle", type=Path)
    k.add_argument("dumps", type=Path, nargs="+", help="directories written by ptts/examples/dump_reference.rs")
    k.add_argument("--teacher", action="store_true", help="feed each step ptts's previous latent, not the bundle's own")
    k.add_argument("--threads", type=int, default=4)
    k.add_argument("--audio-out", type=Path, help="write each dump's audio as a WAV here")

    args = ap.parse_args()
    if args.command == "export":
        from .export import export

        export(args)
    elif args.command == "compile":
        from .compile import compile

        compile(args)
    else:
        from .check import check

        check(args)


if __name__ == "__main__":
    main()
