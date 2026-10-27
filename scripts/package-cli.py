#!/usr/bin/env python3
"""Package a built CLI, its guide and code licenses for a desktop release."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--tag", help="Require this tag to match the workspace version")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    if args.tag and args.tag != f"v{version}":
        parser.error(f"tag {args.tag} does not match workspace version {version}")
    binary = args.binary.resolve(strict=True)
    reported = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if reported != f"ptts {version}":
        parser.error(f"binary reports {reported!r}, expected 'ptts {version}'")
    args.output.mkdir(parents=True, exist_ok=True)
    # `[package.metadata.binstall]` in ptts/Cargo.toml relies on this name and folder layout.
    name = f"ptts-{version}-{args.target}"
    windows = "windows" in args.target
    archive = args.output / f"{name}{'.zip' if windows else '.tar.gz'}"
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp) / name
        folder.mkdir()
        destination = folder / ("ptts.exe" if windows else "ptts")
        shutil.copy2(binary, destination)
        destination.chmod(0o755)
        for license_file in ("LICENSE-MIT", "LICENSE-APACHE"):
            shutil.copy2(ROOT / license_file, folder / license_file)
        shutil.copy2(ROOT / "docs/cli.md", folder / "README.md")
        if windows:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for file in sorted(folder.iterdir()):
                    output.write(file, f"{name}/{file.name}")
        else:
            with tarfile.open(archive, "w:gz") as output:
                output.add(folder, arcname=name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(f"{archive.name}.sha256").write_bytes(f"{digest}  {archive.name}\n".encode("ascii"))
    print(archive)


if __name__ == "__main__":
    main()
