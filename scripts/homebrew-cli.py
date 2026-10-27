#!/usr/bin/env python3
"""Generate a Homebrew formula from the exact CLI archives being released."""

import argparse
import hashlib
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = {
    "macos": {"arm": "aarch64-apple-darwin", "intel": "x86_64-apple-darwin"},
    "linux": {"arm": "aarch64-unknown-linux-gnu", "intel": "x86_64-unknown-linux-gnu"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    base = f"https://github.com/gradium-ai/xn-ptts/releases/download/v{version}"
    lines = [
        "class Ptts < Formula",
        '  desc "Generate speech locally with Phonon"',
        '  homepage "https://github.com/gradium-ai/xn-ptts"',
        f'  version "{version}"',
        '  license any_of: ["MIT", "Apache-2.0"]',
    ]
    for os_name, architectures in PLATFORMS.items():
        lines.extend(["", f"  on_{os_name} do"])
        if os_name == "macos":
            lines.append("    depends_on macos: :sequoia")
        for arch, target in architectures.items():
            filename = f"ptts-{version}-{target}.tar.gz"
            digest = hashlib.sha256((args.directory / filename).read_bytes()).hexdigest()
            lines.extend([
                f"    on_{arch} do",
                f'      url "{base}/{filename}"',
                f'      sha256 "{digest}"',
                "    end",
            ])
        lines.append("  end")
    lines.extend([
        "",
        "  def install",
        '    bin.install "ptts"',
        "  end",
        "",
        "  def caveats",
        "    <<~EOS",
        "      Intel/x86_64 downloads require an x86-64-v3 CPU (AVX2, FMA, F16C).",
        "      Linux downloads require glibc 2.35 or later.",
        "      ptts downloads the default model on first use; --model picks another. --lang is required.",
        "    EOS",
        "  end",
        "",
        "  test do",
        f'    assert_equal "ptts {version}\\n", shell_output("#{{bin}}/ptts --version")',
        "  end",
        "end",
        "",
    ])
    (args.directory / "ptts.rb").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
