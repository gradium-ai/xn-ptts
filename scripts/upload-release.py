#!/usr/bin/env python3
"""Resume asset uploads without changing files already attached to a release."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    for file in args.files:
        if not file.is_file():
            parser.error(f'missing asset: {file}')
    if len({file.name for file in args.files}) != len(args.files):
        parser.error('asset filenames must be unique')
    release = json.loads(subprocess.check_output(
        ['gh', 'release', 'view', args.tag, '--json', 'assets'], text=True,
    ))
    existing = {asset['name'] for asset in release['assets']}
    missing = []
    with tempfile.TemporaryDirectory() as temp:
        for file in args.files:
            if file.name not in existing:
                missing.append(file)
                continue
            subprocess.run([
                'gh', 'release', 'download', args.tag, '--pattern', file.name, '--dir', temp,
            ], check=True)
            if sha256(Path(temp) / file.name) != sha256(file):
                parser.error(f'{file.name} already exists with different contents; release assets are immutable')
            print(f'Already uploaded: {file.name}')
    # Verify all existing assets before uploading any missing ones.
    if missing:
        subprocess.run(['gh', 'release', 'upload', args.tag, *map(str, missing)], check=True)


if __name__ == '__main__':
    main()
