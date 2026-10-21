"""Device-side failure checks for phonon_benchmark; supply a valid local bundle."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--lib-dir', type=Path, required=True)
    parser.add_argument('--text-library', type=Path, required=True)
    parser.add_argument('--soc-model', default='SC8380XP')
    args = parser.parse_args()
    env = dict(os.environ, PATH=str(args.lib_dir.resolve()) + os.pathsep + os.environ['PATH'])
    with tempfile.TemporaryDirectory(prefix='phonon-strict-htp-') as directory:
        base = [str(args.runner.resolve()), '--bundle', str(args.bundle.resolve()),
                '--lib-dir', str(args.lib_dir.resolve()), '--text-library', str(args.text_library.resolve()),
                '--soc-model', args.soc_model, '--lang', 'en', '--text', 'This test must fail.',
                '--repeats', '1', '--out', str(Path(directory) / 'negative')]
        wrong_soc = base.copy()
        wrong_soc[wrong_soc.index('--soc-model') + 1] = 'UNSUPPORTED'
        missing_library = base.copy()
        missing_library[missing_library.index('--lib-dir') + 1] = str(Path(directory) / 'missing-runtime')
        tests = [
            ('cpu', base + ['--backend', 'cpu'], 'unknown argument: --backend'),
            ('gpu', base + ['--backend', 'gpu'], 'unknown argument: --backend'),
            ('soc', wrong_soc, 'no compiled QNN model'),
            ('runtime', missing_library, 'dlopen'),
        ]
        for name, command, expected in tests:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
            if result.returncode == 0 or expected not in result.stderr:
                raise RuntimeError(f'{name}: unexpected result {result.returncode}: {result.stderr}')
            if list(Path(directory).glob('negative*.wav')):
                raise RuntimeError(f'{name}: failure path wrote audio')
            print(f'{name}: rejected as expected')


if __name__ == '__main__':
    main()
