"""Create separate local code and data archives; never publish them."""
import argparse
import os
from pathlib import Path
import tarfile

from data_utils import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error('Archive directory must be outside the release tree')
    output.mkdir(parents=True, exist_ok=True)
    code, data = output / 'CoRe-code.tar.gz', output / 'CoRe-data.tar.gz'
    for path in (code, data, output / 'SHA256SUMS'):
        if path.exists():
            parser.error(f'Refusing to overwrite {path}')
    excluded = {'.git', '.venv', '__pycache__', 'outputs', '.cache'}
    with tarfile.open(code, 'w:gz', compresslevel=6) as archive:
        for directory, dirs, filenames in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in excluded and not d.endswith('.egg-info')
                             and not (Path(directory) == root and d == 'data'))
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.suffix in ('.pyc', '.pyo', '.log'):
                    continue
                archive.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
    with tarfile.open(data, 'w:gz', compresslevel=6) as archive:
        archive.add(root / 'data', arcname=root.name + '/data')
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in (code, data)))
    for path in (code, data):
        print(f'{path.name}: {path.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
