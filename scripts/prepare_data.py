"""Resolve portable image paths for ms-swift without changing published data."""
import argparse
import json
from pathlib import Path

from data_utils import image_path, read_jsonl, validate_row


def prepare(source, data_root, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Output must differ from the published annotation file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    count = 0
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            for row in read_jsonl(source):
                validate_row(row, data_root)
                row['images'] = [str(image_path(data_root, p)) for p in row['images']]
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
        if not count:
            raise ValueError('Empty dataset')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = prepare(args.input, args.data_root, args.output)
    print(f'Prepared {count} records: {args.output}')


if __name__ == '__main__':
    main()
