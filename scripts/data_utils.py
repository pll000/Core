"""Portable CoRe annotations. Image paths are relative to the data directory."""
import hashlib
import json
import math
import re
from pathlib import Path

TASKS = ('counting', 'depth', 'distance', 'relation')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSON') from exc


def image_path(data_root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'images':
        raise ValueError(f'Expected images/... relative path: {relative}')
    resolved = (Path(data_root) / path).resolve()
    resolved.relative_to(Path(data_root).resolve())
    return resolved


def validate_row(row, data_root=None):
    if row.get('task_type') not in TASKS:
        raise ValueError('Unknown or missing task_type')
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('Missing sample id')
    messages = row.get('messages', [])
    if not messages or any(m['role'] not in ('system', 'user') for m in messages):
        raise ValueError('Model inputs must contain only system/user messages; answers belong in solution')
    questions = [m['content'] for m in messages if m['role'] == 'user']
    if not questions or row.get('question_text') != questions[-1]:
        raise ValueError('question_text must preserve the last user question exactly')
    choices = 'ABC' if row['task_type'] in ('depth', 'distance') else 'ABCD'
    if not re.fullmatch(rf'<answer>[{choices}]{{3}}</answer>', row.get('solution', '')):
        raise ValueError('Expected exactly three task-appropriate answer letters')
    if len(row.get('images', [])) != 3:
        raise ValueError('Expected three ordered images')
    for relative in row['images']:
        path = image_path(data_root or '.', relative)
        if data_root is not None and not path.is_file():
            raise FileNotFoundError(path)
    if row['task_type'] == 'counting':
        for index in (1, 2, 3):
            count = row.get('image_label', {}).get(f'image_{index}', {}).get('count')
            if not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0:
                raise ValueError('Counting requires image_label.image_N.count for all three images')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
