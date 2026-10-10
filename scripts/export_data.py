"""Export the confirmed metadata-based Count split and the other three tasks.

The source root is the parent directory containing remot/ and the image folders.
Existing output directories must be empty. No source file is modified.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil

from data_utils import read_jsonl, sha256, validate_row, write_json


def convert_row(source, task, source_root):
    row = dict(source)
    messages = source['messages']
    if 'solution' not in row:
        if messages[-1]['role'] != 'assistant':
            raise ValueError('Test record has no final reference answer')
        row['solution'] = messages[-1]['content']
        row['messages'] = messages[:-1]
    row['task_type'] = task
    row['question_text'] = [m['content'] for m in row['messages'] if m['role'] == 'user'][-1]
    row['images'] = ['images/' + Path(p).resolve().relative_to(source_root).as_posix()
                     for p in source['images']]
    row['id'] = task + '/' + Path(source['images'][0]).parent.name
    validate_row(row)
    return row


def portable_metadata(value, image_mapping):
    if isinstance(value, dict):
        return {key: portable_metadata(item, image_mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item, image_mapping) for item in value]
    if isinstance(value, str):
        if value in image_mapping:
            return image_mapping[value]
        if value.startswith('/'):
            raise ValueError(f'Unmapped absolute metadata path: {value}')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8, help='Parallel small-file copies')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    source_root, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output is not empty; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    inputs = [('train.jsonl', 'mixed_dataset_with_question_text.jsonl', None)]
    inputs += [(f'test/{task}.jsonl', f'{prefix}_sft_test.jsonl', task)
               for prefix, task in [('count', 'counting'), ('depth', 'depth'),
                                    ('distance', 'distance'), ('relation', 'relation')]]
    manifest = {'format_version': 1, 'image_path_base': 'data directory',
                'count_version': 'metadata-regenerated (remot), confirmed by author',
                'image_provenance': 'Author confirmed on 2026-10-09: these image files were used in the experiments; exported without resizing.',
                'splits': {}, 'images': {}}
    split_hashes = {}
    split_triplets = {}
    image_sources = {}
    converted_splits = {}
    # Write annotations first, then copy the deduplicated image paths concurrently.
    for destination, filename, task in inputs:
        source = source_root / 'remot' / filename
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        counts, ids, converted = Counter(), set(), []
        with target.open('w', encoding='utf-8') as stream:
            for original in read_jsonl(source):
                row = convert_row(original, task or original['task_type'], source_root)
                if row['id'] in ids:
                    raise ValueError(f'Duplicate id: {row["id"]}')
                ids.add(row['id'])
                counts[row['task_type']] += 1
                for old_path, new_path in zip(original['images'], row['images']):
                    image_sources[new_path] = old_path
                converted.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['splits'][destination] = {
            'records': sum(counts.values()), 'tasks': dict(counts),
            'sha256': sha256(target), 'source': 'remot/' + filename,
            'source_sha256': sha256(source)}
        converted_splits[destination] = converted
        print(f'{destination}: {sum(counts.values())} records', flush=True)

    def copy_image(item):
        relative, source = item
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return relative, {'sha256': sha256(target), 'bytes': target.stat().st_size}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (relative, info) in enumerate(pool.map(copy_image, image_sources.items()), 1):
            manifest['images'][relative] = info
            if index % 5000 == 0:
                print(f'Copied {index}/{len(image_sources)} images', flush=True)
    for destination, rows in converted_splits.items():
        triplets = {tuple(manifest['images'][p]['sha256'] for p in row['images']) for row in rows}
        image_hashes = {h for triplet in triplets for h in triplet}
        manifest['splits'][destination].update({
            'unique_image_contents': len(image_hashes), 'unique_ordered_image_triplets': len(triplets)})
        split_hashes[destination], split_triplets[destination] = image_hashes, triplets
    manifest['metadata'] = {}
    image_mapping = {str(Path(source).resolve()): relative for relative, source in image_sources.items()}
    for task, folder in [('counting', 'remot_count_meta'), ('depth', 'remot_depth_easy'),
                         ('distance', 'remot_distance'), ('relation', 'remot_relation_easy')]:
        source = source_root / folder / 'metadata.json'
        relative = f'metadata/{task}.json'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        records = portable_metadata(json.loads(source.read_text()), image_mapping)
        write_json(target, records)
        manifest['metadata'][relative] = {'source': f'{folder}/metadata.json', 'sha256': sha256(target),
                                          'source_sha256': sha256(source), 'records': len(records)}
    manifest['train_test_overlap'] = {
        name: {'shared_image_contents': len(hashes & split_hashes['train.jsonl']),
               'shared_ordered_image_triplets': len(split_triplets[name] & split_triplets['train.jsonl'])}
        for name, hashes in split_hashes.items() if name != 'train.jsonl'}
    manifest['image_files'] = len(manifest['images'])
    manifest['image_bytes'] = sum(x['bytes'] for x in manifest['images'].values())
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'images'}, indent=2))


if __name__ == '__main__':
    main()
