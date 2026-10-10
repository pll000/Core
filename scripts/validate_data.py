"""Validate every annotation, split, image checksum, and optionally image decoding."""
import argparse
from collections import Counter
import json
from pathlib import Path

from data_utils import image_path, read_jsonl, sha256, validate_row, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--check-images', action='store_true', help='Decode/verify images using Pillow')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-disjoint-triplets', action='store_true',
                        help='Fail if the recovered split has any shared ordered image triplet')
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    report, used_images, id_sets, content_sets, triplet_sets = {}, set(), {}, {}, {}
    triplet_ids = {}
    metadata = {}
    for name, info in manifest.get('metadata', {}).items():
        path = root / name
        if sha256(path) != info['sha256']:
            raise ValueError(f'Metadata checksum mismatch: {name}')
        task = Path(name).stem
        items = json.loads(path.read_text())
        if len(items) != info['records']:
            raise ValueError(f'Metadata count mismatch: {name}')
        for item in items:
            sample = f'sample_{item["sample_id"]:06d}' if task == 'relation' else item['sample_id']
            key = task + '/' + sample
            if key in metadata:
                raise ValueError(f'Duplicate metadata id: {key}')
            metadata[key] = item
    for name, info in manifest['splits'].items():
        path = root / name
        if sha256(path) != info['sha256']:
            raise ValueError(f'Annotation checksum mismatch: {name}')
        ids, counts, contents, triplets = set(), Counter(), set(), set()
        from collections import defaultdict
        triplet_ids[name] = defaultdict(list)
        for row in read_jsonl(path):
            validate_row(row, root)
            if metadata:
                item = metadata[row['id']]
                answers = item['final_answer'] if row['task_type'] == 'distance' else item['correct_answers']
                if isinstance(answers, list):
                    answers = ''.join(answers)
                if row['solution'] != '<answer>' + answers + '</answer>':
                    raise ValueError(f'Answer/metadata mismatch: {row["id"]}')
                if row['task_type'] == 'counting':
                    for index in (1, 2, 3):
                        key = f'image_{index}'
                        if row['image_label'][key]['count'] != item[key]['count']:
                            raise ValueError(f'Count metadata mismatch: {row["id"]}/{key}')
            if row['id'] in ids:
                raise ValueError(f'Duplicate sample id in {name}: {row["id"]}')
            ids.add(row['id'])
            counts[row['task_type']] += 1
            used_images.update(row['images'])
            hashes = tuple(manifest['images'][p]['sha256'] for p in row['images'])
            contents.update(hashes)
            triplets.add(hashes)
            triplet_ids[name][hashes].append(row['id'])
        if len(ids) != info['records'] or dict(counts) != info['tasks']:
            raise ValueError(f'Split count mismatch: {name}')
        id_sets[name], content_sets[name], triplet_sets[name] = ids, contents, triplets
        report[name] = dict(counts)
    if used_images != set(manifest['images']):
        raise ValueError('Image manifest does not match annotations')
    size_counts = Counter()
    for relative, info in manifest['images'].items():
        path = image_path(root, relative)
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError(f'Image checksum mismatch: {relative}')
        if args.check_images:
            from PIL import Image
            with Image.open(path) as img:
                size_counts[max(img.size)] += 1
                img.verify()
    overlap = {}
    for name in report:
        if name == 'train.jsonl':
            continue
        overlap[name] = {
            'shared_sample_ids': len(id_sets['train.jsonl'] & id_sets[name]),
            'shared_image_contents': len(content_sets['train.jsonl'] & content_sets[name]),
            'shared_ordered_image_triplets': len(triplet_sets['train.jsonl'] & triplet_sets[name])}
        overlap[name]['matching_sample_ids'] = [
            {'train': triplet_ids['train.jsonl'][triplet], 'test': triplet_ids[name][triplet]}
            for triplet in sorted(triplet_sets['train.jsonl'] & triplet_sets[name])]
    duplicates = {name: [ids for ids in mapping.values() if len(ids) > 1]
                  for name, mapping in triplet_ids.items()}
    has_overlap = any(v['shared_sample_ids'] or v['shared_ordered_image_triplets'] for v in overlap.values())
    findings = has_overlap or any(duplicates.values())
    result = {'status': 'verified_with_findings' if findings else 'passed',
              'integrity': 'passed', 'splits': report, 'image_files': len(used_images),
              'metadata_records': len(metadata),
              'image_long_edge_pixels': dict(size_counts), 'train_test_overlap': overlap,
              'duplicate_ordered_image_triplets': duplicates}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_disjoint_triplets and has_overlap:
        raise SystemExit('Ordered image triplets overlap across splits; see report. Source data was preserved.')


if __name__ == '__main__':
    main()
