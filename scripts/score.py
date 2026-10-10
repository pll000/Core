"""Score predictions against reference annotations, including missing outputs as zero.

Answer extraction and length handling follow the recovered evaluation script.
Predictions are JSONL records containing id and prediction (the full model text).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from data_utils import read_jsonl, write_json


def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not match:
        match = re.search(r'<answer>(.*?)$', text, re.DOTALL)
    return match.group(1).strip() if match else ''


def score_prediction(solution, prediction):
    truth, answer = extract_answer(solution), extract_answer(prediction)
    if len(truth) != 3:
        raise ValueError(f'Invalid reference solution: {solution}')
    correct = sum(a == b for a, b in zip(truth, answer)) if len(answer) == 3 else 0
    return int(correct == 3), correct / 3


def score_records(records, predictions):
    references = {r['id']: r for r in records}
    if len(references) != len(records):
        raise ValueError('Duplicate reference ids')
    if set(predictions) - set(references):
        raise ValueError('Predictions contain ids outside the selected dataset')
    groups = defaultdict(list)
    for row in records:
        result = score_prediction(row['solution'], predictions.get(row['id'], ''))
        groups[row['task_type']].append(result)
        groups['all'].append(result)
    return {
        'reference_records': len(records), 'predictions': len(predictions),
        'missing_predictions': len(references.keys() - predictions.keys()),
        'metrics_percent': {
            task: {'records': len(values),
                   'overall_accuracy': 100 * sum(v[0] for v in values) / len(values),
                   'partial_accuracy': 100 * sum(v[1] for v in values) / len(values)}
            for task, values in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    predictions = {}
    for row in read_jsonl(args.predictions):
        if row['id'] in predictions:
            raise ValueError(f'Duplicate prediction id: {row["id"]}')
        predictions[row['id']] = row['prediction']
    result = score_records(list(read_jsonl(args.dataset)), predictions)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
