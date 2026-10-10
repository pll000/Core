"""CPU integration check: actual Swift preprocessing preserves reward supervision."""
from collections import Counter
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile

from data_utils import read_jsonl
from prepare_data import prepare


def main():
    root = Path(__file__).resolve().parents[1]
    from swift.llm import load_dataset
    from swift.plugin import orms
    from swift.llm.dataset.preprocessor import RowPreprocessor
    original = []
    counts = Counter()
    for row in read_jsonl(root / 'data/train.jsonl'):
        if counts[row['task_type']] < 4:
            original.append(row)
            counts[row['task_type']] += 1
        if len(original) == 16:
            break
    with tempfile.TemporaryDirectory(prefix='core-dataset-check-') as directory:
        directory = Path(directory)
        portable, runtime = directory / 'sample.jsonl', directory / 'runtime.jsonl'
        portable.write_text(''.join(json.dumps(row) + '\n' for row in original))
        prepare(portable, root / 'data', runtime)
        dataset, validation = load_dataset(
            [str(runtime)], split_dataset_ratio=0, remove_unused_columns=False,
            load_from_cache_file=False, strict=True, shuffle=False)
        assert validation is None and len(dataset) == 16
        by_id = {row['id']: row for row in original}
        for row in dataset:
            expected = by_id[row['id']]
            for field in ('solution', 'question_text', 'task_type'):
                assert row[field] == expected[field], (row['id'], field)
            if row['task_type'] == 'counting':
                assert row['image_label'] == expected['image_label']
            assert all(m['role'] != 'assistant' for m in row['messages'])
        spec = importlib.util.spec_from_file_location('core_reward_integration', root / 'rewards/unified_reward_four_new.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reward = orms['unified_triplet_orm']()
        values = []
        for task in counts:
            row = next(row for row in dataset if row['task_type'] == task)
            kwargs = RowPreprocessor.rows_to_batched([row] * 4)
            with redirect_stdout(io.StringIO()):
                actual = reward(['<think>x</think>' + row['solution']] * 4, **kwargs)
            assert len(actual) == 4 and all(math.isfinite(value) for value in actual)
            values.extend(actual)
        print(json.dumps({'status': 'passed', 'preprocessed_records': len(dataset),
                          'tasks': dict(counts), 'actual_plugin_reward_calls': 4,
                          'finite_rewards': len(values), 'model_loaded': False}, indent=2))


if __name__ == '__main__':
    main()
