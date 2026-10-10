"""CPU regression tests for data portability, scoring, and the preserved reward.

Reward tests load the bundled ORM base without importing the GPU training stack.
The separate environment check validates actual framework imports.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy  # Load before the isolated import context; keep one NumPy module instance.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from data_utils import image_path, validate_row
from export_data import convert_row
from prepare_data import prepare
from score import extract_answer, score_prediction, score_records


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    return {'id': 'counting/example', 'task_type': 'counting',
            'images': [f'images/example/image_{i}.jpg' for i in (1, 2, 3)],
            'messages': [{'role': 'user', 'content': 'Compare the three images.'}],
            'question_text': 'Compare the three images.', 'solution': '<answer>ABC</answer>',
            'image_label': {f'image_{i}': {'count': i * 100} for i in (1, 2, 3)}}


class DataTests(unittest.TestCase):
    def test_metadata_required(self):
        row = sample()
        validate_row(row)
        del row['image_label']
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_no_answers_in_model_input(self):
        row = sample()
        row['messages'].append({'role': 'assistant', 'content': row['solution']})
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_path_escape_rejected(self):
        for relative in ('/etc/passwd', 'images/../../secret', '../secret', 'other/image.jpg'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                image_path('.', relative)

    def test_task_specific_choices(self):
        row = sample()
        row['task_type'], row['solution'] = 'depth', '<answer>ABD</answer>'
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_conversion_preserves_count_metadata_and_moves_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory).resolve()
            original = sample()
            original['images'] = [str(source_root / p) for p in original['images']]
            original['messages'].append({'role': 'assistant', 'content': original.pop('solution')})
            converted = convert_row(original, 'counting', source_root)
            self.assertEqual(converted['image_label'], original['image_label'])
            self.assertEqual(converted['solution'], '<answer>ABC</answer>')
            self.assertEqual(len(converted['messages']), 1)
            self.assertEqual(len(original['messages']), 2)

    def test_prepare_is_relocatable_and_preserves_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = sample()
            for relative in row['images']:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'image path fixture')
            source, output = root / 'train.jsonl', root / 'runtime/train.jsonl'
            original = json.dumps(row) + '\n'
            source.write_text(original)
            self.assertEqual(prepare(source, root, output), 1)
            result = json.loads(output.read_text())
            self.assertTrue(all(Path(p).is_file() for p in result['images']))
            for key in row.keys() - {'images'}:
                self.assertEqual(result[key], row[key])
            self.assertEqual(source.read_text(), original)
            with self.assertRaises(ValueError):
                prepare(source, root, source)


class MetricTests(unittest.TestCase):
    def test_exact_and_partial(self):
        self.assertEqual(score_prediction('<answer>ABC</answer>', '<answer>ABC</answer>'), (1, 1))
        self.assertEqual(score_prediction('<answer>ABC</answer>', '<answer>ABD</answer>'), (0, 2 / 3))

    def test_malformed_length_is_zero_like_original_evaluator(self):
        for text in ('ABC', '<answer>AB</answer>', '<answer>ABCD</answer>', ''):
            self.assertEqual(score_prediction('<answer>ABC</answer>', text), (0, 0))

    def test_original_unclosed_tag_fallback(self):
        self.assertEqual(extract_answer('reasoning <answer>ABC'), 'ABC')

    def test_missing_predictions_remain_in_denominator(self):
        result = score_records([sample()], {})
        self.assertEqual(result['missing_predictions'], 1)
        self.assertEqual(result['metrics_percent']['all']['partial_accuracy'], 0)
        with self.assertRaises(ValueError):
            score_records([sample()], {'unknown': '<answer>ABC</answer>'})


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orm = load_module('bundled_orm', ROOT / 'third_party/ms-swift/swift/plugin/orm.py')
        plugin = types.ModuleType('swift.plugin')
        plugin.ORM, plugin.orms = orm.ORM, {}
        with patch.dict(sys.modules, {'swift': types.ModuleType('swift'), 'swift.plugin': plugin}):
            cls.reward = load_module('core_reward', ROOT / 'rewards/unified_reward_four_new.py')
        assert plugin.orms['unified_triplet_orm'] is cls.reward.UnifiedTripletORM

    def test_partial_accuracy_handles_missing_letters(self):
        self.assertEqual(self.reward.answer_judge('AB', 'ABC'), (2 / 3, 2))

    def test_exact_attribute_estimates_and_missing_metadata(self):
        text = '<think>Figure 1: 100 people. Figure 2: 200 people. Figure 3: 300 people.</think>'
        labels = sample()['image_label']
        self.assertAlmostEqual(self.reward.compute_counting_attribute_reward(text, labels), 0.2)
        self.assertEqual(self.reward.compute_counting_attribute_reward(text, None), 0)

    def test_group_reward_golden_and_gt_consistency(self):
        reward = self.reward.UnifiedTripletORM()
        completion = '<think>x</think><answer>ABC</answer>'
        with redirect_stdout(io.StringIO()):
            actual = reward([completion] * 4, solution=['<answer>ABC</answer>'] * 4,
                            task_type=['relation'] * 4)
            inconsistent = reward([completion] * 4,
                                  solution=['<answer>ABC</answer>'] * 3 + ['<answer>ABD</answer>'],
                                  task_type=['relation'] * 4)
        self.assertEqual(actual, [1.4] * 4)
        self.assertEqual(inconsistent, [0.0] * 4)

    def test_relation_structured_components_are_inactive(self):
        self.assertEqual(self.reward.check_spatial_consistency('ABC', 'question'), 0)
        self.assertEqual(self.reward.compute_answer_mapping_reward(
            '<think>top left</think>', 'question', 'ABC', 'relation'), 0)


if __name__ == '__main__':
    unittest.main()
