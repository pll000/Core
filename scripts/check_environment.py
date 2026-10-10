"""Fail early on incompatible dependencies or imports from another Swift checkout."""
import importlib
from importlib import metadata
from pathlib import Path
import sys


def main():
    from packaging.requirements import Requirement
    root = Path(__file__).resolve().parents[1]
    requirements = [
        'torch>=2.6', 'transformers>=4.57,<4.58', 'trl>=0.18,<0.21',
        'datasets>=3,<4', 'peft>=0.11,<0.18', 'accelerate>=1',
        'deepspeed>=0.16,<0.17', 'numpy', 'qwen-vl-utils>=0.0.14',
        'math-verify==0.5.2', 'tensorboard',
    ]
    problems = []
    print('Python:', sys.version.split()[0])
    for text in requirements:
        req = Requirement(text)
        try:
            version = metadata.version(req.name)
            print(f'{req.name}=={version}')
            if version not in req.specifier:
                problems.append(f'{text} required; found {version}')
        except metadata.PackageNotFoundError:
            problems.append(f'Missing dependency: {text}')
    if problems:
        raise SystemExit('\n'.join(problems))
    swift = importlib.import_module('swift')
    expected = root / 'third_party/ms-swift/swift'
    if Path(swift.__file__).resolve().parent != expected:
        raise SystemExit(f'Wrong Swift import: {swift.__file__}; expected {expected}')
    from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
    from swift.trainers.rlhf_trainer import GRPOTrainer
    if 'qwen3_vl' not in MODEL_MAPPING or 'qwen3_vl' not in TEMPLATE_MAPPING:
        raise SystemExit('Qwen3-VL model/template registration missing')
    print('Framework:', swift.__file__)
    print('Trainer:', GRPOTrainer.__module__)
    print('Environment import check passed (does not load a model or run training).')


if __name__ == '__main__':
    main()
