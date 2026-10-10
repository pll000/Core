"""Evaluate one CoRe test task with Qwen3-VL, without exposing answers to the model."""
import argparse
import json
from pathlib import Path

from data_utils import TASKS, image_path, read_jsonl, validate_row, write_json
from score import score_records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--data-root', type=Path, default=root / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=10000)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates the full task; positive values are a smoke run')
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1:
        parser.error('limit must be nonnegative and max-new-tokens positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    rows = list(read_jsonl(args.data_root / 'test' / f'{args.task}.jsonl'))
    total_available = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        validate_row(row, args.data_root)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype='auto', device_map='cuda:0')
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    predictions = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for index, row in enumerate(rows):
            content = [{'type': 'image', 'image': str(image_path(args.data_root, p))} for p in row['images']]
            content.append({'type': 'text', 'text': row['question_text']})
            messages = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info([messages], image_patch_size=16)
            inputs = processor(text=[text], images=image_inputs, do_resize=True, return_tensors='pt').to('cuda')
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    pad_token_id=processor.tokenizer.pad_token_id)
            tokens = output[0, inputs.input_ids.shape[1]:]
            prediction = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            predictions[row['id']] = prediction
            stream.write(json.dumps({'id': row['id'], 'prediction': prediction,
                                     'generated_tokens': len(tokens),
                                     'at_token_limit': len(tokens) >= args.max_new_tokens}, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{index + 1}/{len(rows)} {row["id"]}', flush=True)
    result = score_records(rows, predictions)
    result.update({'model': args.model, 'task': args.task, 'total_available': total_available,
                   'full_task_evaluation': len(rows) == total_available,
                   'max_new_tokens': args.max_new_tokens, 'do_sample': False})
    write_json(args.output.with_suffix('.summary.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
