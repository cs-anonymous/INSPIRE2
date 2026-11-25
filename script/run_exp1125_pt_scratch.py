#!/usr/bin/env python3
"""Replay the pinned PT2 recipe from random weights and evaluate its official test split."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

INSPIRE = Path(__file__).resolve().parents[1]
PROJECT = INSPIRE.parent
RECIPE = PROJECT / 'results/pt_rendering_public_recipe_20261124'
OFFICIAL = RECIPE / 'official_release'
PT2 = RECIPE / 'official_ddp_3gpu'
ARCHITECTURE = PROJECT / 'models/modelscope_pianist_transformer_base/config.json'
RUN = INSPIRE / 'results/exp1125_pt_scratch'
SEED = 42
EVAL_STEPS = 50

BOOTSTRAP = '''
import os
import random
import runpy
import torch
from src.model import pianoformer

random.seed(42)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
original_model = pianoformer.PianoT5Gemma

def random_model(_):
    config = original_model.config_class.from_json_file(os.environ['PT_SCRATCH_ARCHITECTURE'])
    config._attn_implementation = 'sdpa'
    config.torch_dtype = torch.bfloat16
    torch.manual_seed(42)
    model = original_model(config).to(dtype=torch.bfloat16)
    print(f'Random initialization only; {sum(p.numel() for p in model.parameters())} parameters; '
          f'{next(model.parameters()).dtype}', flush=True)
    return model

pianoformer.PianoT5Gemma = random_model
runpy.run_module('src.train.sft', run_name='__main__')
'''


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def train(output):
    source = read_json(PT2 / 'training_protocol.json')
    config = read_json(PT2 / 'official_sft_config.json')
    if len(config['data_paths']) != 1 or sha256(Path(config['data_paths'][0])) != source['sft_dataset_sha256']:
        raise ValueError('SFT data differ from the pinned PT2 run')
    config.update(pretrained_model=None, eval_steps=EVAL_STEPS, output_dir=str(output / 'checkpoints'),
                  logging_dir=str(output / 'tf-logs'))
    config_path = output / 'official_sft_config.json'
    write_json(config_path, config)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
               '--log-dir', str(output / 'rank_logs'), '--tee', '3', '--no-python', sys.executable, '-u',
               '-c', BOOTSTRAP, '--config', str(config_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PT_SCRATCH_ARCHITECTURE=str(ARCHITECTURE))
    write_json(output / 'training_protocol.json', dict(command=command, cwd=str(OFFICIAL), world_size=3,
               gpus=[0, 1, 2], global_batch=source['global_batch'], random_seed=SEED, weight_initialization='random',
               dtype='bfloat16', architecture_config=str(ARCHITECTURE), architecture_sha256=sha256(ARCHITECTURE),
               sft_dataset_sha256=source['sft_dataset_sha256'], pt2_config=str(PT2 / 'official_sft_config.json'),
               evaluation_interval=EVAL_STEPS, checkpoint_selection='final, as PT2'))
    record(output, 'training', config=str(config_path))
    with (output / 'train.log').open('w') as log:
        subprocess.run(command, cwd=OFFICIAL, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify_training(output):
    original = read_json(PT2 / 'completion.json')
    weights, = (output / 'checkpoints').glob('sft_*/model.safetensors')
    step = original['optimizer_steps']
    state_path = weights.parent / f'checkpoint-{step}/trainer_state.json'
    state = read_json(state_path)
    valid = [row for row in state['log_history'] if 'eval_loss' in row]
    if ((state['global_step'], state['epoch']) != (step, original['epochs']) or
            [row['step'] for row in valid] != list(range(EVAL_STEPS, step + 1, EVAL_STEPS))):
        raise ValueError('Scratch run differs from the PT2 optimizer budget or requested validation cadence')
    from safetensors import safe_open

    with safe_open(weights, framework='pt') as scratch, safe_open(original['weights'], framework='pt') as pt2:
        if scratch.keys() != pt2.keys() or any(scratch.get_slice(key).get_shape() != pt2.get_slice(key).get_shape()
                or scratch.get_slice(key).get_dtype() != pt2.get_slice(key).get_dtype() for key in pt2.keys()):
            raise ValueError('Scratch checkpoint architecture or training dtype differs from PT2')
    checksum = sha256(weights)
    if checksum == original['sha256']:
        raise ValueError('Scratch checkpoint matches pretrained PT2 weights')
    completion = dict(status='trained', weights=str(weights), sha256=checksum, optimizer_steps=step,
                      epochs=state['epoch'], trainer_state=str(state_path), final_logged_validation=valid[-1],
                      checkpoint_selection='final, as PT2')
    write_json(output / 'completion.json', completion)
    record(output, 'trained', checkpoint=str(weights.parent), sha256=checksum)
    return weights.parent


def record(output, status, **fields):
    row = dict(training_seed=SEED, status=status, output_dir=str(output), **fields)
    with (output / 'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(row, flush=True)


def render(output, checkpoint):
    path = RECIPE / 'official_split_measure/render_pt2_parallel.py'
    spec = importlib.util.spec_from_file_location('official_pt2_rendering', path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.FINAL = output / 'test_official'
    draws = []
    for index, (gpu, seed) in enumerate(renderer.ASSIGNMENTS):
        dest = output / 'draws' / f'draw_{index:03d}'
        dest.mkdir(parents=True, exist_ok=True)
        process = None
        if not (dest / 'prediction_manifest.json').exists():
            process = renderer.predict(checkpoint, dest, gpu, seed)
        draws.append((dest, gpu, seed, process))
        record(output, 'rendering' if process is not None else 'checking_render', gpu=gpu, sampling_seed=seed,
               checkpoint=str(checkpoint), prediction_dir=str(dest), config=str(output / 'official_sft_config.json'))
    manifests = []
    failures = []
    for dest, gpu, seed, process in draws:
        if process is not None and process.wait() != 0:
            failures.append(str(dest))
            record(output, 'render_failed', gpu=gpu, sampling_seed=seed, prediction_dir=str(dest))
            continue
        manifests.append(renderer.read_manifest(dest, checkpoint, seed))
        record(output, 'rendered', gpu=gpu, sampling_seed=seed, prediction_dir=str(dest))
    if failures:
        raise RuntimeError(f'Failed inference draws: {failures}')
    renderer.merge(manifests)


def evaluate(output):
    command = [sys.executable, '-u', str(INSPIRE / 'script/evaluate_exp1125_chord.py'),
               'score_levels', '--pt-scratch', str(output)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (output / 'score.log').open('w') as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    metrics = output / 'multilevel_metrics.csv'
    record(output, 'evaluated', metrics=str(metrics))
    print(metrics.read_text(), end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finish', action='store_true', help='Verify existing training, then finish inference and scoring')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if not args.finish:
        if (RUN / 'train.log').exists():
            raise RuntimeError('This training directory already has a run')
        train(RUN)
    checkpoint = verify_training(RUN)
    render(RUN, checkpoint)
    evaluate(RUN)


if __name__ == '__main__':
    main()
