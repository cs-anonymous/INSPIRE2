#!/usr/bin/env python3
"""Train Measure A2 on official ASAP with periodic test-window validation."""

# ruff: noqa: E402

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = PROJECT / 'data/ASAP_official_dinr_measure_structural_e512_snap50_crop_v1'
DEFAULT_RUN = ROOT / 'results/exp1123_A2_measure_official_e4_valid50_original_3gpu_20261124'
sys.path.insert(0, str(ROOT))

from script.prepare_exp1123_measure import sha256
from src.data_process.dinr_beat_dataset import BeatTokenSpec, DINRBeatCollator, DINRBeatDataset

EPOCHS = 4
BATCH_PER_GPU = 16
MAX_EVENTS = 512


def nonoverlapping_samples(data):
    samples = []
    position = 0
    for performance in data['performances']:
        start, end = performance['span']
        types = data['events'][start:end, 0].numpy()
        boundaries = np.r_[start + np.flatnonzero(types == BeatTokenSpec.beat_type), end]
        if start != position or len(boundaries) < 2 or boundaries[0] != start:
            raise ValueError(f'Invalid measure stream: {performance["source"]}')
        index = 0
        while index < len(boundaries) - 1:
            left = int(boundaries[index])
            next_index = int(np.searchsorted(boundaries, left + MAX_EVENTS, side='right') - 1)
            if next_index <= index:
                raise ValueError(f'Measure exceeds the window size: {performance["source"]}')
            right = int(boundaries[next_index])
            note_count = int((types[left-start:right-start] == BeatTokenSpec.note_type).sum())
            samples.append(dict(work=performance['work'], performance=performance['performance'],
                                source=performance['source'], span=[left, right], notes=note_count,
                                measures=right-left-note_count, kind='original'))
            position = right
            index = next_index
    if position != len(data['events']) or sum(b-a for a, b in (row['span'] for row in samples)) != position:
        raise ValueError('Validation must cover every test event exactly once')
    return samples


def validation_sidecar(test, destination, policy, expected_original):
    data = torch.load(test, map_location='cpu', weights_only=False)
    if policy == 'original':
        samples = [row for row in data['samples'] if row['kind'] == 'original']
        if len(samples) != expected_original:
            raise ValueError('Original validation window count differs from the data audit')
    else:
        samples = nonoverlapping_samples(data)
        data.update(window_policy='nonoverlapping_measure_boundary', overlap_ratio=0.0)
    data['samples'] = samples
    data.pop('random_cut_seed', None)
    temporary = destination.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(destination)
    dataset = DINRBeatDataset(destination, duration_target='log_dev', onset_target='dev')
    batch = DINRBeatCollator()([dataset[0], dataset[-1]])
    if (len(dataset) != len(samples) or batch['attention_mask'].shape[1] > MAX_EVENTS or
            int(batch['attention_mask'].sum()) != sum(samples[i]['span'][1]-samples[i]['span'][0] for i in (0, -1))):
        raise ValueError('Validation sidecar failed the dynamic padding check')
    return len(samples), len(data['events']), len(data['performances'])


def verify_best(output, eval_steps):
    state = json.loads((output / 'trainer_state.json').read_text())
    records = [x for x in state['log_history'] if 'eval_loss' in x]
    expected = list(range(eval_steps, state['global_step'] + 1, eval_steps))
    if (not math.isclose(state['epoch'], EPOCHS) or state['global_step'] != state['max_steps'] or
            [x['step'] for x in records] != expected):
        raise ValueError('Training did not complete the specified epochs and validation cadence')
    best = min(records, key=lambda x: x['eval_loss'])
    checkpoint = Path(state['best_model_checkpoint'])
    weights = output / 'best' / 'pytorch_model.bin'
    if (state['best_metric'] != best['eval_loss'] or checkpoint.name != f'checkpoint-{best["step"]}' or
            not (checkpoint / 'pytorch_model.bin').is_file() or not weights.is_file()):
        raise ValueError('The best validation checkpoint was not exported')
    report = dict(epochs=state['epoch'], optimizer_steps=state['global_step'], validation_points=len(records),
                  best_step=best['step'], best_valid_loss=best['eval_loss'],
                  checkpoint=str(checkpoint), best_weights=str(weights), sha256=sha256(weights))
    (output / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--gpus', default='0,1,2')
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--validation', choices=('original', 'nonoverlapping'), default='original')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3 or any(not x.isdecimal() for x in gpus):
        parser.error('--gpus requires three distinct CUDA device indices')
    if args.eval_steps <= 0:
        parser.error('--eval-steps must be positive')
    data = args.data_dir.resolve()
    audit = json.loads((data / 'audit.json').read_text())
    if (audit['status'], audit['event_grid'], audit['window_events'], audit['split_for_validation']) != (
            'passed', 'measure', MAX_EVENTS, 'test'):
        raise ValueError('Expected audited official ASAP Measure A2 data')
    sources = {split: data / Path(audit['splits'][split]['path']).name for split in ('train', 'test')}
    for split, path in sources.items():
        if sha256(path) != audit['splits'][split]['sha256']:
            raise ValueError(f'{split} sidecar differs from its audit')
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    validation = output / f'valid_{args.validation}_e512.pt'
    expected_original = audit['splits']['test']['original']['windows']
    windows, events, performances = validation_sidecar(sources['test'], validation, args.validation, expected_original)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
               '-m', 'src.train.train_dinr_beat', '--output-dir', str(output),
               '--train-sidecar', str(sources['train']), '--valid-sidecar', str(validation),
               '--epochs', str(EPOCHS), '--eval-steps', str(args.eval_steps), '--early-stopping-patience', '0',
               '--learning-rate', '3e-4', '--batch-size', str(BATCH_PER_GPU), '--gradient-accumulation-steps', '1',
               '--seed', '42', '--duration-target', 'log_dev', '--beat-duration-target', 'log_dev',
               '--note-duration-target', 'log_dev', '--duration-input-tables', 'shared',
               '--duration-output-head', 'separate', '--onset-target', 'dev', '--normalize-partial-accumulation',
               '--no-beat-type-embedding', '--no-load-best-model-at-end']
    protocol = dict(gpus=[int(x) for x in gpus], world_size=3, global_batch_size=BATCH_PER_GPU*3,
                    epochs=EPOCHS, eval_steps=args.eval_steps, early_stopping=False, padding='dynamic',
                    train_windows=audit['splits']['train']['total']['windows'], valid_windows=windows,
                    valid_events=events, valid_performances=performances, valid_split='test',
                    valid_sidecar=str(validation), valid_sha256=sha256(validation),
                    original_test_sidecar=str(sources['test']), sidecar_audit=str(data / 'audit.json'),
                    validation_window_selection=args.validation, command=command)
    (output / 'exp1123_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(f'Validation: {windows} windows, {events} events, {performances} performances', flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    with (output / 'train.log').open('w') as log:
        print(' '.join(command), file=log, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_best(output, args.eval_steps)
    evaluation = output / 'test_official'
    evaluation.mkdir(exist_ok=True)
    test_command = [sys.executable, '-u', str(ROOT / 'script/evaluate_exp1123_a2_official.py'),
                    '--checkpoint', str(output / 'best'), '--output-dir', str(evaluation), '--gpus', args.gpus]
    test_env = dict(env, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (evaluation / 'launch.log').open('w') as log:
        subprocess.run(test_command, cwd=ROOT, env=test_env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
