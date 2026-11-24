#!/usr/bin/env python3
"""Launch the measure-based A2 baseline with the exp1122 three-GPU trainer."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.prepare_exp1123_measure import LENGTH, OUTPUT as DATA, STRIDE, sha256

DEFAULT_RUN = ROOT.parent / 'results/exp1123_A2_measure_3gpu_bs48_e512_o50'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--gpus', default='0,1,2')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(set(gpus)) != 3 or len(gpus) != 3 or any(not gpu.isdecimal() for gpu in gpus):
        parser.error('--gpus requires three distinct CUDA device indices')
    data = args.data_dir.resolve()
    audit = json.loads((data / 'audit.json').read_text())
    if (audit['status'], audit['event_grid'], audit['window_events'], audit['stride_events'], audit['overlap_ratio']) != (
        'passed', 'measure', LENGTH, STRIDE, .5
    ):
        raise ValueError('Expected an audited MEASURE/NOTE sidecar with 512 events and 50% overlap')
    sidecars = {}
    for split in ('train', 'valid', 'test'):
        path = data / f'dinr_measure_{split}_e512_o50_v1.pt'
        if sha256(path) != audit['splits'][split]['sha256']:
            raise ValueError(f'{split}: sidecar hash differs from the audit')
        sidecars[split] = path
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
               '-m', 'src.train.train_dinr_beat', '--output-dir', str(output),
               '--train-sidecar', str(sidecars['train']), '--valid-sidecar', str(sidecars['valid']),
               '--epochs', '16', '--learning-rate', '3e-4', '--batch-size', '16',
               '--gradient-accumulation-steps', '1', '--seed', '42', '--early-stopping-patience', '4',
               '--duration-target', 'log_dev', '--beat-duration-target', 'log_dev', '--note-duration-target', 'log_dev',
               '--duration-input-tables', 'shared', '--duration-output-head', 'separate', '--onset-target', 'dev',
               '--fixed-event-length', str(LENGTH), '--normalize-partial-accumulation', '--no-beat-type-embedding',
               '--no-load-best-model-at-end']
    protocol = dict(experiment='exp1123', event_grid='measure', event_types=['MEASURE', 'NOTE'],
                    reference='exp1122 A2 structural-earliest', window_events=LENGTH, stride_events=STRIDE,
                    overlap_ratio=.5, gpus=[int(gpu) for gpu in gpus], world_size=3, per_device_batch_size=16,
                    gradient_accumulation_steps=1, global_batch_size=48, sidecar_audit=str(data / 'audit.json'),
                    test_sidecar=str(sidecars['test']), command=command)
    (output / 'exp1123_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    with (output / 'train.log').open('w') as log:
        print(' '.join(command), file=log, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
