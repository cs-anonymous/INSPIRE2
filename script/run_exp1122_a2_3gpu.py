#!/usr/bin/env python3
"""Train and evaluate A2 on the exp1122 structural-earliest sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = PROJECT / 'data/ASAP_processed_dinr_beat_structural_earliest_e512_v1'


def run(command, cwd=ROOT):
    print('>', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT / 'results/exp1122_A2_structural_3gpu_bs48')
    parser.add_argument('--gpus', default='0,1,2')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_sidecar = DATA / 'dinr_beat_train_e512_v1.pt'
    valid_sidecar = DATA / 'dinr_beat_valid_e512_v1.pt'
    test_sidecar = DATA / 'dinr_beat_test_e512_v1.pt'
    for path in (train_sidecar, valid_sidecar, test_sidecar):
        if not path.exists():
            raise FileNotFoundError(path)
    audit_path = ROOT / 'analysis/exp1122_structural_beat/fixed_event_data_audit_checked.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'passed' or audit['splits']['test']['works'] != 19:
        raise RuntimeError('exp1122 fixed-event sidecar audit must pass with all 19 test works')
    for split, row in audit['splits'].items():
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'{split} sidecar hash differs from the audit')
    if not args.skip_training:
        # Global batch remains 48: 16 examples per GPU x 3 GPUs x accumulation 1.
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=3', '--master_port=29517',
                   '-m', 'src.train.train_dinr_beat',
                   '--output-dir', str(output), '--train-sidecar', str(train_sidecar),
                   '--valid-sidecar', str(valid_sidecar), '--epochs', '16',
                   '--learning-rate', '3e-4', '--batch-size', '16',
                   '--gradient-accumulation-steps', '1', '--seed', '42',
                   '--early-stopping-patience', '4', '--duration-target', 'log_dev',
                   '--beat-duration-target', 'log_dev', '--note-duration-target', 'log_dev',
                   '--duration-input-tables', 'shared', '--duration-output-head', 'separate',
                   '--onset-target', 'dev', '--fixed-event-length', '512',
                   '--normalize-partial-accumulation', '--no-beat-type-embedding',
                   '--no-load-best-model-at-end']
        if args.smoke:
            command[command.index('--epochs') + 1] = '0.01'
            command.extend(['--limit-train', '12', '--limit-valid', '6'])
        env = dict(__import__('os').environ, CUDA_VISIBLE_DEVICES=args.gpus,
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1')
        run_env = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if run_env.returncode:
            raise SystemExit(run_env.returncode)
    if args.smoke:
        return
    checkpoint = output / 'best'
    if not (checkpoint / 'pytorch_model.bin').exists():
        raise FileNotFoundError(checkpoint / 'pytorch_model.bin')
    run([sys.executable, '-u', '-m', 'src.evaluate.evaluate_dinr_beat',
         '--checkpoint', str(checkpoint), '--sidecar', str(valid_sidecar),
         '--output-dir', str(output / 'validation'), '--split', 'valid'])
    run([sys.executable, str(ROOT / 'script/evaluate_exp1122_a2_24proc.py')])
    sys.path.insert(0, str(ROOT))
    from script.evaluate_exp1117_beat_dev import evaluate_run
    metrics = evaluate_run(output, test_dir_name='test_24proc')
    metrics['setting'] = 'A2_structural_earliest_3gpu'
    metrics['training_protocol'] = {
        'world_size': 3, 'per_device_batch_size': 16,
        'gradient_accumulation_steps': 1, 'global_batch_size': 48,
    }
    (ROOT / 'analysis/exp1122_structural_beat/model_metrics.json').write_text(
        json.dumps(metrics, indent=2) + '\n')
    run_config = json.loads((output / 'run_config.json').read_text())
    run_config.update({
        'experiment': 'exp1122',
        'anchor_method': 'structural_earliest',
        'distributed_world_size': 3,
        'per_device_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'global_batch_size': 48,
        'sidecar_root': str(DATA),
    })
    (output / 'exp1122_protocol.json').write_text(json.dumps(run_config, indent=2) + '\n')


if __name__ == '__main__':
    main()
