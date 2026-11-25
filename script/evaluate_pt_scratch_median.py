#!/usr/bin/env python3
"""Replay the PT-scratch test cohort with conditional medians and compare with its saved draws."""

# ruff: noqa: E402

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

INSPIRE = Path(__file__).resolve().parents[1]
PROJECT = INSPIRE.parent
sys.path.insert(0, str(INSPIRE))

from script.evaluate_exp1125_chord import evaluate
from script.run_exp1125_pt_scratch import RUN, read_json, sha256, write_json

OUTPUT = RUN / 'ar_median'
WORKERS = 8
GPUS = (0, 1, 2)


def record(status, **fields):
    row = dict(status=status, training_seed=42, sampling_seed=None, **fields)
    with (OUTPUT / 'runs.jsonl').open('a') as ledger:
        ledger.write(json.dumps(row) + '\n')
    print(row, flush=True)


def start_shard(sampled, checkpoint, gpu, output):
    output.mkdir(exist_ok=True)
    if (output / 'prediction_manifest.json').is_file():
        return None
    command = [sys.executable, '-u', '-m', 'src.inference.infer_pt_testset',
               '--model-path', str(checkpoint), '--metadata', sampled['metadata'], '--midi-root', sampled['midi_root'],
               '--output-dir', str(output), '--split', sampled['split'], '--device', 'cuda',
               '--protocol', 'median', '--num-samples', '1', '--num-workers', str(WORKERS),
               '--max-context-length', str(sampled['max_context_length']), '--overlap-ratio', str(sampled['overlap_ratio']),
               '--num-shards', str(len(GPUS)), '--shard-index', str(gpu)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    with (output / 'inference.log').open('w') as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record('rendering', gpu=gpu, pid=process.pid, checkpoint=str(checkpoint), output_dir=str(output),
           config=command, workers=WORKERS)
    return process


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sampled_path = RUN / 'test_official/prediction_manifest.json'
    sampled = read_json(sampled_path)
    completion = read_json(RUN / 'completion.json')
    weights = Path(completion['weights'])
    checkpoint = weights.parent
    if Path(sampled['model_path']) != checkpoint or sha256(weights) != completion['sha256']:
        raise ValueError('The sampling baseline and replay must use the same verified PT-scratch checkpoint')
    baseline = {item['score_source']: item for item in sampled['items']}
    if len(baseline) != 23 or sum(len(item['ground_truth_paths']) for item in baseline.values()) != 101:
        raise ValueError('Unexpected official score/reference coverage')

    jobs = []
    for gpu in GPUS:
        output = OUTPUT / f'gpu_{gpu}'
        jobs.append((gpu, output, start_shard(sampled, checkpoint, gpu, output)))
    failures = [gpu for gpu, _, process in jobs if process is not None and process.wait()]
    if failures:
        raise RuntimeError(f'AR median generation failed on GPUs {failures}')

    items = {}
    for gpu, output, _ in jobs:
        report = read_json(output / 'prediction_manifest.json')
        if (report['protocol'] != 'pt_median' or report['shard_index'] != gpu or report['num_samples'] != 1 or
                report['num_workers'] != WORKERS or Path(report['model_path']) != checkpoint):
            raise ValueError(f'Unexpected inference protocol for GPU {gpu}')
        for row in report['items']:
            work = row['score_source']
            ref = baseline[work]
            if (work in items or any(row[field] != ref[field] for field in ('score_midi', 'ground_truth_paths')) or
                    len(row['prediction_paths']) != 1 or not Path(row['prediction_paths'][0]).is_file()):
                raise ValueError(f'Unexpected median prediction for {work}')
            items[work] = row
            record('rendered', gpu=gpu, checkpoint=str(checkpoint), score_source=work, output_dir=str(output),
                   prediction_path=row['prediction_paths'][0])
    if items.keys() != baseline.keys():
        raise ValueError('The AR median and sampling evaluations cover different scores')
    manifest = dict(model_path=str(checkpoint), checkpoint_sha256=completion['sha256'],
                    step=completion['optimizer_steps'], protocol='pt_median', num_samples=1,
                    max_context_length=sampled['max_context_length'], overlap_ratio=sampled['overlap_ratio'],
                    comparison_manifest=str(sampled_path),
                    comparison_manifest_sha256=hashlib.sha256(sampled_path.read_bytes()).hexdigest(),
                    items=[items[work] for work in baseline])
    write_json(OUTPUT / 'prediction_manifest.json', manifest)
    evaluate(levels='all', pt_scratch=RUN, extra_pt_manifests={'PT-scratch AR median': manifest}, output=OUTPUT)
    record('evaluated', checkpoint=str(checkpoint), output_dir=str(OUTPUT),
           metrics=str(OUTPUT / 'multilevel_metrics.csv'))


if __name__ == '__main__':
    main()
