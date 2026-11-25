#!/usr/bin/env python3
"""Render the complete official ASAP test scores and evaluate Measure A2."""

# ruff: noqa: E402

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.prepare_exp1123_measure_official import PT2, PT2_EVAL, score_geometry, sha256
from src.data_process.build_dinr_beat_sidecars import load_beat_mapper
from src.data_process.dinr_beat_dataset import measure_event_spans
from src.evaluate.evaluate_dinr_beat import atomic_json, load_model, predict_many, to_continuous

DRAWS = 3
CAP = 512
MAX_WORKERS_PER_GPU = 8
RAW = PT2 / 'asap-dataset-master'


def score_events(score, get_bars):
    _, _, measure, _, _, note_fields = score_geometry(RAW, score, get_bars)
    measure_pos = np.arange(len(measure['counts'])) + np.r_[0, measure['counts'].cumsum()[:-1]]
    note_pos = np.arange(len(note_fields)) + measure['ids'] + 1
    result = np.empty((len(measure_pos) + len(note_pos), 6), dtype=np.float32)
    result[measure_pos] = [0, -1, 0, -1, -1, 1]
    result[measure_pos, 2] = measure['score_ms']
    result[note_pos] = note_fields
    return torch.from_numpy(result)


def generate(checkpoint, output, task, *, score_velocity_feedback=False, mean_measure_duration=False):
    score = task['work']
    get_bars, _ = load_beat_mapper()
    events = score_events(score, get_bars)
    chunks = measure_event_spans(events, CAP)
    if len(events) != task['events'] or len(chunks) != task['chunks']:
        raise ValueError(f'Score geometry changed for {score}')
    device = torch.device('cuda:0')
    model, run = load_model(checkpoint, device)
    seed = task['sampling_seed']
    placeholders = torch.zeros((len(events), 3), dtype=torch.float32)
    # One unique RNG stream per score, with three draws from that stream.
    predictions = predict_many(model, events, placeholders, run, device, [seed] * DRAWS, chunks=chunks,
                               score_velocity_feedback=score_velocity_feedback,
                               mean_measure_duration=mean_measure_duration)
    continuous = np.stack([to_continuous(tokens, events, model) for tokens in predictions])
    if continuous.shape != (DRAWS, len(events), 3) or not np.isfinite(continuous).all():
        raise ValueError(f'Invalid draws for {score}')
    dest = output / f'work_{task["index"]:02d}'
    temporary = dest / 'predictions.tmp.npz'
    np.savez_compressed(temporary, work=score, events=events.numpy(), continuous=continuous,
                        predictions=np.stack([tokens.numpy() for tokens in predictions]))
    temporary.replace(dest / 'predictions.npz')


def launch_work(checkpoint, output, task):
    dest = output / f'work_{task["index"]:02d}'
    dest.mkdir(exist_ok=True)
    if (dest / 'predictions.npz').is_file():
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(task['gpu']), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    cmd = [sys.executable, '-u', __file__, '--checkpoint', str(checkpoint), '--output-dir', str(output),
           '--work', str(task['index'])]
    with (dest / 'inference.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def schedule(scores, gpus):
    get_bars, _ = load_beat_mapper()
    tasks = []
    for index, score in enumerate(scores):
        events = score_events(score, get_bars)
        seed = int.from_bytes(hashlib.sha256(f'1115:{score}:0:AR-batch16'.encode()).digest()[:4], 'little')
        tasks.append(dict(index=index, work=score, events=len(events), chunks=len(measure_event_spans(events, CAP)),
                          sampling_seed=seed))
    loads = {gpu: 0 for gpu in gpus}
    counts = {gpu: 0 for gpu in gpus}
    for task in sorted(tasks, key=lambda x: (-x['events'], x['index'])):
        gpu = min((g for g in gpus if counts[g] < MAX_WORKERS_PER_GPU), key=loads.__getitem__)
        task['gpu'] = gpu
        loads[gpu] += task['events']
        counts[gpu] += 1
    if len({x['sampling_seed'] for x in tasks}) != len(tasks):
        raise ValueError('Two inference jobs share a sampling seed')
    return tasks


def record(stream, task, checkpoint, training_seed, output, status):
    row = dict(task, checkpoint=str(checkpoint), training_seed=training_seed,
               output_dir=str(output / f'work_{task["index"]:02d}'),
               log=str(output / f'work_{task["index"]:02d}/inference.log'), draws=DRAWS,
               temperature=1, top_p=.9, window_events=CAP, status=status)
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Directory containing the selected weights')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpus', default='0,1,2')
    parser.add_argument('--work', type=int)
    parser.add_argument('--score-velocity-feedback', action='store_true',
                        help='Feed score note velocities into the decoder history during an audit')
    parser.add_argument('--mean-measure-duration', action='store_true',
                        help='Use the duration head mean for measure events during an audit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    output = (args.output_dir or checkpoint.parent / 'test_official').resolve()
    manifest = output / 'dispatch.json'
    if args.work is not None:
        dispatch = json.loads(manifest.read_text())
        if dispatch['checkpoint'] != str(checkpoint):
            raise ValueError('Worker checkpoint differs from the parent launch')
        generate(checkpoint, output, dispatch['tasks'][args.work],
                 score_velocity_feedback=dispatch.get('score_velocity_feedback', False),
                 mean_measure_duration=dispatch.get('mean_measure_duration', False))
        return

    gpus = args.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3 or any(not x.isdecimal() for x in gpus):
        parser.error('--gpus requires three distinct CUDA device indices')
    scores = sorted(json.loads((PT2 / 'score_split.json').read_text())['test'])
    if len(scores) != 23 or len(set(scores)) != 23:
        raise ValueError('Expected 23 official test scores')
    run = json.loads((checkpoint.parent / 'run_config.json').read_text())
    weights = sha256(checkpoint / 'pytorch_model.bin')
    completion = json.loads((checkpoint.parent / 'completion.json').read_text())
    if weights != completion['sha256']:
        raise ValueError('Checkpoint differs from the exported best validation weights')
    tasks = schedule(scores, gpus)
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=weights, best_step=completion['best_step'],
                    training_seed=run['seed'], draws=DRAWS, temperature=1, top_p=.9, batch_size=16,
                    window_events=CAP, window_policy='nonoverlapping_measure_boundary', workers_per_gpu=8,
                    end_measure_fallback='score duration outside first to last score-note onset', tasks=tasks)
    if args.score_velocity_feedback:
        protocol['score_velocity_feedback'] = True
    if args.mean_measure_duration:
        protocol['mean_measure_duration'] = True
    output.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Existing inference output uses a different protocol')
    atomic_json(manifest, protocol)
    with (output / 'runs.jsonl').open('a') as ledger, ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {}
        for task in sorted(tasks, key=lambda x: -x['events']):
            record(ledger, task, checkpoint, run['seed'], output, 'scheduled')
            jobs[pool.submit(launch_work, checkpoint, output, task)] = task
        for finished, job in enumerate(as_completed(jobs), 1):
            task = jobs[job]
            try:
                job.result()
            except subprocess.CalledProcessError:
                record(ledger, task, checkpoint, run['seed'], output, 'failed')
                raise
            record(ledger, task, checkpoint, run['seed'], output, 'rendered')
            print(f'{finished}/{len(tasks)} rendered · GPU {task["gpu"]} · {task["work"]}', flush=True)
    with (output / 'score.log').open('w') as log:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, '-u', str(PT2_EVAL / 'score_official_split.py'),
                        '--measure-a2', str(output), '--output-dir', str(output)], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print((output / 'five_metrics_measure.csv').read_text(), end='', flush=True)


if __name__ == '__main__':
    main()
