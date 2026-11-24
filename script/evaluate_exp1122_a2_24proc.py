#!/usr/bin/env python3
"""Evaluate each test work in a 24-process pool: eight workers per GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RUN = PROJECT / 'results/exp1122_A2_structural_3gpu_bs48'
SIDECAR = PROJECT / 'data/ASAP_processed_dinr_beat_structural_earliest_e512_v1/dinr_beat_test_e512_v1.pt'
OUTPUT = RUN / 'test_24proc'
SHARDS = RUN / 'test_24proc_shards'
GPUS = (0, 1, 2)
WORKERS_PER_GPU = 8


def run_work(task):
    work_index, work_name, event_count, gpu = task
    shard = SHARDS / f'work_{work_index:02d}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', '-m', 'src.evaluate.evaluate_dinr_beat',
        '--checkpoint', str(RUN / 'best'), '--sidecar', str(SIDECAR),
        '--output-dir', str(shard), '--split', 'test', '--samples', '3',
        '--device', 'cuda:0', '--work-start', str(work_index), '--work-end', str(work_index + 1)]
    log = shard / 'worker.log'
    shard.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(index=work_index, work=work_name, events=event_count, gpu=gpu,
                output=str(shard / f'work_{work_index:02d}'))


def evaluate_main_metrics():
    sys.path.insert(0, str(ROOT))
    from script.evaluate_exp1117_beat_dev import decode_numpy, duration_log_coordinate
    from script.evaluate_exp1117_five_main_metrics import work_metrics
    from src.data_process.dinr_beat_dataset import BeatTokenSpec as S

    run = json.loads((RUN / 'run_config.json').read_text())
    works = sorted(OUTPUT.glob('work_*'))
    if len(works) != 19:
        raise ValueError(f'Expected 19 test works, found {len(works)}')
    per_work = []
    for work_dir in works:
        work = json.loads((work_dir / 'metrics.json').read_text())['work']
        predictions = []
        for sample in range(3):
            with np.load(work_dir / f'autoregressive_{sample:02d}_tokens.npz') as data:
                tokens = data['predictions'][0]
                if sample == 0:
                    events = data['events'].astype(float)
                    human = data['reference_continuous'].astype(float)
                    valid = data['reference_valid']
                    beat = events[:, 0] == S.beat_type
                    note = events[:, 0] == S.note_type
                    score_log = np.log(np.maximum(events[:, 2], 1.))
                    human_log = np.log(np.maximum(human[:, :, 0], 1.))
                    reference = dict(humans=len(human), membership=(beat.cumsum() - 1)[note],
                        human_beat_log=human_log[:, beat], human_velocity=human[:, note, 2],
                        human_onset=human[:, note, 1],
                        human_duration=(human_log - score_log)[:, note])
                    masks = dict(beat=valid[:, beat, 0].all(0), duration=valid[:, note, 0].all(0),
                                 onset=valid[:, note, 1].all(0), velocity=valid[:, note, 2].all(0))
                elif not (np.array_equal(events, data['events'])
                          and np.array_equal(human, data['reference_continuous'])
                          and np.array_equal(valid, data['reference_valid'])):
                    raise ValueError(f'{work}: score or human references differ across draws')
                beat_dev = duration_log_coordinate(tokens[beat, 0], events[beat], run['beat_duration_target'])
                note_dev = duration_log_coordinate(tokens[note, 0], events[note], run['note_duration_target'])
                if run['onset_target'] != 'dev':
                    raise ValueError('This evaluation expects A2 onset dev')
                predictions.append(dict(beat_log=score_log[beat] + beat_dev, duration=note_dev,
                    onset=decode_numpy(tokens[note, 1], S.onset_dev_range, S.onset_dev_bins),
                    velocity=tokens[note, 2]))
        per_work.append(dict(work=work, humans=len(human), **work_metrics(predictions, reference, masks)))
    correlations = {}
    for kind in ('tempo', 'velocity'):
        by_draw = [[row['per_sample_human_correlations'][sample][kind]['mean'] for row in per_work]
                   for sample in range(3)]
        eligible = [i for i in range(len(per_work)) if all(draw[i] is not None for draw in by_draw)]
        if not eligible:
            raise ValueError(f'No works with a defined {kind} correlation in every draw')
        values = [float(np.mean([draw[i] for i in eligible])) for draw in by_draw]
        correlations[kind] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                                  sampling_values=values, works=len(eligible))
    mae = {name: float(np.mean([row['mae'][name] for row in per_work]))
           for name in ('onset', 'duration', 'velocity')}
    result = dict(works=len(works), human_performances=sum(row['humans'] for row in per_work),
        protocol=dict(predictions=str(OUTPUT), observation='native BEAT tokens and aligned Human labels',
            correlation='Pearson vs each human, averaged within work, then across works and draws',
            velocity='Mean note-on velocity per score beat, using common valid beats and notes',
            mae='Absolute difference between 3-draw and human means per score note, macro-averaged by work'),
        correlations=correlations, mae=mae, per_work=per_work)
    metrics_path = ROOT / 'analysis/exp1122_structural_beat/five_main_metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(correlations=correlations, mae=mae), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-only', action='store_true', help='Score saved AR outputs')
    if parser.parse_args().metrics_only:
        evaluate_main_metrics()
        return
    if not (RUN / 'best/pytorch_model.bin').is_file():
        raise FileNotFoundError(RUN / 'best/pytorch_model.bin')
    payload = torch.load(SIDECAR, map_location='cpu', weights_only=False)
    groups = {}
    for row in payload['performances']:
        groups.setdefault(row['work'], []).append(row)
    tasks = []
    for index, (work, rows) in enumerate(sorted(groups.items())):
        start, end = rows[0]['span']
        event_count = end - start
        tasks.append(dict(index=index, work=work, events=event_count))
    tasks.sort(key=lambda row: (-row['events'], row['work']))

    # LPT dispatch: submit longest works first, assigning each to the GPU pool
    # with the least estimated queued event load. Each GPU owns eight processes.
    loads = {gpu: 0 for gpu in GPUS}
    pools = {gpu: ProcessPoolExecutor(max_workers=WORKERS_PER_GPU) for gpu in GPUS}
    futures = {}
    scheduled = []
    try:
        for item in tasks:
            gpu = min(GPUS, key=lambda candidate: (loads[candidate], candidate))
            task = (item['index'], item['work'], item['events'], gpu)
            futures[pools[gpu].submit(run_work, task)] = task
            scheduled.append(dict(index=item['index'], work=item['work'], events=item['events'], gpu=gpu))
            loads[gpu] += item['events']
        completed = []
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"completed {len(completed)}/{len(tasks)} on GPU{result['gpu']}: "
                  f"{result['work']} ({result['events']:,} events)", flush=True)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for result in completed:
        source = Path(result['output'])
        destination = OUTPUT / source.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination)
    (OUTPUT / 'dispatch.json').write_text(json.dumps(dict(
        workers_per_gpu=WORKERS_PER_GPU, gpus=list(GPUS), total_workers=24,
        scheduling='longest-processing-time first; greedy assignment by queued event count',
        tasks_in_dispatch_order=scheduled,
        completed=sorted(completed, key=lambda row: row['index'])), indent=2, ensure_ascii=False) + '\n')

    sys.path.insert(0, str(ROOT))
    from script.evaluate_exp1117_beat_dev import evaluate_run
    metrics = evaluate_run(RUN, test_dir_name='test_24proc')
    metrics['setting'] = 'A2_structural_earliest_3gpu_24proc_inference'
    metrics['inference_protocol'] = dict(gpus=list(GPUS), workers_per_gpu=WORKERS_PER_GPU,
        worker_pool_size=24, scheduling='longest work first; greedy GPU load balance')
    metrics_path = ROOT / 'analysis/exp1122_structural_beat/model_metrics_24proc.json'
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics['metrics'], indent=2), flush=True)
    evaluate_main_metrics()


if __name__ == '__main__':
    main()
