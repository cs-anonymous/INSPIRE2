#!/usr/bin/env python3
"""Generate three autoregressive draws per test work from the measure A2 checkpoint."""

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
from script.prepare_exp1123_measure import OUTPUT as DATA
from src.evaluate.evaluate_dinr_beat import load_model, predict_many, to_continuous

RUN = ROOT.parent / 'results/exp1123_A2_measure_3gpu_bs48_e512_o50'
TEST = DATA / 'dinr_measure_test_e512_o50_v1.pt'
DRAWS = 3


def work_dir(output, index):
    return output / f'work_{index:02d}'


def generate(checkpoint, output, index, work, events):
    device = torch.device('cuda:0')
    model, run = load_model(checkpoint, device)
    # Preserve the exp1122 per-work RNG protocol and inference windows.
    seeds = [int.from_bytes(hashlib.sha256(f'1115:{work}:{draw}:AR-batch16'.encode()).digest()[:4], 'little')
             for draw in range(DRAWS)]
    placeholders = torch.zeros((len(events), 3), dtype=torch.float32)
    predictions = predict_many(model, events, placeholders, run, device, seeds)
    continuous = np.stack([to_continuous(tokens, events, model) for tokens in predictions])
    if continuous.shape != (DRAWS, len(events), 3) or not np.isfinite(continuous).all():
        raise ValueError(f'{work}: invalid output draws')
    dest = work_dir(output, index)
    dest.mkdir(parents=True, exist_ok=True)
    temporary = dest / 'predictions.tmp.npz'
    np.savez_compressed(temporary, work=work, events=events.numpy(),
                        predictions=np.stack([tensor.numpy() for tensor in predictions]), continuous=continuous)
    temporary.replace(dest / 'predictions.npz')


def run_work(checkpoint, sidecar, output, index, gpu):
    dest = work_dir(output, index)
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-u', __file__, '--checkpoint', str(checkpoint), '--sidecar', str(sidecar),
               '--output-dir', str(output), '--work', str(index)]
    with (dest / 'inference.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=RUN / 'best')
    parser.add_argument('--sidecar', type=Path, default=TEST)
    parser.add_argument('--output-dir', type=Path, default=RUN / 'test_24proc')
    parser.add_argument('--work', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint, sidecar, output = (path.resolve() for path in (args.checkpoint, args.sidecar, args.output_dir))
    payload = torch.load(sidecar, map_location='cpu', weights_only=False)
    if payload.get('event_grid') != 'measure' or payload['split'] != 'test':
        raise ValueError('Expected a MEASURE/NOTE test sidecar')
    groups = {}
    for row in payload['performances']:
        groups.setdefault(row['work'], row['span'])
    works = sorted(groups.items())
    if len(works) != 19 or len(payload['performances']) != 82:
        raise ValueError('Expected the frozen 19 works and 82 performances')
    if args.work is not None:
        work, (start, end) = works[args.work]
        generate(checkpoint, output, args.work, work, payload['events'][start:end])
        return
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(enumerate(works), key=lambda entry: (entry[1][1][0] - entry[1][1][1], entry[0]))
    loads = [0, 0, 0]
    tasks = []
    for index, (work, (start, end)) in ordered:
        gpu = min(range(3), key=loads.__getitem__)
        tasks.append(dict(index=index, work=work, events=end - start, gpu=gpu))
        loads[gpu] += end - start
    protocol = dict(checkpoint=str(checkpoint), sidecar=str(sidecar), draws=DRAWS, temperature=1, top_p=.9,
                    window_events=512, stride_events=512, batch_size=16, workers_per_gpu=8, tasks=tasks)
    (output / 'dispatch.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    with ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {pool.submit(run_work, checkpoint, sidecar, output, row['index'], row['gpu']): row for row in tasks}
        for finished, job in enumerate(as_completed(jobs), 1):
            job.result()
            row = jobs[job]
            print(f"{finished}/{len(tasks)} works completed; GPU {row['gpu']}, work {row['index']:02d}", flush=True)


if __name__ == '__main__':
    main()
