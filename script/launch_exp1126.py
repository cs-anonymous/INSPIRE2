#!/usr/bin/env python3
"""Run the three independent design1126 pipelines and compare their checkpoints."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time

from run_exp1126 import CONTINUATION, MODES, PAIRED_PRETRAIN, RUN, evaluate, read_json, write_json


def launch_continuation(output):
    milestones = ('epoch1p5', 'epoch2')
    worker = Path(__file__).with_name('run_exp1126.py')
    jobs = {}
    gpu_env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def launch(name, gpu, *args):
        with (output / f'{name}.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(worker), 'paired', *args],
                                       env=dict(gpu_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                       stdout=log, stderr=subprocess.STDOUT)
        jobs[name] = process
        write_json(output / 'jobs.json', {key: job.pid for key, job in jobs.items()})
        return process

    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        def terminate_running():
            for job in jobs.values():
                if job.poll() is None:
                    job.terminate()

        cleanup.callback(terminate_running)
        pretrain = launch('pretrain', 0, '--continue-pretrain')
        while True:
            for name, process in jobs.items():
                exit_code = process.poll()
                if exit_code not in (None, 0):
                    write_json(output / 'failed.json', {'job': name, 'exit_code': exit_code})
                    raise RuntimeError(f'Continuation job {name} exited with status {exit_code}')
            for gpu, name in enumerate(milestones, start=1):
                checkpoint = output / name / 'pretrain/training_complete.json'
                if name not in jobs and checkpoint.is_file():
                    launch(name, gpu, '--evaluate-continuation', name)
                elif name not in jobs and pretrain.poll() == 0:
                    raise FileNotFoundError(checkpoint)
            if len(jobs) == 3 and all(job.poll() == 0 for job in jobs.values()):
                break
            time.sleep(5)

    manifests = {}
    for name, root in (('epoch1', PAIRED_PRETRAIN.parent), *((name, output / name) for name in milestones)):
        for stage in ('pretrain', 'sft'):
            for kind in ('median', 'sampled'):
                path = root / stage / 'test_official' / f'{kind}.json'
                manifests[f'paired {name} {stage} {kind}'] = read_json(path)
    evaluate(levels='all', extra_chord_manifests=manifests, output=output)
    write_json(output / 'complete.json', {'metrics': str(output / 'multilevel_metrics.csv'),
                                         'outcomes': {name: job.returncode for name, job in jobs.items()}})


def main():
    root = RUN / 'runs'
    root.mkdir(parents=True, exist_ok=True)
    if not (RUN / 'data/manifest.json').is_file():
        raise FileNotFoundError('PianoCoRe data manifest is required')
    for mode in MODES:
        if not (RUN / 'smoke' / mode / 'complete.json').is_file():
            raise FileNotFoundError(f'{mode}: end-to-end smoke is required')

    jobs = {}
    for gpu, mode in enumerate(MODES):
        log = root / f'{mode}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('a') as output:
            jobs[mode] = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run_exp1126.py')),
                                           mode], env=env, stdout=output, stderr=subprocess.STDOUT)
    write_json(root / 'jobs.json', {mode: job.pid for mode, job in jobs.items()})
    outcomes = {mode: job.wait() for mode, job in jobs.items()}
    if any(outcomes.values()):
        write_json(root / 'failed.json', outcomes)
        raise SystemExit(1)

    manifests = {}
    for mode in MODES:
        for stage in ('pretrain', 'sft'):
            for kind in ('median', 'sampled'):
                path = root / mode / stage / 'test_official' / f'{kind}.json'
                manifests[f'{mode} {stage} {kind}'] = read_json(path)
    evaluate(levels='all', extra_chord_manifests=manifests, output=root)
    write_json(root / 'complete.json', {'metrics': str(root / 'multilevel_metrics.csv'), 'outcomes': outcomes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-paired', action='store_true')
    args = parser.parse_args()
    if args.continue_paired:
        launch_continuation(CONTINUATION / 'runs')
    else:
        main()
