#!/usr/bin/env python3
"""Coordinate the three independent design1125 training and test runs."""

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.prepare_exp1125_chord import DATA, RUN, sha256
from script.train_exp1125_chord import EPOCHS, EVAL_STEPS, SEED
from src.data_process.dinr_chord_dataset import MODES


def launch(script, args, log, gpu=None):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '')
    command = [sys.executable, '-u', str(ROOT / 'script' / f'{script}.py'), *args]
    with log.open('w') as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)


def read_json(path):
    return json.loads(path.read_text())


def training_result(output):
    path = output / 'training_complete.json'
    if not path.exists():
        return None
    trained = read_json(path)
    config = read_json(output / 'run_config.json')
    rows = [json.loads(line) for line in (output / 'loss_history.jsonl').read_text().splitlines()]
    valid = [row for row in rows if 'valid_loss' in row]
    best = min(valid, key=lambda row: row['valid_loss'])
    steps = list(range(EVAL_STEPS, config['optimizer_steps'], EVAL_STEPS)) + [config['optimizer_steps']]
    if (trained['epochs'] != EPOCHS or trained['optimizer_steps'] != config['optimizer_steps'] or
            [row['step'] for row in valid] != steps or
            (best['step'], best['valid_loss']) != (trained['best_step'], trained['best_valid_loss']) or
            sha256(output / 'best.pt') != trained['sha256']):
        raise ValueError(f'{output.name}: selected checkpoint failed training audit')
    return trained


def process_alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(')', 1)[1].split()[0] != 'Z'


def plot_losses():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for mode in MODES:
        output = RUN / mode
        rows = [json.loads(line) for line in (output / 'loss_history.jsonl').read_text().splitlines()]
        train = [row for row in rows if 'train_loss' in row]
        valid = [row for row in rows if 'valid_loss' in row]
        best = min(valid, key=lambda row: row['valid_loss'])
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        train_y = np.array([x['train_loss'] for x in train])
        ax.plot([x['step'] for x in train], train_y, alpha=.2, color='#1e6c99')
        ax.plot([x['step'] for x in train][19:], np.convolve(train_y, np.ones(20) / 20, mode='valid'),
                label='Train, 20-step mean', color='#1e6c99')
        ax.plot([x['step'] for x in valid], [x['valid_loss'] for x in valid], 'o-', label='Valid', color='#bea120')
        ax.scatter(best['step'], best['valid_loss'], color='#ba431c', label='Selected checkpoint', zorder=5)
        ax.set(xlabel='Optimizer step', ylabel='Sum of head CE', title=mode)
        ax.legend(frameon=False)
        fig.savefig(output / 'loss_curves.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-pids', type=int, nargs=3, metavar='PID',
                        help='Use training processes already started in abs5000, log_abs, log_dev order')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if not (DATA / 'manifest.json').exists():
        if launch('prepare_exp1125_chord', [], RUN / 'prepare.log').wait():
            raise RuntimeError('Chord data preparation failed')

    with (RUN / 'runs.jsonl').open('a') as ledger:
        def event(mode, gpu, status, **extras):
            base = dict(mode=mode, gpu=gpu, training_seed=SEED, sampling_seed=None, status=status,
                        checkpoint=str(RUN / mode / 'best.pt'), output_dir=str(RUN / mode),
                        config=str(RUN / mode / 'run_config.json'))
            ledger.write(json.dumps(dict(base, **extras)) + '\n')
            ledger.flush()
            print(mode, gpu, status, extras, flush=True)

        active, inference = {}, {}
        for gpu, mode in enumerate(MODES):
            output = RUN / mode
            output.mkdir(exist_ok=True)
            if args.attach_pids:
                pid = args.attach_pids[gpu]
            else:
                if (output / 'loss_history.jsonl').exists():
                    raise RuntimeError(f'{mode}: use --attach-pids for this existing run')
                pid = launch('train_exp1125_chord', [mode], output / 'train.log', gpu).pid
            active[mode] = gpu, pid
            event(mode, gpu, 'training', pid=pid, log=str(output / 'train.log'))

        while active or inference:
            for mode, (gpu, pid) in list(active.items()):
                output = RUN / mode
                trained = training_result(output)
                if trained is None:
                    if not process_alive(pid):
                        event(mode, gpu, 'train_failed', pid=pid)
                        raise RuntimeError(f'{mode}: training process exited without verified completion')
                    continue
                event(mode, gpu, 'trained', best_step=trained['best_step'], sha256=trained['sha256'])
                process = launch('evaluate_exp1125_chord', [mode], output / 'infer.log', gpu)
                inference[mode] = gpu, process
                del active[mode]
                event(mode, gpu, 'rendering', pid=process.pid, log=str(output / 'infer.log'))
            for mode, (gpu, process) in list(inference.items()):
                code = process.poll()
                if code is None:
                    continue
                if code or not (RUN / mode / 'test_official/prediction_manifest.json').exists():
                    event(mode, gpu, 'render_failed', exit_code=code)
                    raise RuntimeError(f'{mode}: inference failed with exit code {code}')
                event(mode, gpu, 'rendered')
                del inference[mode]
            if active or inference:
                time.sleep(4)
        if launch('evaluate_exp1125_chord', ['score'], RUN / 'score.log').wait():
            raise RuntimeError('Test metrics failed')
        plot_losses()
        for gpu, mode in enumerate(MODES):
            event(mode, gpu, 'evaluated', metrics=str(RUN / 'five_metrics.csv'))


if __name__ == '__main__':
    main()
