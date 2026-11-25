#!/usr/bin/env python3
"""Train and evaluate score-input or output ablations with chord AR."""

# ruff: noqa: E402

import argparse
import os
from pathlib import Path
import sys
import time
from typing import NamedTuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.evaluate_exp1125_chord import evaluate
from script.prepare_exp1125_chord import DATA, RUN, write_json
from script.run_exp1125_chord import launch, process_alive, read_json, training_result
from script.run_exp1125_chord_ar import (AR, DRAWS, TEMPERATURE, TOP_P, WORKERS, execute_tasks,
                                        prediction_items, prepare_tasks, record)
from script.train_exp1125_chord import DLM_DIRECT_RUN, INPUT_RUN, OUTPUT_RUN, SEED
from src.model.dinr_chord_ar_input import SCORE_INPUTS
from src.model.dinr_chord_ar_output import DIRECT_DLM_MODES, OUTPUT_MODES

DECODING_RUNS = {'median': 1, 'sampled': DRAWS}


class Experiment(NamedTuple):
    directory: Path
    variants: tuple[str, ...]
    key: str
    architecture: str
    seed_offset: int


INPUTS = Experiment(INPUT_RUN, SCORE_INPUTS, 'score_input', 'ar_input', 200000)
OUTPUTS = Experiment(OUTPUT_RUN, OUTPUT_MODES, 'output_mode', 'ar_output', 300000)
DIRECT_DLM = Experiment(DLM_DIRECT_RUN, DIRECT_DLM_MODES, 'output_mode', 'ar_output', 400000)


def decodings_for(variant):
    if variant in ('timing_dlm', 'all_dlm', *DIRECT_DLM_MODES):
        return DECODING_RUNS | {'continuous_median': 1}
    return DECODING_RUNS


def verify_report(report, completion, draws):
    items = report['items']
    if (len(items) != 23 or len({item['work'] for item in items}) != 23 or report['draws'] != draws or
            report['checkpoint_sha256'] != completion['sha256'] or
            Path(report['checkpoint']) != Path(completion['checkpoint']) or
            any(len(item['draws']) != draws for item in items)):
        raise ValueError('Prediction coverage or checkpoint identity differs')


def render(experiment, variant):
    training = experiment.directory / variant
    finished = training_result(training)
    config = read_json(training / 'run_config.json')
    if finished is None or config[experiment.key] != variant:
        raise ValueError(f'{variant}: training checkpoint or config missing')
    checkpoint = training / 'best.pt'
    output = training / 'test_official'
    output.mkdir(exist_ok=True)
    scores = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True, mmap=True)
    base_tasks = prepare_tasks('log_dev', checkpoint, finished['sha256'], scores)
    gpu = experiment.variants.index(variant)
    decodings = decodings_for(variant)
    tasks = []
    for base in base_tasks:
        relative = Path(base['output_dir'])
        task = dict(base, architecture=experiment.architecture, gpu=gpu,
                    config=str(training / 'run_config.json'), **{experiment.key: variant})
        if base['draw'] == 0:
            median = dict(task, decoding='median', sampling_seed=None,
                          output_dir=str(output / 'median' / relative.parent.name / relative.name))
            if 'continuous_median' in decodings:
                median['continuous_output'] = True
            tasks.append(median)
        tasks.append(dict(task, decoding='sampled',
                          sampling_seed=experiment.seed_offset + 1000 * gpu + base['sampling_seed'],
                          output_dir=str(output / 'sampled' / relative.parent.name / relative.name)))
    if len(tasks) != 92 or sum(task['windows'] for task in tasks) != 368:
        raise ValueError(f'{variant}: expected one median and three draws for each of 23 scores')
    write_json(output / 'dispatch.json', dict(workers=WORKERS, ordering='descending chord count',
               temperature=TEMPERATURE, top_p=TOP_P, tasks=tasks))
    execute_tasks(tasks, {experiment.architecture: checkpoint}, log_dir=experiment.directory, variant=variant)
    for decoding, draws in decodings.items():
        task_decoding = 'median' if decoding == 'continuous_median' else decoding
        items = prediction_items([task for task in tasks if task['decoding'] == task_decoding])
        if decoding == 'continuous_median':
            items = [dict(work=item['work'], draws=[item['draws'][0]['continuous_prediction']]) for item in items]
        report = dict(mode='log_dev', checkpoint=str(checkpoint), **{experiment.key: variant},
                      checkpoint_sha256=finished['sha256'], best_step=finished['best_step'],
                      draws=draws, workers=WORKERS, decoding=decoding, items=items)
        if decoding == 'sampled':
            report.update(temperature=TEMPERATURE, top_p=TOP_P)
        verify_report(report, finished, draws)
        write_json(output / f'{decoding}.json', report)


def setup_training(experiment):
    path = experiment.directory / 'training_jobs.json'
    if path.is_file():
        return read_json(path)
    jobs = []
    for gpu, variant in enumerate(experiment.variants):
        output = experiment.directory / variant
        output.mkdir(exist_ok=True)
        args = ['log_dev', '--autoregressive', '--' + experiment.key.replace('_', '-'), variant]
        process = launch('train_exp1125_chord', args, output / 'train.log', gpu)
        job = dict(mode='log_dev', gpu=gpu, pid=process.pid, training_seed=SEED, **{experiment.key: variant},
                   sampling_seed=None, checkpoint=str(output / 'best.pt'), config=str(output / 'run_config.json'),
                   output_dir=str(output))
        jobs.append(job)
        record(job, 'training', log_dir=experiment.directory)
    write_json(path, jobs)
    return jobs


def run(experiment):
    out = experiment.directory
    out.mkdir(exist_ok=True)
    jobs = setup_training(experiment)
    training = {job[experiment.key]: job for job in jobs}
    generating = {}
    while training or generating:
        for variant, job in list(training.items()):
            path = out / variant
            completion = training_result(path)
            if completion is None:
                if process_alive(job['pid']):
                    continue
                raise RuntimeError(f'{variant}: training stopped without verified completion')
            record(job, 'trained', log_dir=out, best_step=completion['best_step'], sha256=completion['sha256'])
            if not all((path / 'test_official' / f'{decoding}.json').is_file() for decoding in decodings_for(variant)):
                args = ['--render', variant]
                process = launch('run_exp1125_chord_ar_inputs', args, path / 'infer.log', job['gpu'])
                generating[variant] = process, job
                record(job, 'rendering', log_dir=out, worker_count=WORKERS)
            del training[variant]
        for variant, (process, job) in list(generating.items()):
            status = process.poll()
            if status is None:
                continue
            if status:
                record(job, 'render_failed', log_dir=out, exit_code=status)
                raise RuntimeError(f'{variant}: generation failed')
            record(job, 'rendered', log_dir=out, samples=23 * sum(decodings_for(variant).values()))
            del generating[variant]
        if training or generating:
            time.sleep(10)

    lookup = training_result(AR / 'log_dev')
    manifests = {'AR lookup median': read_json(AR / 'decoding_audit/log_dev/ar.json'),
                 'AR lookup sampled': read_json(AR / 'log_dev/test_official/prediction_manifest.json')}
    for label, draws in (('AR lookup median', 1), ('AR lookup sampled', DRAWS)):
        verify_report(manifests[label], lookup, draws)
    if experiment == DIRECT_DLM:
        for variant in OUTPUT_MODES:
            directory = OUTPUT_RUN / variant
            completion = training_result(directory)
            for decoding, draws in DECODING_RUNS.items():
                report = read_json(directory / 'test_official' / f'{decoding}.json')
                verify_report(report, completion, draws)
                manifests[f'AR {variant} {decoding}'] = report
    for job in jobs:
        variant = job[experiment.key]
        directory = out / variant
        completion = read_json(directory / 'training_complete.json')
        for decoding, draws in decodings_for(variant).items():
            report = read_json(directory / 'test_official' / f'{decoding}.json')
            verify_report(report, completion, draws)
            manifests[f'AR {variant} {decoding}'] = report
    evaluate(levels='all', pt_scratch=RUN.parent / 'exp1125_pt_scratch',
             extra_chord_manifests=manifests, output=out)
    for job in jobs:
        record(job, 'evaluated', log_dir=out, metrics=str(out / 'multilevel_metrics.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--render', choices=(*SCORE_INPUTS, *OUTPUT_MODES, *DIRECT_DLM_MODES))
    action.add_argument('--output-ablation', action='store_true')
    action.add_argument('--direct-dlm', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    experiment = INPUTS
    if args.direct_dlm or args.render in DIRECT_DLM_MODES:
        experiment = DIRECT_DLM
    elif args.output_ablation or args.render in OUTPUT_MODES:
        experiment = OUTPUTS
    if args.render:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(experiment.variants.index(args.render)):
            raise RuntimeError('Render pool must see its assigned GPU')
        render(experiment, args.render)
    else:
        run(experiment)


if __name__ == '__main__':
    main()
