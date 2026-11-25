#!/usr/bin/env python3
"""Train chord AR variants, then sample and score the official test cohort."""

# ruff: noqa: E402

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.evaluate_exp1125_chord import evaluate, write_chord_prediction
from script.prepare_exp1125_chord import DATA, RUN, write_json
from script.run_exp1125_chord import launch, process_alive, read_json, training_result
from script.train_exp1125_chord import SEED
from src.data_process.dinr_chord_dataset import COARSE_TIMING_BINS, TIMING_BINS, ChordBatches, MODES
from src.model.dinr_chord import DINRChordModel
from src.model.dinr_chord_ar import DINRChordARModel, sample_head
from src.model.dinr_chord_ar_input import DINRChordARInputModel
from src.model.dinr_chord_ar_output import DINRChordAROutputModel

AR = RUN.parent / 'exp1125_chord_ar'
DECODING_AUDIT = AR / 'decoding_audit'
WORKERS = 8
DRAWS = 3
TEMPERATURE = 1.0
TOP_P = 0.95


def record(task, status, log_dir=AR, **extras):
    row = dict(task, status=status, **extras)
    with (log_dir / 'runs.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
    print(task['mode'], status, task.get('work', ''), task.get('draw', ''), extras, flush=True)


def prepare_tasks(mode, checkpoint, digest, scores):
    index = MODES.index(mode)
    output = AR / mode / 'test_official'
    tasks = []
    for work_idx, (work, score) in enumerate(sorted(scores.items())):
        for draw in range(DRAWS):
            tasks.append(dict(mode=mode, gpu=index, work=work, draw=draw, training_seed=SEED,
                              sampling_seed=42000 + index * 1000 + work_idx * DRAWS + draw,
                              chords=len(score['score']), windows=len(score['spans']),
                              note_count=len(score['pitch']), checkpoint=str(checkpoint),
                              checkpoint_sha256=digest, config=str(AR / mode / 'run_config.json'),
                              output_dir=str(output / 'samples' / f'{work_idx:02d}' / f'{draw:02d}')))
    return sorted(tasks, key=lambda task: (-task['chords'], task['work'], task['draw']))


def prediction(task):
    summary = Path(task['output_dir']) / 'result.json'
    if not summary.is_file():
        return None
    row = read_json(summary)
    if any(row.get(key) != value for key, value in task.items()):
        raise ValueError(f'Existing prediction has a different task identity: {summary}')
    outputs = [row]
    if task.get('continuous_output'):
        if 'continuous_prediction' not in row:
            return None
        outputs.append(row['continuous_prediction'])
    if all(Path(output[field]).is_file() and Path(output[field]).stat().st_size > 0
           for output in outputs for field in ('midi', 'notes')):
        return row
    return None


def init_worker(checkpoints, variant=None):
    global models, scores, device
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    scores = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True, mmap=True)
    models = {}
    for architecture, checkpoint in checkpoints.items():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if architecture == 'ar_input':
            model = DINRChordARInputModel(variant)
        elif architecture == 'ar_output':
            model = DINRChordAROutputModel(variant)
        else:
            model = {'ar': DINRChordARModel, 'parallel': DINRChordModel}[architecture]()
        model.load_state_dict(state, assign=True)
        models[architecture] = model.to(device).eval()


def run_task(task):
    start = time.monotonic()
    output = Path(task['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        score = scores[task['work']]
        architecture = task.get('architecture', 'ar')
        timing_bins = COARSE_TIMING_BINS if architecture == 'ar_output' else TIMING_BINS
        batches = ChordBatches(score, task['mode'], device, raw_score=architecture == 'ar_input',
                              timing_bins=timing_bins)
        batch = batches(torch.arange(len(batches)))
        attention = batch[2]
        seed = task['sampling_seed']
        rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        model = models[architecture]
        locations = None
        continuous = task.get('continuous_output', False)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if architecture in ('ar', 'ar_input', 'ar_output'):
                sampled = model.sample_windows(*batch[:4], rng, TEMPERATURE, TOP_P, return_locations=continuous)
            else:
                choose = partial(sample_head, rng=rng, temperature=TEMPERATURE, top_p=TOP_P)
                sampled = model.predict(batch, choose=choose)
            if continuous:
                sampled, locations = sampled
                locations = locations[attention].cpu()
            tokens = sampled[attention].cpu()
        row = write_chord_prediction(task['work'], score, tokens, task['mode'], output / 'sample.mid',
                                     output / 'notes.npz', timing_bins=timing_bins)
        if architecture == 'ar_output':
            advancing = score['score'][:, 0] > 0
            fields = (tokens[advancing, 0], tokens[:, 1], tokens[:, 2])
            row['edge_bins'] = {key: dict(low=int((field == 0).sum()), high=int((field == high).sum()), total=len(field))
                                for key, field, high in zip(('ioi', 'duration', 'velocity'), fields,
                                                            (timing_bins - 1, timing_bins - 1, 127), strict=True)}
        if continuous:
            row['continuous_prediction'] = write_chord_prediction(
                task['work'], score, tokens, task['mode'], output / 'continuous.mid', output / 'continuous.npz',
                timing_bins=timing_bins, continuous_timing=locations)
        row.update(task, worker_pid=os.getpid(), seconds=time.monotonic() - start)
        (output / 'task.log').write_text(json.dumps(row) + '\n')
        write_json(output / 'result.json', row)
        return row
    except Exception:
        (output / 'task.log').write_text(traceback.format_exc())
        raise


def execute_tasks(tasks, checkpoints, log_dir=AR, variant=None):
    pending = [task for task in tasks if prediction(task) is None]
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(checkpoints, variant)) as pool:
            futures = {}
            for task in pending:
                record(task, 'queued', log_dir=log_dir)
                futures[pool.submit(run_task, task)] = task
            failures = []
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    record(task, 'sample_failed', log_dir=log_dir, error=str(error))
                    failures.append(task)
                else:
                    record(task, 'sampled', log_dir=log_dir, worker_pid=row['worker_pid'], seconds=row['seconds'])
            if failures:
                raise RuntimeError(f'{len(failures)} generation tasks failed')


def prediction_items(tasks):
    by_work = defaultdict(list)
    for task in tasks:
        result = prediction(task)
        if result is None:
            raise ValueError(f'Missing prediction for {task["work"]}, draw {task["draw"]}')
        by_work[task['work']].append(result)
    return [dict(work=work, draws=rows) for work, rows in sorted(by_work.items())]


def render(mode):
    training = AR / mode
    completion = training_result(training)
    if completion is None:
        raise RuntimeError(f'{mode}: no verified checkpoint')
    checkpoint = training / 'best.pt'
    output = training / 'test_official'
    output.mkdir(parents=True, exist_ok=True)
    score_data = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True, mmap=True)
    tasks = prepare_tasks(mode, checkpoint, completion['sha256'], score_data)
    if len(tasks) != 69 or sum(t['windows'] for t in tasks) != 276:
        raise ValueError(f'{mode}: unexpected test coverage')
    write_json(output / 'dispatch.json', dict(workers=WORKERS, ordering='descending chord count',
               temperature=TEMPERATURE, top_p=TOP_P, tasks=tasks))
    execute_tasks(tasks, {'ar': checkpoint})
    write_json(output / 'prediction_manifest.json', dict(mode=mode, checkpoint=str(checkpoint),
               checkpoint_sha256=completion['sha256'], best_step=completion['best_step'],
               draws=DRAWS, workers=WORKERS, temperature=TEMPERATURE, top_p=TOP_P,
               items=prediction_items(tasks)))


def render_decoding_audit(mode):
    output = DECODING_AUDIT / mode
    output.mkdir(parents=True, exist_ok=True)
    score_data = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True, mmap=True)
    groups = {'ar': (AR / mode, 'median', 1), 'parallel': (RUN / mode, 'top_p', DRAWS)}
    checkpoints, manifests, case_tasks = {}, {}, {}
    for architecture, (training, decoding, draws) in groups.items():
        completion = training_result(training)
        checkpoint = training / 'best.pt'
        checkpoints[architecture] = checkpoint
        manifests[architecture] = dict(mode=mode, checkpoint=str(checkpoint),
                                       checkpoint_sha256=completion['sha256'], best_step=completion['best_step'],
                                       draws=draws, workers=WORKERS, decoding=decoding)
        case_tasks[architecture] = []
        group = prepare_tasks(mode, checkpoint, completion['sha256'], score_data)
        for task in group:
            if task['draw'] >= draws:
                continue
            original = Path(task['output_dir'])
            task.update(architecture=architecture, decoding=decoding, config=str(training / 'run_config.json'),
                        sampling_seed=None if decoding == 'median' else task['sampling_seed'] + 100000,
                        output_dir=str(output / architecture / original.parent.name / original.name))
            case_tasks[architecture].append(task)
    tasks = sorted((task for group in case_tasks.values() for task in group),
                   key=lambda task: (-task['chords'], task['work'], task['architecture'], task['draw']))
    if len(tasks) != 92:
        raise ValueError(f'{mode}: expected four decodes for each of 23 scores')
    write_json(output / 'dispatch.json', dict(workers=WORKERS, temperature=TEMPERATURE, top_p=TOP_P, tasks=tasks))
    execute_tasks(tasks, checkpoints, log_dir=DECODING_AUDIT)
    for architecture, report in manifests.items():
        write_json(output / f'{architecture}.json', dict(report, items=prediction_items(case_tasks[architecture])))


def decoding_audit():
    DECODING_AUDIT.mkdir(parents=True, exist_ok=True)
    processes = [(mode, launch('run_exp1125_chord_ar', ['--render-decoding-audit', mode],
                               DECODING_AUDIT / f'{mode}.log', gpu)) for gpu, mode in enumerate(MODES)]
    failures = [mode for mode, process in processes if process.wait()]
    if failures:
        raise RuntimeError(f'Decoding audit failed for {failures}')
    manifests = {}
    for mode in MODES:
        for case, label in (('ar', 'AR median'), ('parallel', 'Parallel sampled')):
            report = read_json(DECODING_AUDIT / mode / f'{case}.json')
            if len(report['items']) != 23 or any(len(item['draws']) != report['draws'] for item in report['items']):
                raise ValueError(f'{mode}/{case}: incomplete audit predictions')
            manifests[f'{label} {mode}'] = report
    evaluate(levels='all', pt_scratch=RUN.parent / 'exp1125_pt_scratch', chord_ar=AR,
             extra_chord_manifests=manifests, output=DECODING_AUDIT)


def pipeline():
    AR.mkdir(exist_ok=True)
    jobs_file = AR / 'training_jobs.json'
    if jobs_file.exists():
        jobs = read_json(jobs_file)
    else:
        jobs = []
        for gpu, mode in enumerate(MODES):
            output = AR / mode
            output.mkdir(exist_ok=True)
            process = launch('train_exp1125_chord', [mode, '--autoregressive'], output / 'train.log', gpu)
            job = dict(mode=mode, gpu=gpu, training_seed=SEED, sampling_seed=None, pid=process.pid,
                       checkpoint=str(output / 'best.pt'), config=str(output / 'run_config.json'),
                       output_dir=str(output))
            jobs.append(job)
            record(job, 'training')
        write_json(jobs_file, jobs)

    training = {job['mode']: job for job in jobs}
    rendering = {}
    while training or rendering:
        for mode, job in list(training.items()):
            output = AR / mode
            completion = training_result(output)
            if completion is None:
                if process_alive(job['pid']):
                    continue
                raise RuntimeError(f'{mode}: training stopped without a verified checkpoint')
            event = {k: v for k, v in job.items() if k != 'status'}
            record(event, 'trained', best_step=completion['best_step'], sha256=completion['sha256'])
            if not (output / 'test_official/prediction_manifest.json').exists():
                process = launch('run_exp1125_chord_ar', ['--render', mode], output / 'infer.log', job['gpu'])
                rendering[mode] = process, event
                record(event, 'rendering', worker_count=WORKERS)
            del training[mode]
        for mode, (process, job) in list(rendering.items()):
            code = process.poll()
            if code is None:
                continue
            if code or not (AR / mode / 'test_official/prediction_manifest.json').is_file():
                record(job, 'render_failed', exit_code=code)
                raise RuntimeError(f'{mode}: generation did not complete')
            record(job, 'rendered', samples=23 * DRAWS)
            del rendering[mode]
        if training or rendering:
            time.sleep(10)

    evaluate(levels='all', pt_scratch=RUN.parent / 'exp1125_pt_scratch', chord_ar=AR)
    for job in jobs:
        record({k: v for k, v in job.items() if k != 'status'}, 'evaluated',
               metrics=str(AR / 'multilevel_metrics.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--render', choices=MODES)
    action.add_argument('--render-decoding-audit', choices=MODES)
    action.add_argument('--decoding-audit', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    render_mode = args.render or args.render_decoding_audit
    if render_mode:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(MODES.index(render_mode)):
            raise RuntimeError('Render worker must start with its assigned physical GPU visible')
        job = render if args.render else render_decoding_audit
        job(render_mode)
    elif args.decoding_audit:
        decoding_audit()
    else:
        pipeline()


if __name__ == '__main__':
    main()
