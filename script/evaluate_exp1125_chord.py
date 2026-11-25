#!/usr/bin/env python3
"""Render complete official scores and score design1125 in milliseconds."""

# ruff: noqa: E402

import argparse
from collections import defaultdict
import csv
from functools import partial
import json
from pathlib import Path
import sys

from miditoolkit import Instrument, MidiFile, Note, TempoChange
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.prepare_exp1125_chord import DATA, RUN, RAW, load_beat_mapper, sha256, write_json
from script.prepare_exp1123_measure_official import PT2_EVAL, representation, intervals_inside_segments
from script.evaluate_exp1116_baselines import align_note_ons, read_note_on_stream
from script.evaluate_exp1117_five_main_metrics import average_reference_correlations, beat_mean_curve
from score_official_split import extract, grid, measure_edges, read_a2_draws, read_prediction, score_time_ms
from src.data_process.beat_mapping import map_score_beats, valid_beat_intervals
from src.data_process.build_dinr_beat_sidecars import beat_grid
from src.data_process.dinr_chord_dataset import TIMING_BINS, ChordBatches, MODES, decode_tokens
from src.model.dinr_chord import DINRChordModel

A2 = ROOT / 'results/exp1123_A2_measure_official_e4_valid50_original_3gpu_20261124/test_official'
METRICS = ('tempo_r', 'velocity_r', 'ioi_mae_ms', 'duration_mae_ms', 'velocity_mae')
MAE_FIELDS = {'ioi_mae_ms': 'ioi', 'duration_mae_ms': 'duration', 'velocity_mae': 'velocity'}
CORRELATIONS = {'tempo_r': 'tempo', 'velocity_r': 'measure_velocity'}
BEAT_CORRELATIONS = {'beat_tempo_r': 'beat_tempo', 'beat_velocity_r': 'beat_velocity'}
CHORD_CORRELATIONS = {'onset_group_tempo_r': 'onset_group_tempo', 'chord_velocity_r': 'chord_velocity'}


def export_chords(pitches, start, duration, velocity, path):
    midi = MidiFile(ticks_per_beat=1000)
    midi.tempo_changes = [TempoChange(60, 0)]
    track_ends = []
    # Separate overlapping repeats of one pitch so MIDI preserves both ends.
    for idx in np.lexsort((pitches, start)):
        pitch, onset, end, vel = (int(v) for v in (pitches[idx], start[idx], start[idx] + duration[idx], velocity[idx]))
        track = next((i for i, ends in enumerate(track_ends) if ends[pitch] <= onset), len(track_ends))
        if track == len(track_ends):
            track_ends.append(np.zeros(128, dtype=np.int64))
            midi.instruments.append(Instrument(0, name='Piano'))
        track_ends[track][pitch] = end
        midi.instruments[track].notes.append(Note(vel, pitch, onset, end))
    midi.dump(str(path))
    decoded = read_note_on_stream(path)
    expected = sorted(zip(pitches, start, duration, velocity, strict=True))
    actual = sorted(zip(*(decoded[k] for k in ('pitch', 'start', 'duration', 'velocity')), strict=True))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)


def write_chord_prediction(work, score, tokens, mode, midi, notes, timing_bins=TIMING_BINS, continuous_timing=None):
    values = decode_tokens(tokens, score['score'], mode, timing_bins=timing_bins,
                           continuous_timing=continuous_timing).numpy()
    if values.shape != score['score'].shape or not np.isfinite(values).all():
        raise ValueError(f'{work}: incomplete or non-finite predictions')
    same_onset = score['score'][:, 0].numpy() == 0
    if np.any(tokens.numpy()[same_onset, 0]) or np.any(values[same_onset, 0]):
        raise ValueError(f'{work}: a structural zero advanced the clock')
    membership = score['membership'].numpy()
    pitches, ticks = score['pitch'].numpy(), score['ticks'].numpy()
    start = np.floor(np.cumsum(values[:, 0], dtype=np.float64) + .5).astype(np.int64)[membership]
    duration = np.rint(values[:, 1]).astype(np.int64)[membership]
    velocity = values[:, 2].astype(np.int64)[membership]
    if not np.array_equal(np.diff(start) == 0, np.diff(ticks) == 0):
        raise ValueError(f'{work}: rendered onsets do not match the score simultaneity')
    export_chords(pitches, start, duration, velocity, midi)
    np.savez_compressed(notes, work=work, pitch=pitches, ticks=ticks, start=start,
                        duration=duration, velocity=velocity, chord_predictions=values)
    return dict(work=work, midi=str(midi), notes=str(notes), note_count=len(pitches),
                chords=len(tokens), windows=len(score['spans']))


@torch.inference_mode()
def render(mode, smoke=False):
    device = torch.device('cuda:0')
    training = RUN / ('smoke' if smoke else mode)
    checkpoint = training / 'best.pt'
    completion = json.loads((training / 'training_complete.json').read_text())
    if sha256(checkpoint) != completion['sha256']:
        raise ValueError(f'{mode}: inference checkpoint differs from the selected weights')
    output = training / 'test_official'
    output.mkdir(exist_ok=True)
    model = DINRChordModel().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    scores = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True)
    items = []
    for index, (work, score) in enumerate(scores.items()):
        data = ChordBatches(score, mode, device)
        pieces = []
        for indices in torch.arange(len(data)).split(16):
            batch = data(indices)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.predict(batch)
            pieces.append(prediction[batch[2]].cpu())
        tokens = torch.cat(pieces)
        items.append(write_chord_prediction(work, score, tokens, mode, output / f'{index:02d}.mid',
                                            output / f'{index:02d}.npz'))
        print(mode, f'{len(items)}/{len(scores)}', work, flush=True)
    if len(items) != 23 or sum(row['windows'] for row in items) != 92:
        raise ValueError('Inference did not cover the complete score cohort')
    write_json(output / 'prediction_manifest.json', dict(mode=mode, checkpoint=str(checkpoint),
               checkpoint_sha256=completion['sha256'], best_step=completion['best_step'], draws=1, items=items))


def features(notes, geometry, segment=None, beats=None):
    measure, quarters, group_ticks, groups = geometry
    tempo = extract(notes, measure, np.zeros(len(notes['start'])), quarters, group_ticks, groups)['tempo']
    values = {field: np.asarray(notes[field], dtype=float).copy() for field in ('duration', 'velocity')}
    if segment is None:
        ioi = np.diff(np.minimum.reduceat(notes['start'], groups))
    else:
        keep = segment >= 0
        onset = np.minimum.reduceat(np.where(keep, notes['start'], np.inf), groups)
        segment_min = np.minimum.reduceat(np.where(keep, segment, len(segment)), groups)
        segment_max = np.maximum.reduceat(segment, groups)
        same = (segment_min == segment_max) & (segment_max >= 0)
        adjacent = same[1:] & same[:-1] & (segment_max[1:] == segment_max[:-1])
        ioi = np.full(len(groups) - 1, np.nan)
        ioi[adjacent] = onset[1:][adjacent] - onset[:-1][adjacent]
        ioi[ioi <= 0] = np.nan
        tempo[~intervals_inside_segments(measure['edges'], group_ticks, groups, segment)] = np.nan
        for vals in values.values():
            vals[~keep] = np.nan
    values.update(tempo=tempo, ioi=ioi)
    if beats is not None:
        edges, quarters = beats
        first = np.minimum.reduceat(notes['start'], groups)
        anchors, supported = map_score_beats(edges, group_ticks, first)
        valid, durations = valid_beat_intervals(anchors, supported)
        if segment is not None:
            valid &= intervals_inside_segments(edges, group_ticks, groups, segment)
        bpm = np.full(len(quarters), np.nan)
        bpm[valid] = 60000 * quarters[valid] / durations[valid]
        values['beat_tempo'] = bpm
    return values


def read_json(path):
    return json.loads(path.read_text())


def score_correlations(candidates, references, masks, fields):
    result = {}
    for metric, field in fields.items():
        values = [average_reference_correlations(row[field], [ref[field] for ref in references], masks[field])['mean']
                  for row in candidates]
        valid = [value for value in values if value is not None]
        result[metric] = float(np.mean(valid)) if valid else None
    return result


def score_candidates(candidates, references, masks):
    result = score_correlations(candidates, references, masks, CORRELATIONS)
    for metric, field in MAE_FIELDS.items():
        a, b = (np.stack([row[field][masks[field]] for row in group]).mean(0) for group in (candidates, references))
        result[metric] = float(np.mean(np.abs(a - b)))
    return result


def score_fine_levels(candidates, references, note_mask, chord_membership, beat_membership, quarter_lengths):
    all_rows = [*references, *(row for draws in candidates.values() for row in draws)]
    num_chords = int(chord_membership.max()) + 1
    for row in all_rows:
        ioi = row['ioi']
        bpm = np.full(ioi.shape, np.nan)
        np.divide(60000 * quarter_lengths, ioi, out=bpm, where=ioi > 0)
        row['onset_group_tempo'] = np.round(bpm, 8)
        row['chord_velocity'] = beat_mean_curve(row['velocity'], chord_membership, note_mask, num_chords)
        row['beat_velocity'] = beat_mean_curve(row['velocity'], beat_membership, note_mask, len(row['beat_tempo']))
    fields = BEAT_CORRELATIONS | CHORD_CORRELATIONS
    masks = {field: np.isfinite(np.stack([row[field] for row in all_rows])).all(0)
             for field in fields.values()}
    if any(mask.sum() < 3 for mask in masks.values()):
        raise ValueError('Insufficient common chord or onset-group support')
    results = {name: score_correlations(draws, references, masks, fields)
               for name, draws in candidates.items()}
    if len(references) > 1:
        held_out = [score_correlations([row], references[:i] + references[i+1:], masks, fields)
                    for i, row in enumerate(references)]
        results['Human LOO'] = summarize(held_out, fields)
    return results, {name: int(mask.sum()) for name, mask in masks.items()}


def score_work(work, refs, pt_models, a2, chord_predictions, get_bars, chord_layout=None):
    score_path = RAW / work
    midi = MidiFile(str(score_path))
    mapping, pitch, ticks, order, notes = representation(midi)
    edges = measure_edges(midi, get_bars)
    grid_info = grid(edges, score_time_ms(midi, mapping, edges), score_time_ms(midi, mapping, ticks), ticks)
    quarters = np.diff(edges) / midi.ticks_per_beat
    group_ticks, groups = np.unique(ticks, return_index=True)
    geometry = (grid_info, quarters, group_ticks, groups)
    beats = None
    if chord_layout is not None:
        beat_edges = beat_grid(midi, get_bars)[0]
        beats = (beat_edges, np.diff(beat_edges) / midi.ticks_per_beat)
    featurize = partial(features, geometry=geometry, beats=beats)
    reference_rows = []
    for ref in refs:
        with np.load(ref['path']) as raw:
            np.testing.assert_array_equal(raw['pitch'], pitch)
            values = {key: raw[key][order].astype(float) for key in ('start', 'duration', 'velocity')}
            reference_rows.append(featurize(values, segment=raw['segment'][order]))
    score = dict(start=score_time_ms(midi, mapping, ticks),
                 duration=np.array([note.end - note.start for note in notes])[order],
                 velocity=np.array([note.velocity for note in notes])[order])
    score_fields = featurize(score)
    score_fields['tempo'] = np.round(60000 * quarters / grid_info['score_ms'], 8)
    if beats is not None:
        score_fields['beat_tempo'] = np.round(60000 * beats[1] / np.diff(score_time_ms(midi, mapping, beats[0])), 8)
    reference_paths = {(RAW / ref['performance']).resolve() for ref in refs}
    candidates = {'Score MIDI': [score_fields], 'Measure A2': []}
    for name, manifest in pt_models.items():
        if manifest.get('expected_draws', 3) == 1:
            paths = manifest['prediction_paths']
            if (len(paths) != 1 or Path(manifest['score_midi']).resolve() != score_path.resolve() or
                    {Path(path).resolve() for path in manifest['ground_truth_paths']} != reference_paths):
                raise ValueError(f'Single-render prediction identity differs for {score_path}')
            pt_notes = [align_note_ons(read_note_on_stream(paths[0]), pitch[order])]
        else:
            pt_notes, _ = read_prediction(manifest, pitch[order], score_path, reference_paths)
        candidates[name] = [featurize(row) for row in pt_notes]
    for sample in read_a2_draws(a2, work, grid_info, pitch[order], ticks, allow_extra_trailing_measures=True):
        obs = featurize(sample)
        obs['tempo'] = 60000 * quarters / sample['measure_duration_ms']
        candidates['Measure A2'].append(obs)
    for mode, predicted in chord_predictions.items():
        candidates[mode] = []
        for draw in predicted.get('draws', (predicted,)):
            with np.load(draw['notes']) as stream:
                if str(stream['work']) != work:
                    raise ValueError(f'Prediction identity differs for {work}')
                np.testing.assert_array_equal(stream['pitch'], pitch[order])
                np.testing.assert_array_equal(stream['ticks'], ticks)
                candidates[mode].append(featurize(stream))

    all_rows = [*reference_rows, *(row for draws in candidates.values() for row in draws)]
    masks = {field: np.isfinite(np.stack([row[field] for row in all_rows])).all(0)
             for field in ('tempo', *MAE_FIELDS.values())}
    for row in all_rows:
        row['measure_velocity'] = beat_mean_curve(row['velocity'], grid_info['ids'], masks['velocity'], len(quarters))
    masks['measure_velocity'] = np.isfinite(np.stack([row['measure_velocity'] for row in all_rows])).all(0)
    if not all(mask.any() for mask in masks.values()):
        raise ValueError(f'Missing common evaluation support for {work}')
    result = {name: score_candidates(draws, reference_rows, masks) for name, draws in candidates.items()}
    if len(reference_rows) > 1:
        splits = [score_candidates([row], reference_rows[:i] + reference_rows[i+1:], masks)
                  for i, row in enumerate(reference_rows)]
        result['Human LOO'] = summarize(splits)
    curves = {name: np.nan_to_num(np.median(np.stack([row['tempo'] for row in candidates[name]]), axis=0), nan=0.)
              for name in candidates}
    curves['Human'] = np.nan_to_num(np.median(np.stack([row['tempo'] for row in reference_rows]), axis=0), nan=0.)
    row = dict(work=work, metrics=result, support={k: int(v.sum()) for k, v in masks.items()}, references=len(refs))
    if chord_layout is not None:
        np.testing.assert_array_equal(chord_layout['ticks'].numpy(), ticks)
        np.testing.assert_array_equal(chord_layout['pitch'].numpy(), pitch[order])
        beat_membership = (np.searchsorted(beats[0], ticks, side='right') - 1).clip(0, len(beats[1]) - 1)
        row['fine_metrics'], row['fine_support'] = score_fine_levels(
            candidates, reference_rows, masks['velocity'], chord_layout['membership'].numpy(),
            beat_membership, np.diff(group_ticks) / midi.ticks_per_beat,
        )
    return row, curves, masks['tempo']


def summarize(rows, metrics=METRICS):
    result = {}
    for metric in metrics:
        values = [row[metric] for row in rows if row[metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
    return result


def save_level_metrics(per_work, checkpoints, models, all_levels, output):
    correlations = (CORRELATIONS | BEAT_CORRELATIONS | CHORD_CORRELATIONS) if all_levels else CHORD_CORRELATIONS
    fields = (*correlations, *MAE_FIELDS) if all_levels else tuple(correlations)
    details = []
    for row in per_work:
        metrics = {model: {key: (row['metrics'][model] | values)[key] for key in fields}
                   for model, values in row['fine_metrics'].items()}
        support = row['support'] | row['fine_support'] if all_levels else {
            key: row['fine_support'][key] for key in CHORD_CORRELATIONS.values()}
        details.append(dict(work=row['work'], metrics=metrics, support=support, references=row['references']))
    report = {}
    for model in models:
        scores = [row['metrics'][model] for row in details if model in row['metrics']]
        report[model] = dict(summarize(scores, fields), works=len(scores),
                             **{f'{field}_works': sum(row[field] is not None for row in scores) for field in correlations})
    columns = ('model', 'works', *fields, *(f'{field}_works' for field in correlations))
    path = output / ('multilevel_metrics.csv' if all_levels else 'chord_correlations.csv')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(model=model, **values) for model, values in report.items())
    protocol = dict(onset_group_tempo='60000 times the score quarter-note distance between consecutive distinct '
                                     'onsets, divided by their positive performance IOI in milliseconds.',
                    chord_velocity='Arithmetic mean of aligned note velocities in each score onset-and-duration chord.',
                    masking='Common positive IOI support and common HMM retained notes across draws, models and humans.',
                    aggregation='Pearson per draw and human reference within score, then average scores equally.')
    if all_levels:
        protocol.update(measure_tempo='Measure quarter-note BPM, using native durations for Measure A2.',
                        measure_velocity='Mean note velocity within each score measure.',
                        beat_tempo='Quarter-note-equivalent BPM on the score beat grid. Map beat boundaries from '
                                   'first onsets of aligned notes, including for Measure A2.',
                        beat_velocity='Mean velocity of common aligned notes that start within each score beat.',
                        mae='Same native-millisecond and MIDI-velocity protocol as five_metrics.json.')
    write_json(path.with_suffix('.json'), dict(protocol=protocol, metrics=report, per_work=details,
                                              checkpoints=checkpoints))
    print(path.read_text(), end='', flush=True)


def evaluate(levels=None, pt_scratch=None, chord_ar=None, extra_chord_manifests=None,
             extra_pt_manifests=None, output=None):
    refs = defaultdict(list)
    for row in read_json(PT2_EVAL / 'reference_manifest.json'):
        refs[row['score']].append(row)
    if len(refs) != 23 or sum(map(len, refs.values())) != 101:
        raise ValueError('Expected the 23 official test scores and 101 references')
    pt_reports = {}
    if levels == 'all':
        pt_reports['PT base'] = read_json(PT2_EVAL / 'base/prediction_manifest.json')
    pt_reports['PT2'] = read_json(PT2_EVAL / 'rendering_reproduced/prediction_manifest.json')
    if pt_scratch is not None:
        pt_reports['PT-scratch'] = read_json(pt_scratch / 'test_official/prediction_manifest.json')
    if extra_pt_manifests:
        pt_reports.update(extra_pt_manifests)
    pt_models = {name: {row['score_source']: dict(row, expected_draws=report['num_samples'])
                        for row in report['items']} for name, report in pt_reports.items()}
    a2 = {row['work']: A2 / f'work_{row["index"]:02d}/predictions.npz' for row in read_json(A2 / 'dispatch.json')['tasks']}
    manifests = {mode: read_json(RUN / mode / 'test_official/prediction_manifest.json') for mode in MODES}
    if chord_ar is not None:
        for mode in MODES:
            training = chord_ar / mode
            report = read_json(training / 'test_official/prediction_manifest.json')
            completion = read_json(training / 'training_complete.json')
            if (report['checkpoint_sha256'] != completion['sha256'] or len(report['items']) != 23 or
                    report['draws'] != 3 or any(len(item['draws']) != 3 for item in report['items'])):
                raise ValueError(f'{mode}: incomplete AR run or checkpoint mismatch')
            manifests[f'AR {mode}'] = report
    if extra_chord_manifests:
        manifests.update(extra_chord_manifests)
    predictions = {mode: {row['work']: row for row in report['items']} for mode, report in manifests.items()}
    if not all(refs.keys() == rows.keys() for rows in (a2, *pt_models.values(), *predictions.values())):
        raise ValueError('Prediction manifests have different score coverage')
    chord_layouts = torch.load(DATA / 'scores.pt', map_location='cpu', weights_only=True) if levels else {}
    if levels and refs.keys() != chord_layouts.keys():
        raise ValueError('Chord layout score coverage differs')
    get_bars, _ = load_beat_mapper()
    per_work, tempo_curves = [], {}
    for work, references in sorted(refs.items()):
        models = {mode: by_work[work] for mode, by_work in predictions.items()}
        pt_items = {name: by_work[work] for name, by_work in pt_models.items()}
        row, curves, mask = score_work(work, references, pt_items, a2[work], models, get_bars, chord_layouts.get(work))
        per_work.append(row)
        tempo_curves[work] = (curves, mask)
    names = (*manifests, *pt_models, 'Measure A2', 'Score MIDI', 'Human LOO')
    checkpoints = {k: {x: v[x] for x in ('checkpoint', 'checkpoint_sha256', 'best_step')} for k, v in manifests.items()}
    if 'PT base' in pt_reports:
        checkpoint = Path(pt_reports['PT base']['model_path'])
        checkpoints['PT base'] = dict(checkpoint=str(checkpoint),
                                      checkpoint_sha256=sha256(checkpoint / 'model.safetensors'),
                                      prediction_manifest=str(PT2_EVAL / 'base/prediction_manifest.json'))
    if pt_scratch is not None:
        run = read_json(pt_scratch / 'completion.json')
        if Path(pt_reports['PT-scratch']['model_path']) != Path(run['weights']).parent:
            raise ValueError('Scratch prediction checkpoint does not match the final training checkpoint')
        checkpoints['PT-scratch'] = dict(checkpoint=str(Path(run['weights']).parent),
                                         checkpoint_sha256=run['sha256'], step=run['optimizer_steps'])
    for name, report in (extra_pt_manifests or {}).items():
        checkpoints[name] = dict(checkpoint=report['model_path'], checkpoint_sha256=report['checkpoint_sha256'],
                                 step=report['step'], decoding=report['protocol'])
    if levels:
        save_level_metrics(per_work, checkpoints, names, all_levels=levels == 'all',
                           output=output or chord_ar or pt_scratch or RUN)
        return
    result = {}
    for model in names:
        scores = [row['metrics'][model] for row in per_work if model in row['metrics']]
        result[model] = dict(summarize(scores), works=len(scores),
                             tempo_works=sum(row['tempo_r'] is not None for row in scores),
                             velocity_works=sum(row['velocity_r'] is not None for row in scores))
    protocol = dict(timing='Absolute native performance milliseconds for all methods and HMM references.',
                    ioi='Between consecutive distinct score onsets, using earliest retained aligned note as human anchor. '
                        'Common HMM segment; omit first onset, structural zero IOI and nonpositive human IOI.',
                    mae='Absolute difference of prediction means and human-reference means at matched positions, '
                        'then average positions within work and weight works equally.',
                    tempo='Measure quarter-note BPM, using native durations for Measure A2.',
                    velocity='Mean velocity of notes starting in each measure, on common note support.')
    with (RUN / 'five_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('model', 'works', *METRICS, 'tempo_works', 'velocity_works'))
        writer.writeheader()
        writer.writerows(dict(model=key, **row) for key, row in result.items())
    write_json(RUN / 'five_metrics.json', dict(protocol=protocol, metrics=result, per_work=per_work,
                                            checkpoints=checkpoints))
    plot_curves(tempo_curves)
    print((RUN / 'five_metrics.csv').read_text(), end='', flush=True)


def plot_curves(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selected = ('Bach/Fugue/bwv_854', 'Ravel/Miroirs/3_Une_Barque', 'Beethoven/Piano_Sonatas/32-1',
                'Chopin/Etudes_op_25/4', 'Beethoven/Piano_Sonatas/4-1', 'Bach/Prelude/bwv_885')
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
    colors = dict(zip(('Score MIDI', 'Human', 'PT2', *MODES), ('#333333', '#1965ac', '#00846e', '#c6541b', '#a05aab', '#d39e10'), strict=True))
    for ax, work in zip(axes.flat, selected, strict=True):
        curves, support = rows[f'{work}/midi_score.mid']
        x = np.arange(1, len(support) + 1)
        for name, color in colors.items():
            ax.plot(x, np.where(support, curves[name], np.nan), label=name, color=color,
                    linestyle='--' if name == 'Score MIDI' else '-', linewidth=1.15)
        ax.set(title=work.replace('_', ' '), ylabel='BPM', xlabel='Score measure')
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.savefig(RUN / 'measure_tempo_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=(*MODES, 'score', 'score_chords', 'score_levels'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pt-scratch', type=Path)
    args = parser.parse_args()
    if args.pt_scratch is not None and args.job != 'score_levels':
        parser.error('--pt-scratch requires score_levels')
    torch.set_num_threads(1)
    score_jobs = {'score': None, 'score_chords': 'chord', 'score_levels': 'all'}
    if args.job in score_jobs:
        evaluate(score_jobs[args.job], args.pt_scratch)
    else:
        render(args.job, args.smoke)


if __name__ == '__main__':
    main()
