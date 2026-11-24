#!/usr/bin/env python3
"""Rebuild raw-aligned work and fixed-event sidecars with beat-median scaling."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from miditoolkit import MidiFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_process.beat_normalization import METHOD, normalize_performance
from src.data_process.build_dinr_beat_sidecars import build, beat_grid, load_beat_mapper, score_notes
from script.prepare_exp1116_beat import prepare
from script.audit_dinr_beat import audit
from script.plot_beat_duration_curves import load_curves, select_works, plot_matrix

PROJECT = ROOT.parent
SOURCE = PROJECT / 'data/ASAP_processed'
WORKS = PROJECT / 'data/ASAP_processed_beat_median_v1'
EVENTS = PROJECT / 'data/ASAP_processed_dinr_beat_median_v1'
FIXED = PROJECT / 'data/ASAP_processed_dinr_beat_median_e512_v1'
OUTPUT = ROOT / 'analysis/exp1117_beat_median'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_distributions(data_root=FIXED, output=OUTPUT):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    result = {}
    for split in ('train', 'valid', 'test'):
        data = torch.load(data_root / f'dinr_beat_{split}_e512_v1.pt', weights_only=False)
        e, t, v = data['events'], data['targets'], data['target_valid']
        beat = e[:, 0] == 0
        valid_beat, valid_note = beat & v[:, 0], ~beat & v[:, 0]
        note = ~beat & v[:, 1]
        distributions = {
            'beat log duration (ms)': t[valid_beat, 0].log().numpy(),
            'note log duration (ms)': t[valid_note, 0].log().numpy(),
            'beat log duration ratio': (t[valid_beat, 0] / e[valid_beat, 2]).log().numpy(),
            'note log duration ratio': (t[valid_note, 0] / e[valid_note, 2]).log().numpy(),
            'note onset dev': t[note, 1].numpy(),
            'note velocity': t[~beat & v[:, 2], 2].numpy(),
        }
        result[split] = {}
        for ax, (name, values) in zip(axes.flat, distributions.items()):
            percentiles = np.percentile(values, [0, 1, 25, 50, 75, 99, 100])
            result[split][name] = dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                percentiles=dict(zip(('min', 'p01', 'p25', 'median', 'p75', 'p99', 'max'), percentiles.tolist())))
            ax.hist(values, bins=100, density=True, histtype='step', label=split)
            ax.set(xlabel=name, ylabel='Density')
            ax.legend(frameon=False)
    fig.savefig(output / 'normalized_distributions.png', dpi=150)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_doc = ROOT / 'docs/exp1116_BEAT.md'
    snapshot = OUTPUT / 'previous_experiments.md'
    if not snapshot.exists():
        snapshot.write_bytes(original_doc.read_bytes())
    old_paths = set()
    for split in ('train', 'valid', 'test'):
        old = torch.load(PROJECT / f'data/ASAP_processed_dinr_beat_dev_v2/dinr_beat_{split}_b512_o50.pt', weights_only=False)
        old_paths.update(Path(row['source_sidecar']) for row in old['performances'])
    get_beats, _ = load_beat_mapper()
    overrides, records, sources = {}, [], []
    for index, old_path in enumerate(sorted(old_paths)):
        # Prefer original, unscaled labels over reversing an earlier scaling.
        raw_path = old_path.with_name(old_path.name.replace('ASAP_GROUPED_MEDIAN', 'ASAP_MUSICAL51'))
        payload = torch.load(raw_path, weights_only=False)
        if payload.get('performance_time_normalization') not in (None, 'none', 'raw_ms'):
            raise ValueError(f'Expected raw source: {raw_path}')
        old_payload = torch.load(old_path, weights_only=False)
        if (payload['score']['pitch'] != old_payload['score']['pitch'] or
            not np.allclose(np.asarray(payload['score']['score_raw'])[:, :3],
                            np.asarray(old_payload['score']['score_raw'])[:, :3], atol=1e-7)):
            raise ValueError(f'Raw and old score timing/pitch differ: {raw_path}')
        # Musical51 sidecars carry extra score pedal/features. Retain the exact
        # historical score payload, while rebuilding performance timing raw.
        payload['score'] = copy.deepcopy(old_payload['score'])
        old_by_source = {p['performance_source']: p for p in old_payload['performances']}
        midi = MidiFile(str(SOURCE / payload['meta']['score_source']))
        notes = score_notes(midi)
        ticks = np.asarray([note.start for note in notes])
        edges, _, score_ms, _, _ = beat_grid(midi, get_beats)
        converted = []
        for performance in payload['performances']:
            # Keep precisely the cohort of the historical experiment.
            if performance['performance_source'] not in old_by_source:
                continue
            old_perf = old_by_source[performance['performance_source']]
            old_values = np.asarray(old_perf['label_shared_raw'], dtype=float).copy()
            prior = old_perf.get('global_timing_scale')
            old_values[:, :2] /= float(prior['scale']) if prior else 1.0
            if not np.allclose(old_values, performance['label_shared_raw'], atol=1e-7, rtol=1e-10):
                raise ValueError(f'Raw label identity differs: {performance["performance_source"]}')
            normalized = normalize_performance(performance, ticks, edges, score_ms)
            converted.append(normalized)
            records.append(dict(work=old_path.parent.relative_to(SOURCE).as_posix(),
                                performance=performance['performance_id'],
                                **normalized['global_timing_scale']))
        if len(converted) != len(old_payload['performances']):
            raise ValueError('Raw performance cohort mismatch')
        payload['performances'] = converted
        payload['performances_by_source'] = {p['performance_source']: p for p in converted}
        payload['label_cache'], payload['perf_style_cache'] = {}, {}
        payload.pop('_cache_signature', None)
        payload['performance_time_normalization'] = METHOD
        payload['meta'] = copy.deepcopy(payload['meta'])
        payload['meta'].update(time_normalization=METHOD, performance_count=len(converted))
        payload['normalization_provenance'] = dict(raw_source=str(raw_path), raw_sha256=digest(raw_path),
            original_experiment_source=str(old_path), original_sha256=digest(old_path),
            protocol='one factor per complete performance; median log(D_perf_beat / D_score_beat); group mean + linear interpolation; no endpoint extrapolation')
        destination = WORKS / old_path.relative_to(SOURCE)
        destination = destination.with_name(destination.name.replace('ASAP_GROUPED_MEDIAN', 'ASAP_BEAT_MEDIAN').replace('ASAP_MUSICAL51', 'ASAP_BEAT_MEDIAN'))
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        overrides[str(old_path.resolve())] = str(destination.resolve())
        sources.append(dict(path=str(destination), sha256=digest(destination), **payload['normalization_provenance']))
        if index % 20 == 0:
            print(f'normalized {index+1}/{len(old_paths)} works', flush=True)
    (OUTPUT / 'normalization_manifest.json').write_text(json.dumps(dict(method=METHOD, works=sources,
        performances=records), indent=2) + '\n')
    build(SOURCE, EVENTS, ROOT / 'configs/asap_valid_heldout_12works.json', 512, .5,
          sidecar_overrides=overrides, normalization_description=METHOD)
    fixed_audit = prepare(EVENTS, FIXED, output_tag='e512_v1')
    (OUTPUT / 'fixed_event_data_audit.json').write_text(json.dumps(fixed_audit, indent=2) + '\n')
    checked = audit(FIXED, 'e512_v1')
    for split in ('train', 'valid', 'test'):
        current = torch.load(FIXED / f'dinr_beat_{split}_e512_v1.pt', weights_only=False)
        old = torch.load(PROJECT / f'data/ASAP_processed_dinr_beat_dev_e512_v2/dinr_beat_{split}_e512_v2.pt', weights_only=False)
        assert torch.equal(current['events'], old['events']), 'Score inputs changed'
        assert current['samples'] == old['samples'], 'Window cohort changed'
        assert torch.equal(current['target_valid'], old['target_valid']), 'Validity masks changed'
        assert torch.equal(current['targets'][:, 2], old['targets'][:, 2]), 'Velocity changed'
        beat = current['events'][:, 0] == 0
        residuals = []
        for p in current['performances']:
            a, b = p['span']
            valid = beat[a:b] & current['target_valid'][a:b, 0]
            ratios = torch.log(current['targets'][a:b, 0][valid] / current['events'][a:b, 2][valid])
            residuals.append(float(np.median(ratios.numpy())))
        assert max(map(abs, residuals)) < 1e-5
        checked['splits'][split]['max_abs_median_log_beat_ratio'] = max(map(abs, residuals))
        note = ~beat & current['target_valid'][:, 1]
        abs_onset = current['targets'][note, 1] + current['events'][note, 3]
        checked['splits'][split]['clipping']['note_onset_abs'] = dict(count=len(abs_onset),
            below=float((abs_onset < -.25).float().mean()), above=float((abs_onset > 1).float().mean()))
    distribution_stats = summarize_distributions()
    checked['checks'] = ['positive finite duration', 'exactly-once fixed512 event coverage',
        'identical score inputs, sample cohorts, validity masks and velocity to previous experiment',
        'work-disjoint splits', 'all performances median log beat ratio near zero', '19 test works / 82 performances']
    assert checked['splits']['test']['performances'] == 82
    (OUTPUT / 'fixed_event_data_audit_checked.json').write_text(json.dumps(checked, indent=2) + '\n')
    scales = np.array([p['scale'] for p in records])
    stats = dict(method=METHOD, works=len(sources), performances=len(records),
        scale_percentiles=dict(zip(('min','p25','median','p75','max'), np.percentile(scales,[0,25,50,75,100]).tolist())),
        max_abs_raw_normalized_median_log_ratio=max(abs(p['normalized_median_log_deviation']) for p in records),
        distributions=distribution_stats, splits=checked['splits'])
    (OUTPUT / 'statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
    combined = []
    for split, count in (('test', 15), ('valid', 10)):
        curves = select_works(load_curves(FIXED / f'dinr_beat_{split}_e512_v1.pt'), count)
        for i, item in enumerate(curves, 1):
            item.update(split=split, panel=f'{"T" if split=="test" else "V"}{i:02d}')
        combined.extend(curves)
    plot_matrix(combined, OUTPUT / 'test15_valid10_beat_duration_5x5.png')
    for item in combined:
        for p in item['performances']:
            p['duration_ms'] = [v if np.isfinite(v) else None for v in p['duration_ms']]
    (OUTPUT / 'test15_valid10_curves.json').write_text(json.dumps(dict(method=METHOD, curves=combined), indent=2, allow_nan=False) + '\n')
    print('Preparation passed:', json.dumps({k: v for k, v in stats.items() if k != 'splits'}), flush=True)


if __name__ == '__main__':
    main()
