#!/usr/bin/env python3
"""Compare saved exp1112/PT/BEAT outputs using four score-aligned metrics.

All methods use the frozen exp1116 continuous human targets. Beat timing and
note onset deviations are re-extracted from score-aligned predicted note
onsets with the current onset-group-mean mapper. This is a separate common
observation protocol, not the native BEAT-token evaluation.

PT MIDI files can contain complete note-on streams but unpaired/zero-length
notes. Preserve onset/velocity identity using the raw positive note-ons; keep
duration unavailable when standard FIFO MIDI decoding has no positive-length
pair. Never shift a later same-pitch note into an earlier missing note's slot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import mido
from miditoolkit import MidiFile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.data_process.beat_mapping import map_score_beats, valid_beat_intervals
from src.data_process.build_dinr_beat_sidecars import beat_grid, load_beat_mapper, score_notes
from script.plot_exp1112 import table_fields

NAMES = ('tempo_curve_pearson', 'onset_w1', 'duration_w1', 'velocity_w1')
BEGIN = '<!-- EXP1116_BASELINE_COMPARISON_BEGIN -->'
END = '<!-- EXP1116_BASELINE_COMPARISON_END -->'
_REFERENCES = None


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ticks_to_ms(ticks, tempo_events, ticks_per_beat):
    """Integrate the tempo map sparsely, avoiding a max_tick-sized array."""
    tempos = {0: 500000}
    for tick, microseconds in sorted(tempo_events, key=lambda pair: pair[0]):
        tempos[int(tick)] = int(microseconds)
    boundaries = np.array(sorted(tempos), dtype=np.float64)
    rates = np.array([tempos[int(t)] for t in boundaries], dtype=np.float64)
    rates /= float(ticks_per_beat) * 1000.0
    starts = np.r_[0.0, np.cumsum(np.diff(boundaries) * rates[:-1])]
    ticks = np.asarray(ticks, dtype=np.float64)
    indices = np.searchsorted(boundaries, ticks, side='right') - 1
    return starts[indices] + (ticks - boundaries[indices]) * rates[indices]


def read_note_on_stream(path):
    """Retain every positive note-on and decode ends using standard FIFO."""
    raw = Path(path).read_bytes()
    midi = mido.MidiFile(file=BytesIO(raw))
    notes, tempos = [], []
    orphan_offs = zero_length = dangling = 0
    for track in midi.tracks:
        tick = 0
        opened = defaultdict(deque)
        for message in track:
            tick += message.time
            if message.type == 'set_tempo':
                tempos.append((tick, message.tempo))
            if message.type not in ('note_on', 'note_off') or message.channel == 9:
                continue
            key = (message.channel, message.note)
            if message.type == 'note_on' and message.velocity > 0:
                opened[key].append(len(notes))
                notes.append([message.note, tick, np.nan, message.velocity])
            elif opened[key]:
                index = opened[key].popleft()
                if tick > notes[index][1]:
                    notes[index][2] = tick
                else:
                    zero_length += 1
            else:
                orphan_offs += 1
        dangling += sum(len(indices) for indices in opened.values())
    if not notes:
        raise ValueError(f'No positive note-ons: {path}')
    notes = np.asarray(notes, dtype=np.float64)
    start_ms = ticks_to_ms(notes[:, 1], tempos, midi.ticks_per_beat)
    paired = np.isfinite(notes[:, 2])
    duration = np.full(len(notes), np.nan)
    duration[paired] = (ticks_to_ms(notes[paired, 2], tempos, midi.ticks_per_beat)
                        - start_ms[paired])
    return {
        'pitch': notes[:, 0].astype(np.int64), 'start_tick': notes[:, 1],
        'start': start_ms, 'duration': duration, 'velocity': notes[:, 3],
        'audit': {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'note_ons': len(notes), 'paired_positive_durations': int(paired.sum()),
                  'zero_length_pairs': zero_length, 'dangling_note_ons': dangling,
                  'orphan_note_offs': orphan_offs},
    }


def align_note_ons(stream, score_pitches):
    """Match full note-on pitch streams by within-pitch onset rank."""
    score_pitches = np.asarray(score_pitches)
    if set(stream['pitch']) != set(score_pitches):
        raise ValueError('Prediction and score pitch sets differ')
    permutation = np.empty(len(score_pitches), dtype=np.int64)
    ambiguous = np.zeros(len(score_pitches), dtype=bool)
    for pitch in np.unique(score_pitches):
        target = np.flatnonzero(score_pitches == pitch)
        source = np.flatnonzero(stream['pitch'] == pitch)
        if len(source) != len(target):
            raise ValueError(f'Pitch {pitch}: {len(source)} note-ons vs {len(target)} score notes')
        source = source[np.argsort(stream['start_tick'][source], kind='stable')]
        permutation[target] = source
        _, inverse, counts = np.unique(stream['start_tick'][source], return_inverse=True, return_counts=True)
        ambiguous[target] = counts[inverse] > 1
    result = {name: stream[name][permutation].copy() for name in ('start', 'duration', 'velocity')}
    # MIDI sorts simultaneous repeated-pitch note-ons by note end, so their
    # original score identities cannot be recovered. Onsets are identical and
    # remain usable; do not invent per-note velocity/duration assignments.
    for name in ('duration', 'velocity'):
        result[name][ambiguous] = np.nan
    stream['audit']['ambiguous_repeated_pitch_note_ons'] = int(ambiguous.sum())
    return result


def load_references(path):
    torch.set_num_threads(1)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    grouped = defaultdict(list)
    for row in payload['performances']:
        grouped[row['work']].append(row)
    get_bar, _ = load_beat_mapper()
    references = []
    for work, rows in grouped.items():
        a, b = rows[0]['span']
        events = payload['events'][a:b].numpy().astype(np.float64)
        note = events[:, 0] == 1
        beat = ~note
        sidecar = Path(rows[0]['source_sidecar'])
        score_path = sidecar.with_name(sidecar.name.split('.ASAP_')[0] + '.mid')
        midi = MidiFile(str(score_path))
        notes = score_notes(midi)
        ticks = np.array([n.start for n in notes], dtype=np.int64)
        pitches = np.array([n.pitch for n in notes], dtype=np.int64)
        if not np.array_equal(pitches, events[note, 1]):
            raise ValueError(f'Score identity mismatch: {work}')
        grid = beat_grid(midi, get_bar)
        edges, beat_ticks, score_beat_ms, _, _ = grid
        if len(score_beat_ms) != int(beat.sum()):
            raise ValueError(f'Frozen beat topology mismatch: {work}')
        membership = np.searchsorted(beat_ticks, ticks, side='right') - 1
        ratios = (ticks - beat_ticks[membership]) / np.diff(edges)[membership]
        if not np.allclose(ratios, events[note, 3], atol=1e-6):
            raise ValueError(f'Frozen note positions mismatch: {work}')
        targets, masks = [], []
        for row in rows:
            a, b = row['span']
            if not np.array_equal(payload['events'][a:b].numpy(), events):
                raise ValueError(f'Performance score topology mismatch: {work}')
            targets.append(payload['targets'][a:b].numpy().astype(np.float64))
            masks.append(payload['target_valid'][a:b].numpy().astype(bool))
        targets, masks = np.stack(targets), np.stack(masks)
        score_log_duration = np.log(np.maximum(events[note, 2], 1.0))
        references.append(dict(
            work=work, sidecar=str(sidecar), score_path=str(score_path),
            score_source=str(score_path.relative_to(PROJECT / 'data/ASAP_processed')),
            notes=len(ticks), humans=len(rows), ticks=ticks, pitches=pitches,
            edges=edges, beat_ticks=beat_ticks, membership=membership, score_ratio=ratios,
            events=events, note_mask=note, beat_mask=beat, score_log_duration=score_log_duration,
            human_onset=targets[:, note, 1],
            human_duration=np.log(np.maximum(targets[:, note, 0], 1.0)) - score_log_duration,
            human_velocity=targets[:, note, 2],
            human_beat_log=np.log(np.maximum(targets[:, beat, 0], 1.0)),
            onset_valid=np.all(masks[:, note, 1], axis=0),
            duration_valid=np.all(masks[:, note, 0], axis=0),
            velocity_valid=np.all(masks[:, note, 2], axis=0),
            beat_valid=np.all(masks[:, beat, 0], axis=0),
        ))
    references.sort(key=lambda row: (-row['notes'], row['sidecar']))
    if len(references) != 19 or sum(row['humans'] for row in references) != 82:
        raise ValueError('Expected 19 works and 82 frozen human references')
    return references


def note_features(notes, reference, mapper=map_score_beats):
    anchors, supported = mapper(reference['edges'], reference['ticks'], notes['start'])
    valid, durations = valid_beat_intervals(anchors, supported)
    beat_log = np.full(len(durations), np.nan)
    beat_log[valid] = np.log(np.maximum(durations[valid], 1.0))
    membership = reference['membership']
    onset = np.full(reference['notes'], np.nan)
    ok = valid[membership]
    onset[ok] = ((notes['start'][ok] - anchors[membership[ok]]) / durations[membership[ok]]
                 - reference['score_ratio'][ok])
    duration = np.log(np.maximum(notes['duration'], 1.0)) - reference['score_log_duration']
    return dict(beat_log=beat_log, onset=onset, duration=duration, velocity=notes['velocity'])


def beat_notes(npz_path, run, reference):
    with np.load(npz_path) as data:
        events = data['events'].astype(np.float64)
        if not np.array_equal(events, reference['events']):
            raise ValueError(f'BEAT NPZ score identity mismatch: {npz_path}')
        tokens = data['predictions']
        if tokens.ndim == 3:
            if tokens.shape[0] != 1:
                raise ValueError('Expected one AR history')
            tokens = tokens[0]
    note, beat = reference['note_mask'], reference['beat_mask']
    logs = np.empty(len(events), dtype=np.float64)
    for mask, field in ((beat, 'beat_duration_target'), (note, 'note_duration_target')):
        mode = run.get(field, run['duration_target'])
        low = 3.0 if mode == 'absolute_log' else -3.0
        logs[mask] = low + (tokens[mask, 0] + 0.5) * 0.01
        if mode == 'log_dev':
            logs[mask] += np.log(np.maximum(events[mask, 2], 1.0))
    periods = np.exp(logs[beat])
    anchors = np.r_[0.0, np.cumsum(periods)[:-1]]
    if run.get('onset_target', 'dev') == 'dev':
        position = -1.0 + (tokens[note, 1] + 0.5) * 0.01
        position += events[note, 3]
    else:
        position = -.25 + (tokens[note, 1] + .5) * .0025
    starts = anchors[reference['membership']] + position * periods[reference['membership']]
    return dict(start=starts, duration=np.exp(logs[note]), velocity=tokens[note, 2].astype(float))


def discover_methods():
    campaign = read_json(PROJECT / 'results/exp1112_dinr_abs5000/manifest.json')
    sampling = read_json(PROJECT / 'results/exp1112_sampling3/manifest.json')
    methods = []
    for job in sampling['jobs']:
        if job['kind'] == 'pt':
            methods.append(dict(id=job['id'], family='PT', label='PT-1' if job['id']=='pt_base' else 'PT-scratch',
                                mask=job['mask'], representation=job['time'],
                                source=str(Path(job['run_dir']) / 'all_sample_metrics.json')))
    for job in campaign['jobs']:
        method, mask, representation = table_fields(job)
        source = PROJECT / 'results/exp1112_sampling3' / job['id'] / 'all_sample_metrics.json'
        if not source.exists():
            source = Path(job['run_dir']) / 'all_sample_metrics.json'
        if not source.is_file():
            raise FileNotFoundError(source)
        methods.append(dict(id=job['id'], family='1112', label=method, mask=mask,
                            representation=representation, source=str(source)))
    for setting in ('A1', 'A2', 'A3', 'B1', 'B2', 'D4'):
        directory = PROJECT / 'results/exp1117_beat_dev_bs48' / f'{setting}_seed42'
        run = read_json(directory / 'run_config.json')
        methods.append(dict(id='beat_'+setting, family='BEAT', label='BEAT '+setting,
                            mask='timestep', representation=run['duration_target'], source=str(directory)))
    if len(methods) != 32:
        raise ValueError(f'Expected 2 PT + 24 DINR + 6 BEAT settings, got {len(methods)}')
    return methods


def init_worker(reference_path):
    global _REFERENCES
    _REFERENCES = load_references(reference_path)


def extract_method(method):
    references = _REFERENCES
    result, sources = {}, []
    if method['family'] == 'BEAT':
        directory = Path(method['source'])
        run = read_json(directory / 'run_config.json')
        by_work = {reference['work']: i for i, reference in enumerate(references)}
        rows = []
        for work_dir in sorted((directory / 'test').glob('work_*')):
            work = read_json(work_dir / 'metrics.json')['work']
            for sample in range(3):
                rows.append(dict(work=by_work[work], sample=sample,
                                 prediction=str(work_dir / f'autoregressive_{sample:02d}_tokens.npz')))
    else:
        rows = read_json(method['source'])['rows']
    expected = {(work, sample) for work in range(19) for sample in range(3)}
    if len(rows) != 57 or {(int(r['work']), int(r['sample'])) for r in rows} != expected:
        raise ValueError(f"Incomplete/duplicate generations for {method['id']}")
    for row in rows:
        reference = references[int(row['work'])]
        if row.get('score_source', reference['score_source']) != reference['score_source']:
            raise ValueError(f"Work ordering mismatch for {method['id']}: {row['work']}")
        path = Path(row['prediction'])
        if method['family'] == 'BEAT':
            notes = beat_notes(path, run, reference)
            audit = dict(path=str(path), sha256=sha256(path), note_ons=reference['notes'],
                         paired_positive_durations=reference['notes'], zero_length_pairs=0,
                         dangling_note_ons=0, orphan_note_offs=0)
        else:
            stream = read_note_on_stream(path)
            notes = align_note_ons(stream, reference['pitches'])
            audit = stream['audit']
        key = (int(row['work']), int(row['sample']))
        result[key] = note_features(notes, reference)
        sources.append(dict(work=key[0], sample=key[1], **audit))
    return dict(method=method, features=result, sources=sources)


def common_masks(extracted, references):
    masks = []
    for work, reference in enumerate(references):
        valid = dict(beat=reference['beat_valid'].copy(), onset=reference['onset_valid'].copy(),
                     duration=reference['duration_valid'].copy(), velocity=reference['velocity_valid'].copy())
        for result in extracted:
            for sample in range(3):
                values = result['features'][(work, sample)]
                valid['beat'] &= np.isfinite(values['beat_log'])
                for name in ('onset', 'duration', 'velocity'):
                    valid[name] &= np.isfinite(values[name])
        if valid['beat'].sum() < 3 or any(valid[k].sum() == 0 for k in ('onset','duration','velocity')):
            raise ValueError(f'Insufficient common support: {reference["work"]}')
        masks.append(valid)
    return masks


def score_features(prediction, reference, valid):
    beat = valid['beat']
    p = -prediction['beat_log'][beat]
    h = -np.median(reference['human_beat_log'][:, beat], axis=0)
    if p.std() <= 1e-12 or h.std() <= 1e-12:
        raise ValueError('Undefined Pearson on the shared beat mask')
    values = {'tempo_curve_pearson': float(np.corrcoef(p, h)[0, 1])}
    for name in ('onset', 'duration', 'velocity'):
        mask = valid[name]
        values[name+'_w1'] = float(np.abs(prediction[name][None, mask]
                                         - reference['human_'+name][:, mask]).mean())
    return values


def summarize(extracted, references, masks):
    output = []
    for result in extracted:
        per_work, rounds = [], []
        for sample in range(3):
            rows = []
            for work, reference in enumerate(references):
                values = score_features(result['features'][(work,sample)], reference, masks[work])
                rows.append(dict(work=work, work_path=reference['work'], sample=sample, **values))
            rounds.append(dict(sample=sample, works=len(rows),
                               **{name: float(np.mean([r[name] for r in rows])) for name in NAMES}))
            per_work.extend(rows)
        metrics = {name: dict(mean=float(np.mean([r[name] for r in rounds])),
                              std=float(np.std([r[name] for r in rounds], ddof=1)),
                              sampling_values=[r[name] for r in rounds]) for name in NAMES}
        output.append(dict(**result['method'], metrics=metrics, per_sample=rounds,
                           per_work=per_work, sources=result['sources']))
    return output


def write_report(data, output, document):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    import csv
    fields = ['id','family','label','mask','representation'] + [n+s for n in NAMES for s in ('_mean','_std')]
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data['results']:
            writer.writerow({**{key:row[key] for key in fields[:5]},
                             **{name+'_'+stat:row['metrics'][name][stat] for name in NAMES for stat in ('mean','std')}})
    coverage = data['coverage']
    def count(key): return sum(work[key] for work in coverage)
    lines = [BEGIN, '### 4.1 exp1112、PT 与 BEAT 的统一观测指标对照', '',
             '复用已保存的最佳 checkpoint 预测：exp1112 的 24 个 DINR 设置、PT-1（即 PT base-1）、PT-scratch，以及本轮 6 个 BEAT 设置。每项均为 19 首作品 × 3 轮 AR sampling；不重训、不重新采样、不增加全曲或分段时间缩放。', '',
             '**本节采用统一的音符起音重提取协议，与第 4 节上表的原生 BEAT-token 协议分开报告。** 所有模型都用同一个 onset-group 均值＋线性插值映射，从预测音符起音重新估计 beat duration 与 note onset dev。旧模型读取已保存的 MIDI；BEAT 模型由原始 token NPZ 重建连续音符时间，保留 score 事件身份，避免按实际起音排序重新配对。', '',
             '人类参考固定为当前冻结 sidecar 中的 19 首作品、82 个连续演奏标签；不使用各旧方法原先导出的不同人类 MIDI。Tempo 为负 log beat duration 对人类逐拍 log-duration 中位数曲线的 Pearson；三项 W1 均逐 score note 比较单个预测与该音符的人类参考分布，再对音符、作品平均。Duration 使用 ln(D_perf/D_score)，onset 使用拍内 onset dev，velocity 使用原始 MIDI velocity；duration 的 1 ms 下限与主表一致，不按 token 范围裁剪数值。', '',
             f"为保证本节 32 个设置及三轮采样的覆盖完全相同，各指标使用所有预测与人类参考的共同有效位置：tempo {count('shared_beats'):,}/{count('beats'):,} 个 beat；onset {count('shared_onsets'):,}/{count('notes'):,} 个 note；duration {count('shared_durations'):,}/{count('notes'):,} 个 note；velocity {count('shared_velocities'):,}/{count('notes'):,} 个 note。该共同范围仅用于本节，不改变上表主结果。", '',
             'PT-scratch 的原始 MIDI 保留完整的正 velocity note-on 序列，但部分 note-on 没有常规 FIFO 解码可得到的正时长结束事件。对齐使用完整 note-on 的 pitch 与同 pitch 内起音次序，起音与力度不因缺失 duration 而错位；duration 不可用的位置在本节所有模型中统一排除。JSON 保留每份预测的配对与覆盖审计。', '',
             '若 MIDI 中同一 pitch 的多个 note-on 落在同一时刻，导出时的排序无法恢复它们各自的 score 身份：相同起音仍可用于 beat/onset，但这些音符的 duration/velocity 在所有模型中共同排除，不按结束时间或力度任意配对。', '',
             '以下 mean ± sample std（ddof=1）仅反映同一 checkpoint 的三轮采样变化，全部保留三位小数。', '']
    lines += ['下表合并 PT、exp1112 与 BEAT：DINR 和 BEAT 仅展示 log dev 表示，PT 保留原始表示作为参考。共同有效范围仍按全部 32 个设置确定；完整设置见 JSON/CSV。', '',
              '| 设置 | 训练 mask | 表示 | Tempo curve Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |',
              '|---|---|---|---:|---:|---:|---:|']
    for family in ('PT', '1112', 'BEAT'):
        for row in data['results']:
            if row['family'] != family: continue
            if family != 'PT' and row['representation'].replace('_', ' ') != 'log dev': continue
            values = [f"{row['metrics'][name]['mean']:.3f} ± {row['metrics'][name]['std']:.3f}" for name in NAMES]
            label = ('1112 ' if family == '1112' else '') + row['label']
            representation = row['representation'].replace('_', ' ')
            lines.append('| '+' | '.join([label,row['mask'],representation]+values)+' |')
    lines.append('')
    relative = output.relative_to(ROOT).as_posix()
    lines += [f'[完整定义、逐作品/逐轮结果与输入审计](../{relative}/metrics.json) · [汇总 CSV](../{relative}/metrics.csv)', END]
    section = '\n'.join(lines)+'\n'
    (output / 'comparison.md').write_text(section)
    old = document.read_text()
    if BEGIN in old:
        first = old.index(BEGIN)
        last = old.index(END, first)+len(END)
        updated = old[:first]+section.rstrip()+old[last:]
    else:
        marker = '\n## 5. 运行与归档'
        if old.count(marker) != 1:
            raise ValueError('Cannot uniquely locate report insertion boundary')
        updated = old.replace(marker, '\n'+section+'\n'+marker, 1)
    temporary = document.with_suffix('.md.tmp')
    temporary.write_text(updated)
    temporary.replace(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=PROJECT/'data/ASAP_processed_dinr_beat_dev_e512_v2/dinr_beat_test_e512_v2.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/exp1116_beat_dev/baseline_comparison')
    parser.add_argument('--document', type=Path, default=ROOT/'docs/exp1116_BEAT.md')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    references = load_references(args.reference)
    methods = discover_methods()
    extracted = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.reference),)) as pool:
        for result in pool.map(extract_method, methods):
            extracted.append(result)
            print(f"extracted {len(extracted)}/{len(methods)} {result['method']['id']}", flush=True)
    masks = common_masks(extracted, references)
    coverage = [dict(work=i, work_path=r['work'], humans=r['humans'], notes=r['notes'],
                     beats=len(r['beat_ticks']), reference_beats=int(r['beat_valid'].sum()),
                     reference_onsets=int(r['onset_valid'].sum()),
                     **{'shared_'+name:int(mask[key].sum()) for name,key in
                        [('beats','beat'),('onsets','onset'),('durations','duration'),('velocities','velocity')]})
                for i,(r,mask) in enumerate(zip(references,masks))]
    data = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                protocol=dict(reference=str(args.reference.resolve()), reference_sha256=sha256(args.reference),
                    methods=32, works=19, human_performances=82, sampling_draws=3, ddof=1,
                    timing_normalization='frozen existing grouped-median labels, with two existing unscaled MS-score test fallbacks; no new prediction scaling',
                    observation='all methods: score-onset-group mean + piecewise-linear beat mapping of predicted note onsets; no endpoint extrapolation or monotonic filtering',
                    human_targets='frozen unquantized sidecar targets and validity masks',
                    alignment='full positive note-on stream, pitch and within-pitch onset rank; BEAT NPZ retains explicit score event identity',
                    missing_duration='unpaired/nonpositive FIFO MIDI durations unavailable; no imputation or prefix shifting',
                    support='per-metric intersection over frozen human masks and every method/draw; identical support for all rows',
                    tempo='Pearson of negative log beat duration versus per-beat median human negative log duration',
                    onset='mean singleton-to-human W1 at each score note in beat-relative onset-deviation units',
                    duration='mean singleton-to-human W1 at each score note in log(performance/score duration)',
                    velocity='mean singleton-to-human W1 at each score note in MIDI velocity units',
                    aggregation='mean over notes and human draws within work; macro-average 19 works within draw; mean/sample std over three draws'),
                coverage=coverage, results=summarize(extracted,references,masks))
    write_report(data,args.output,args.document)
    for row in data['results']:
        print(row['id'], {n:f"{row['metrics'][n]['mean']:.3f} ± {row['metrics'][n]['std']:.3f}" for n in NAMES},flush=True)


if __name__ == '__main__':
    main()
