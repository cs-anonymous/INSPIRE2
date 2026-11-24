"""Build continuous beat/note event sidecars from aligned ASAP sidecars."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from miditoolkit import MidiFile

from src.data_process.dinr_dataset import _sidecar_paths
from src.data_process.beat_mapping import (
    map_score_beats, map_score_beats_structural_earliest, valid_beat_intervals,
)


ROOT = Path(__file__).resolve().parents[2]
MIN_WINDOW_NOTES = 8


def load_beat_mapper():
    path = ROOT.parent / "scoreperformer/scoreperformer/data/midi/beats.py"
    spec = importlib.util.spec_from_file_location("scoreperformer_beats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_bar_beat_ticks, module.get_performance_beats


def score_notes(midi):
    notes = [note for instrument in midi.instruments if not instrument.is_drum
             for note in instrument.notes]
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end, note.velocity))


def canonical_beat_windows(notes_per_beat, block_notes, overlap_ratio):
    if sum(notes_per_beat) < MIN_WINDOW_NOTES:
        return []
    stride = max(1, int(round(block_notes * (1.0 - overlap_ratio))))
    starts = []
    cumulative = np.concatenate(([0], np.cumsum(notes_per_beat)))
    start_beat = 0
    while start_beat < len(notes_per_beat):
        end_beat = start_beat
        while end_beat < len(notes_per_beat) and cumulative[end_beat + 1] - cumulative[start_beat] <= block_notes:
            end_beat += 1
        if end_beat == start_beat:
            raise ValueError("A single beat exceeds the note limit; refusing to split it")
        note_count = int(cumulative[end_beat] - cumulative[start_beat])
        if note_count >= MIN_WINDOW_NOTES:
            starts.append((start_beat, end_beat))
        if end_beat >= len(notes_per_beat):
            break
        next_note = cumulative[start_beat] + stride
        next_start = int(np.searchsorted(cumulative, next_note, side="right") - 1)
        start_beat = max(start_beat + 1, min(next_start, end_beat - 1))
    if starts and starts[-1][1] < len(notes_per_beat):
        tail_start = max(0, len(notes_per_beat) - 1)
        while tail_start > 0 and cumulative[-1] - cumulative[tail_start] < MIN_WINDOW_NOTES:
            tail_start -= 1
        tail = (tail_start, len(notes_per_beat))
        if tail[1] > tail[0] and tail not in starts:
            starts.append(tail)
    return list(dict.fromkeys(starts))


def beat_grid(score_midi, get_bar_beat_ticks):
    bar_ticks, beat_ticks = get_bar_beat_ticks(midi=score_midi)
    bar_ticks = np.asarray(bar_ticks, dtype=np.int64)
    beat_ticks = np.asarray(beat_ticks, dtype=np.int64)
    if len(beat_ticks) < 1:
        return None
    tpb = int(score_midi.ticks_per_beat)
    last_sig = score_midi.time_signature_changes[-1]
    denom_group = {6: 2, 9: 3, 12: 4, 18: 3, 24: 4}.get(last_sig.numerator,
                                                                    last_sig.numerator)
    last_interval = int(tpb * 4 * last_sig.numerator / last_sig.denominator / denom_group)
    edges = np.append(beat_ticks, beat_ticks[-1] + max(1, last_interval))
    tick_to_ms = np.asarray(score_midi.get_tick_to_time_mapping(), dtype=np.float64) * 1000.0
    durations = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right < len(tick_to_ms):
            duration = tick_to_ms[right] - tick_to_ms[left]
        elif index > 0:
            duration = durations[-1]
        else:
            stop = min(left + max(1, last_interval), len(tick_to_ms) - 1)
            duration = tick_to_ms[stop] - tick_to_ms[left]
        durations.append(max(float(duration), 1.0))
    beat_to_index = np.searchsorted(beat_ticks, np.arange(int(max(edges[-1], 0)) + 1), side="right") - 1
    beat_to_index = np.clip(beat_to_index, 0, len(beat_ticks) - 1)
    measure_start = np.isin(beat_ticks, bar_ticks).astype(np.int64)
    return edges, beat_ticks, np.asarray(durations), beat_to_index, measure_start


def performance_beat_anchors(score_ticks, score_beat_edges, perf_onsets,
                             get_performance_beats, ticks_per_beat):
    unique_ticks, inverse = np.unique(score_ticks, return_inverse=True)
    group_times = (np.bincount(inverse, weights=perf_onsets)
                   / np.bincount(inverse))
    pairs = np.column_stack((unique_ticks, group_times / 1000.0))
    _, anchors = get_performance_beats(
        score_beat_edges, pairs, monotonic_times=True, ticks_per_beat=ticks_per_beat
    )
    return np.asarray(anchors, dtype=np.float64) * 1000.0


def make_event_stream(score, performance, score_midi, get_bar_beat_ticks,
                      get_performance_beats, anchor_method="onset_group_mean"):
    notes = score_notes(score_midi)
    score_pitch = np.asarray(score["pitch"], dtype=np.int64)
    score_raw = np.asarray(score["score_raw"], dtype=np.float64)
    raw = np.asarray(performance["label_shared_raw"], dtype=np.float64)
    if len(notes) != len(score_pitch) or not np.array_equal(
            np.asarray([note.pitch for note in notes]), score_pitch):
        raise ValueError("Refined score MIDI does not align with sidecar pitch sequence")
    if len(raw) != len(notes) or len(score_raw) != len(notes):
        raise ValueError("Score/performance note arrays have inconsistent lengths")
    grid = beat_grid(score_midi, get_bar_beat_ticks)
    if grid is None:
        return None
    edges, beat_ticks, score_beat_ms, beat_lookup, measure_start = grid
    score_ticks = np.asarray([note.start for note in notes], dtype=np.int64)
    perf_onsets = np.cumsum(raw[:, 0])
    mapper = (map_score_beats_structural_earliest if anchor_method == "structural_earliest"
              else map_score_beats)
    if anchor_method not in ("onset_group_mean", "structural_earliest"):
        raise ValueError(f"Unknown beat anchor method: {anchor_method}")
    anchors, supported = mapper(edges, score_ticks, perf_onsets)
    valid_intervals, mapped_durations = valid_beat_intervals(anchors, supported)
    # Unsupported leading/trailing beat intervals contain no reliable paired
    # onset anchor. Keep their event topology, but exclude their onset target
    # from loss and use score duration as a neutral duration fallback.
    # Keep score timing immutable: event inputs must be identical across
    # performances of the same work.  A view here would let performance
    # durations overwrite the score beat durations in ``score_beat_ms``.
    perf_beat_ms = np.array(score_beat_ms, dtype=np.float64, copy=True)
    valid_duration = valid_intervals & np.isfinite(mapped_durations)
    perf_beat_ms[valid_duration] = np.maximum(mapped_durations[valid_duration], 1.0)
    if len(perf_beat_ms) != len(beat_ticks):
        raise ValueError("Beat mapper returned wrong number of beat intervals")
    note_beat_ids = np.searchsorted(beat_ticks, score_ticks, side="right") - 1
    note_beat_ids = np.clip(note_beat_ids, 0, len(beat_ticks) - 1)
    score_beat_lengths = np.diff(edges)
    score_note_onsets = (score_ticks - beat_ticks[note_beat_ids]) / score_beat_lengths[note_beat_ids]
    note_onset_dev = np.zeros(len(notes), dtype=np.float64)
    note_valid = valid_intervals[note_beat_ids]
    note_onset_ratio = ((perf_onsets - anchors[note_beat_ids]) /
                        np.maximum(perf_beat_ms[note_beat_ids], 1.0))
    note_onset_dev[note_valid] = note_onset_ratio[note_valid] - score_note_onsets[note_valid]
    score_note_durations = np.maximum(score_raw[:, 1], 1.0)
    perf_note_durations = np.maximum(raw[:, 1], 1.0)
    score_velocity = np.clip(np.rint(score_raw[:, 2]), 0, 127)
    perf_velocity = np.clip(np.rint(raw[:, 2]), 0, 127)

    notes_by_beat = [[] for _ in beat_ticks]
    for note_index, beat_id in enumerate(note_beat_ids.tolist()):
        notes_by_beat[beat_id].append(note_index)
    events = []
    targets = []
    target_valid = []
    beat_event_spans = []
    for beat_id, note_indices in enumerate(notes_by_beat):
        event_start = len(events)
        # [type, pitch, score duration ms, score onset ratio, score velocity, measure_start]
        events.append([0, -1, score_beat_ms[beat_id], -1.0, -1, measure_start[beat_id]])
        # [performance duration ms, performance onset ratio, performance velocity]
        targets.append([perf_beat_ms[beat_id], -1.0, -1])
        target_valid.append([bool(valid_duration[beat_id]), False, False])
        for note_index in note_indices:
            events.append([1, score_pitch[note_index], score_note_durations[note_index],
                           score_note_onsets[note_index], score_velocity[note_index], -1])
            targets.append([perf_note_durations[note_index], note_onset_dev[note_index],
                            perf_velocity[note_index]])
            target_valid.append([True, bool(note_valid[note_index]), True])
        beat_event_spans.append((event_start, len(events), len(note_indices)))
    return (torch.tensor(events, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(target_valid, dtype=torch.bool), beat_event_spans)


def build(root, output_dir, manifest_path, block_notes, overlap_ratio, splits=("train", "valid", "test"),
          sidecar_overrides=None, normalization_description=None,
          anchor_method="onset_group_mean"):
    get_bar_beat_ticks, get_performance_beats = load_beat_mapper()
    heldout = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_paths = {row["work_path"] if isinstance(row, dict) else row
                     for row in heldout["valid_work_paths"]}
    for requested_split in splits:
        all_events, all_targets, all_target_valid, samples, performances = [], [], [], [], []
        offset = 0
        paths = _sidecar_paths(root)
        if requested_split == "test":
            # Match the established 19-work test protocol, including two
            # MS-score works without an ASAP-score grouped sidecar.
            rows = list(csv.DictReader((root / "metadata.csv").open(encoding="utf-8")))
            paths = []
            for source in sorted({r["refined_score_midi_path"] for r in rows
                                  if r.get("split") == "test" and r.get("refined_score_midi_path")}):
                base = root / source
                grouped = base.with_suffix(".ASAP_GROUPED_MEDIAN.pt")
                musical = base.with_suffix(".ASAP_MUSICAL51.pt")
                path = grouped if grouped.is_file() else musical
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(path)
            if len(paths) != 19:
                raise ValueError(f"Expected 19 test works, found {len(paths)}")
        for sidecar in paths:
            relative_work = sidecar.parent.relative_to(root).as_posix()
            if sidecar_overrides is not None:
                sidecar = Path(sidecar_overrides[str(sidecar.resolve())])
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            score = payload.get("score") or {}
            for perf_index, performance in enumerate(payload.get("performances", [])):
                source_split = str(performance.get("split", ""))
                is_heldout = relative_work in heldout_paths
                actual_split = ("valid" if source_split == "train" and is_heldout else
                                "train" if source_split == "train" else source_split)
                if actual_split != requested_split:
                    continue
                score_source = payload.get("meta", {}).get("score_source")
                score_path = root / score_source if score_source else sidecar.with_name("score_ASAP_refined.mid")
                score_midi = MidiFile(str(score_path))
                stream = make_event_stream(score, performance, score_midi,
                                           get_bar_beat_ticks, get_performance_beats,
                                           anchor_method=anchor_method)
                if stream is None:
                    continue
                events, targets, target_valid, beat_spans = stream
                note_counts = [span[2] for span in beat_spans]
                for beat_start, beat_end in canonical_beat_windows(
                        note_counts, block_notes, overlap_ratio):
                    while beat_end > beat_start and note_counts[beat_end - 1] == 0:
                        beat_end -= 1
                    if beat_end <= beat_start or sum(note_counts[beat_start:beat_end]) < MIN_WINDOW_NOTES:
                        continue
                    event_start = beat_spans[beat_start][0]
                    event_end = beat_spans[beat_end - 1][1]
                    samples.append({
                        "span": [offset + event_start, offset + event_end],
                        "work": relative_work,
                        "performance": performance.get("performance_id", str(perf_index)),
                        "notes": int(sum(note_counts[beat_start:beat_end])),
                        "beats": int(beat_end - beat_start),
                    })
                performances.append({
                    "span": [offset, offset + len(events)], "work": relative_work,
                    "performance": performance.get("performance_id", str(perf_index)),
                    "source": performance.get("performance_source"),
                    "global_timing_scale": performance.get("global_timing_scale"),
                    "source_normalization": payload.get("performance_time_normalization"),
                    "source_sidecar": str(sidecar.resolve()),
                })
                all_events.append(events)
                all_targets.append(targets)
                all_target_valid.append(target_valid)
                offset += len(events)
            del payload
        if not samples:
            if requested_split == "valid":
                raise ValueError("No validation samples found; check held-out manifest paths")
            if requested_split == "test":
                raise ValueError("No test performances found")
        result = {
            "format": "dinr_beat_events_v2",
            "split": requested_split,
            "sidecar_root": str(root.resolve()),
            "valid_work_manifest": str(manifest_path.resolve()),
            "window_notes": block_notes,
            "overlap_ratio": overlap_ratio,
            "event_fields": ["type", "pitch", "score_duration_ms", "score_onset_ratio",
                             "score_velocity", "measure_start"],
            "target_fields": ["performance_duration_ms", "performance_onset_dev",
                              "performance_velocity"],
            "beat_anchor_protocol": (
                "earliest performance onset in each score onset group; "
                "interpolate score beat edges between structural onset-group anchors; "
                "no endpoint extrapolation; unsupported endpoints masked"
                if anchor_method == "structural_earliest" else
                "score-onset-group mean + linear interpolation; no monotonic filtering or endpoint extrapolation; unsupported endpoint onset labels are masked"),
            "onset_target_protocol": "performance beat-relative onset ratio minus score beat-relative onset ratio",
            "events": torch.cat(all_events) if all_events else torch.empty((0, 6)),
            "targets": torch.cat(all_targets) if all_targets else torch.empty((0, 3)),
            "target_valid": torch.cat(all_target_valid) if all_target_valid else torch.empty((0, 3), dtype=torch.bool),
            "samples": samples,
            "performances": performances,
            "beat_mapper_time_unit": "seconds in mapper; milliseconds in sidecar",
            "performance_time_normalization": normalization_description or "existing grouped-median labels; two raw MS-score test fallbacks as in exp1112",
        }
        output = output_dir / f"dinr_beat_{requested_split}_b{block_notes}_o{int(overlap_ratio * 100)}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        print(f"wrote {output}: events={len(result['events']):,}, windows={len(samples):,}, "
              f"notes={sum(row['notes'] for row in samples):,}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, default=Path("../data/ASAP_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("../data/ASAP_processed_dinr_beat_v1"))
    parser.add_argument("--valid-work-manifest", type=Path,
                        default=Path("configs/asap_valid_heldout_12works.json"))
    parser.add_argument("--block-notes", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid", "test"])
    args = parser.parse_args()
    build(args.sidecar_root, args.output_dir, args.valid_work_manifest,
          args.block_notes, args.overlap_ratio, args.splits)


if __name__ == "__main__":
    main()
