#!/usr/bin/env python3
"""Evaluate score-aligned timing, duration, and velocity metrics.

The legacy AR HR evaluator sorts generated MIDI notes by their realized onset.
This diagnostic keeps score-event identity by matching notes by pitch and
within-pitch score/performance order, then evaluates four fixed metrics:

* beat-level tempo-curve Pearson correlation;
* per-score-event onset-deviation W1;
* per-score-event duration log-ratio W1;
* per-score-event velocity W1.

The same evaluator can be applied to the note-only 1112 outputs and the
beat-aware outputs once their prediction manifests are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile
from scipy.stats import pearsonr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for path in (ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data_process.build_dinr_beat_sidecars import (  # noqa: E402
    beat_grid,
    load_beat_mapper,
    score_notes,
)
from src.data_process.beat_mapping import map_score_beats, valid_beat_intervals  # noqa: E402
from src.evaluate.evaluate_dinr_absolute import test_sidecars  # noqa: E402


DURATION_DEV_EDGES = np.linspace(-3.0, 2.0, 501)
VELOCITY_EDGES = np.arange(-0.5, 128.5, 1.0)
ONSET_DEV_EDGES = np.concatenate(([-np.inf], np.linspace(-1.0, 1.0, 201), [np.inf]))


def tick_ms(mapping: list[float], tick: int) -> float:
    if tick < len(mapping):
        return float(mapping[tick] * 1000.0)
    return float(mapping[-1] * 1000.0)


def midi_events(path: Path | MidiFile) -> list[dict]:
    midi = path if isinstance(path, MidiFile) else MidiFile(str(path))
    mapping = midi.get_tick_to_time_mapping()
    events = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            events.append({
                "pitch": int(note.pitch),
                "start": tick_ms(mapping, int(note.start)),
                "end": tick_ms(mapping, int(note.end)),
                "duration": max(tick_ms(mapping, int(note.end)) - tick_ms(mapping, int(note.start)), 1.0),
                "velocity": float(note.velocity),
            })
    return sorted(events, key=lambda x: (x["start"], x["pitch"], x["end"], x["velocity"]))


def score_midi_for_sidecar(sidecar: Path) -> Path:
    name = sidecar.name
    for suffix in (".ASAP_GROUPED_MEDIAN.pt", ".ASAP_MUSICAL51.pt"):
        if name.endswith(suffix):
            return sidecar.with_name(name[: -len(suffix)] + ".mid")
    raise ValueError(f"Unsupported sidecar name: {sidecar}")


def score_data(sidecar: Path, get_bar_beat_ticks) -> dict:
    midi_path = score_midi_for_sidecar(sidecar)
    midi = MidiFile(str(midi_path))
    notes = score_notes(midi)
    ticks = np.asarray([note.start for note in notes], dtype=np.int64)
    score_events = midi_events(midi)
    if len(notes) != len(score_events):
        raise ValueError(f"Score note count mismatch for {midi_path}")
    # score_notes and midi_events use the same deterministic key.
    score_starts = np.asarray([item["start"] for item in score_events], dtype=np.float64)
    score_durations = np.asarray([item["duration"] for item in score_events], dtype=np.float64)
    score_pitches = np.asarray([item["pitch"] for item in score_events], dtype=np.int64)
    grid = beat_grid(midi, get_bar_beat_ticks)
    if grid is None:
        raise ValueError(f"No beat grid for {midi_path}")
    return {
        "path": midi_path,
        "midi": midi,
        "ticks": ticks,
        "pitches": score_pitches,
        "starts": score_starts,
        "durations": score_durations,
        "grid": grid,
    }


def align_to_score(score: dict, performance_path: Path) -> dict:
    """Match performance events to score events without global onset sorting.

    MIDI export may reorder simultaneous notes. Matching by pitch and rank
    within that pitch preserves score-event identity while remaining
    deterministic for repeated pitches.
    """
    perf = midi_events(performance_path)
    score_by_pitch: dict[int, list[int]] = defaultdict(list)
    perf_by_pitch: dict[int, list[dict]] = defaultdict(list)
    for index, pitch in enumerate(score["pitches"]):
        score_by_pitch[int(pitch)].append(index)
    for event in perf:
        perf_by_pitch[event["pitch"]].append(event)
    starts = np.full(len(score["pitches"]), np.nan, dtype=np.float64)
    durations = np.full(len(score["pitches"]), np.nan, dtype=np.float64)
    velocities = np.full(len(score["pitches"]), np.nan, dtype=np.float64)
    missing = []
    for pitch, score_indices in score_by_pitch.items():
        perf_events = perf_by_pitch.get(pitch, [])
        if len(perf_events) != len(score_indices):
            missing.append((pitch, len(score_indices), len(perf_events)))
            continue
        for score_index, event in zip(score_indices, perf_events):
            starts[score_index] = event["start"]
            durations[score_index] = event["duration"]
            velocities[score_index] = event["velocity"]
    if missing:
        raise ValueError(f"Cannot score-align {performance_path}: pitch counts {missing[:5]}")
    return {"starts": starts, "durations": durations, "velocities": velocities}


def beat_anchors(score: dict, aligned: dict, get_bar_beat_ticks, get_performance_beats) -> tuple[np.ndarray, np.ndarray]:
    grid = score["grid"]
    edges, beat_ticks, _, _, _ = grid
    anchors, supported = map_score_beats(edges, score["ticks"], aligned["starts"])
    valid, durations = valid_beat_intervals(anchors, supported)
    return np.asarray(beat_ticks), anchors, valid, durations


def tempo_curve(score: dict, aligned: dict, get_bar_beat_ticks, get_performance_beats) -> tuple[np.ndarray, np.ndarray]:
    beat_ticks, _, valid, periods = beat_anchors(
        score, aligned, get_bar_beat_ticks, get_performance_beats
    )
    keep = valid & np.isfinite(periods) & (periods > 0)
    if np.sum(keep) < 3:
        return np.empty(0), np.empty(0)
    # Multiplicative tempo changes are additive in log space. The constant
    # 60 seconds/minute factor is irrelevant to correlation.
    return beat_ticks[keep], -np.log(periods[keep])


def velocity_curve(score: dict, aligned: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean score-event-aligned velocity per score beat."""
    beat_ticks = np.asarray(score["grid"][1], dtype=np.int64)
    beat_ids = np.searchsorted(beat_ticks, score["ticks"], side="right") - 1
    valid = (beat_ids >= 0) & (beat_ids < len(beat_ticks)) & np.isfinite(aligned["velocities"])
    beat_ids = beat_ids[valid]
    if not len(beat_ids):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    sums = np.bincount(beat_ids, weights=aligned["velocities"][valid], minlength=len(beat_ticks))
    counts = np.bincount(beat_ids, minlength=len(beat_ticks))
    occupied = counts > 0
    return beat_ticks[occupied], sums[occupied] / counts[occupied]


def tempo_pearson(pred: tuple[np.ndarray, np.ndarray], human: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not human or len(pred[1]) < 3:
        return float("nan")
    common = set(pred[0].tolist())
    for beats, _ in human:
        common &= set(beats.tolist())
    common = sorted(common)
    if len(common) < 3:
        return float("nan")

    def indexed(curve):
        return {int(beat): value for beat, value in zip(curve[0], curve[1])}

    p = indexed(pred)
    h = np.asarray([[indexed(curve)[beat] for beat in common] for curve in human], dtype=np.float64)
    reference = np.nanmedian(h, axis=0)
    values = np.asarray([p[beat] for beat in common], dtype=np.float64)
    if np.std(values) < 1e-12 or np.std(reference) < 1e-12:
        return float("nan")
    return float(pearsonr(values, reference).statistic)


def onset_dev(score: dict, aligned: dict, get_bar_beat_ticks, get_performance_beats) -> dict:
    grid = score["grid"]
    edges, beat_ticks, _, _, _ = grid
    _, anchors, valid_intervals, periods = beat_anchors(
        score, aligned, get_bar_beat_ticks, get_performance_beats
    )
    beat_ids = np.searchsorted(beat_ticks, score["ticks"], side="right") - 1
    valid = (
        np.isfinite(aligned["starts"])
        & (beat_ids >= 0)
        & (beat_ids < len(periods))
        & valid_intervals[np.clip(beat_ids, 0, len(periods) - 1)]
    )
    beat_ids = beat_ids[valid]
    score_ratio = (
        score["ticks"][valid] - beat_ticks[beat_ids]
    ) / np.diff(edges)[beat_ids]
    perf_ratio = (aligned["starts"][valid] - anchors[beat_ids]) / periods[beat_ids]
    dev = perf_ratio - score_ratio
    return {
        "mean_abs": float(np.mean(np.abs(dev))) if len(dev) else float("nan"),
        "signed_mean": float(np.mean(dev)) if len(dev) else float("nan"),
        "p95_abs": float(np.quantile(np.abs(dev), 0.95)) if len(dev) else float("nan"),
        "count": int(len(dev)),
        "values": dev,
        "indices": np.flatnonzero(valid),
        "ratio_values": perf_ratio,
        "coverage": float(len(dev) / len(score["ticks"])) if len(score["ticks"]) else float("nan"),
    }


def clipped_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.zeros(len(edges) - 1, dtype=np.float64)
    values = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64) / len(values)


def js_divergence(pred: np.ndarray, human: np.ndarray, edges: np.ndarray) -> float:
    p = clipped_hist(pred, edges)
    q = clipped_hist(human, edges)
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    p_term = np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
    q_term = np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    return float(0.5 * p_term + 0.5 * q_term)


def distribution_pair(pred: np.ndarray, human: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    pred = pred[np.isfinite(pred)]
    human = human[np.isfinite(human)]
    if not len(pred) or not len(human):
        return float("nan"), float("nan")
    # W1 is computed on the raw continuous coordinate. Histogram support and
    # binning affect JS only; clipping W1 would silently make it range-dependent.
    return float(wasserstein_distance(pred, human)), float(js_divergence(pred, human, edges))


def pn_wasserstein(pred: np.ndarray, humans: list[np.ndarray], pred_indices=None,
                   human_indices=None) -> float:
    """Per-score-event PN Wasserstein, with one prediction and human draws per note.

    For a single predicted value, the one-dimensional Wasserstein distance to
    the human values is their mean absolute difference.  Unlike a global
    histogram W1, this preserves score-event identity.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if pred_indices is None:
        pred_indices = np.arange(len(pred), dtype=np.int64)
    pred_map = {int(i): float(v) for i, v in zip(pred_indices, pred) if np.isfinite(v)}
    human_maps = []
    for values, indices in zip(humans, human_indices or [None] * len(humans)):
        values = np.asarray(values, dtype=np.float64)
        if indices is None:
            indices = np.arange(len(values), dtype=np.int64)
        human_maps.append({int(i): float(v) for i, v in zip(indices, values) if np.isfinite(v)})
    distances = []
    for index, value in pred_map.items():
        reference = [mapping[index] for mapping in human_maps if index in mapping]
        if reference:
            distances.append(float(np.mean(np.abs(value - np.asarray(reference)))))
    return float(np.mean(distances)) if distances else float("nan")


def duration_dev(aligned: dict, score: dict) -> np.ndarray:
    return np.log(np.maximum(aligned["durations"], 1.0)) - np.log(np.maximum(score["durations"], 1.0))


def human_paths(run_summary: Path, work: int) -> list[Path]:
    root = run_summary.parent / "sample_2" / f"score_{work:02d}"
    return sorted(root.glob("gt_*.mid"))


def evaluate_summary(summary_path: Path, sidecars: list[Path], get_bar_beat_ticks, get_performance_beats) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    score_cache = {index: score_data(sidecar, get_bar_beat_ticks)
                   for index, sidecar in enumerate(sidecars)}
    human_cache: dict[tuple[int, str], dict] = {}
    for row in rows:
        work = int(row["work"])
        for path in human_paths(summary_path, work):
            human_cache[(work, str(path))] = align_to_score(score_cache[work], path)

    per_sample = []
    per_work = []
    for sample in sorted({int(row["sample"]) for row in rows}):
        sample_rows = [row for row in rows if int(row["sample"]) == sample]
        metric_rows = []
        for row in sample_rows:
            work = int(row["work"])
            score = score_cache[work]
            pred_path = Path(row["prediction"])
            pred = align_to_score(score, pred_path)
            gt_paths = human_paths(summary_path, work)
            humans = [human_cache[(work, str(path))] for path in gt_paths]
            pred_curve = tempo_curve(score, pred, get_bar_beat_ticks, get_performance_beats)
            human_curves = [tempo_curve(score, item, get_bar_beat_ticks, get_performance_beats) for item in humans]
            pred_velocity_curve = velocity_curve(score, pred)
            human_velocity_curves = [velocity_curve(score, item) for item in humans]
            onset = onset_dev(score, pred, get_bar_beat_ticks, get_performance_beats)
            human_onsets = [onset_dev(score, item, get_bar_beat_ticks, get_performance_beats)
                            for item in humans]
            pred_duration = duration_dev(pred, score)
            human_duration = [duration_dev(item, score) for item in humans]
            pred_velocity = pred["velocities"]
            human_velocity = [item["velocities"] for item in humans]
            onset_w1 = pn_wasserstein(
                onset["values"], [item["values"] for item in human_onsets],
                onset["indices"], [item["indices"] for item in human_onsets])
            duration_w1 = pn_wasserstein(pred_duration, human_duration)
            velocity_w1 = pn_wasserstein(pred_velocity, human_velocity)
            metric_rows.append({
                "work": work,
                "sample": sample,
                "tempo_curve_pearson": tempo_pearson(pred_curve, human_curves),
                "onset_dev_w1": onset_w1,
                "duration_w1": duration_w1,
                "velocity_w1": velocity_w1,
                "onset_count": onset["count"],
                "onset_coverage": onset["coverage"],
                "human_count": len(humans),
            })
        names = ["tempo_curve_pearson", "onset_dev_w1", "duration_w1", "velocity_w1"]
        aggregate = {name: float(np.nanmean([row[name] for row in metric_rows])) for name in names}
        aggregate.update({"sample": sample, "works": len(metric_rows)})
        per_sample.append(aggregate)
        per_work.extend(metric_rows)

    names = ["tempo_curve_pearson", "onset_dev_w1", "duration_w1", "velocity_w1"]
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in per_sample], dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "samples": values.tolist(),
        }
    return {
        "method": summary_path.parent.name,
        "source_summary": str(summary_path.resolve()),
        "protocol": {
            "works": len(score_cache),
            "samples": len(per_sample),
            "duration_space": "log(performance_duration_ms) - log(score_duration_ms); W1 uses raw log-ratio values, JS clips to [-3,2] with 500 uniform bins",
            "velocity_space": "MIDI velocity 0..127, 128 bins",
            "onset_dev_space": "performance onset ratio within its performance beat minus score onset ratio within its score beat; W1 uses raw beat-ratio units, JS uses 0.01-width bins on [-1,1] plus underflow/overflow bins",
            "beat_mapper": "mean performance onset per exact score-onset group, piecewise-linear interpolation only between observed score-onset groups; no 600-BPM filter, no monotonic filtering, no endpoint extrapolation; non-positive mapped beat intervals are unsupported",
            "aggregation": "human references averaged within work, then works macro-averaged; sample mean/std across three AR draws",
            "alignment": "score-event pitch and within-pitch order; no global realized-onset/pitch sorting",
            "tempo": "Pearson correlation of beat-level log tempo against the per-work median human curve",
            "onset_dev": "score-event-aligned performance onset ratio minus score onset ratio, each performance uses the same deterministic score-to-performance beat mapper",
            "onset_coverage": "share of score events whose full containing beat interval has supported, strictly increasing mapped anchors",
        },
        "metrics": summary,
        "per_sample": per_sample,
        "per_work": per_work,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, nargs="+", required=True,
                        help="exp1112_sampling3/*/all_sample_metrics.json")
    parser.add_argument("--sidecar-root", type=Path, default=PROJECT_ROOT / "data/ASAP_processed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecars = test_sidecars(args.sidecar_root)
    get_bar_beat_ticks, get_performance_beats = load_beat_mapper()
    results = [evaluate_summary(path, sidecars, get_bar_beat_ticks, get_performance_beats)
               for path in args.summary]
    output = {"results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(result["method"])
        for name, values in result["metrics"].items():
            print(f"  {name}: {values['mean']:.6f} ± {values['std']:.6f}")


if __name__ == "__main__":
    main()
