#!/usr/bin/env python3
"""Recompute score-event-aligned PN metrics for completed exp1115 beat runs."""

from __future__ import annotations

import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for path in (ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from script.evaluate_six_metrics import (  # noqa: E402
    align_to_score,
    duration_dev,
    midi_events,
    onset_dev,
    score_data,
    tempo_curve,
    tempo_pearson,
)
from src.data_process.build_dinr_beat_sidecars import load_beat_mapper  # noqa: E402
from src.evaluate.evaluate_dinr_absolute import test_sidecars  # noqa: E402


def pn_w1(pred_values: np.ndarray, pred_indices: np.ndarray,
          references: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean per-score-event W1; one prediction vs human draws is mean absolute error."""
    pred_values = np.asarray(pred_values, dtype=np.float64)
    pred_indices = np.asarray(pred_indices, dtype=np.int64)
    max_index = max([int(np.max(pred_indices)) if len(pred_indices) else -1]
                    + [int(np.max(indices)) if len(indices) else -1
                       for _, indices in references])
    matrix = np.full((len(references), max_index + 1), np.nan, dtype=np.float64)
    for row, (values, indices) in enumerate(references):
        matrix[row, np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    selected = matrix[:, pred_indices]
    per_note = np.nanmean(np.abs(selected - pred_values[None, :]), axis=0)
    return float(np.nanmean(per_note))


def aligned_indices(size: int) -> np.ndarray:
    return np.arange(size, dtype=np.int64)


def curve_correlation(pred, humans) -> float:
    return tempo_pearson(pred, humans)


def event_metrics(score, pred, humans, mapper, human_features=None):
    get_bar_beat_ticks, get_performance_beats = mapper
    onset = onset_dev(score, pred, get_bar_beat_ticks, get_performance_beats)
    if human_features is None:
        human_onsets = [onset_dev(score, item, get_bar_beat_ticks, get_performance_beats)
                        for item in humans]
        human_durations = [duration_dev(item, score) for item in humans]
        human_tempos = [tempo_curve(score, item, get_bar_beat_ticks, get_performance_beats)
                        for item in humans]
    else:
        human_onsets, human_durations, human_tempos = human_features
    onset_refs = [(item["values"], item["indices"]) for item in human_onsets]
    onset_w1 = pn_w1(onset["values"], onset["indices"], onset_refs)
    onset_ratio_w1 = pn_w1(onset["ratio_values"], onset["indices"],
                           [(item["ratio_values"], item["indices"]) for item in human_onsets])

    pred_duration = duration_dev(pred, score)
    duration_refs = [(item, aligned_indices(len(score["pitches"]))) for item in human_durations]
    indices = aligned_indices(len(score["pitches"]))
    duration_w1 = pn_w1(pred_duration, indices, duration_refs)
    velocity_refs = [(item["velocities"], indices) for item in humans]
    velocity_w1 = pn_w1(pred["velocities"], indices, velocity_refs)

    pred_tempo = tempo_curve(score, pred, get_bar_beat_ticks, get_performance_beats)
    human_tempo = human_tempos
    return {
        "tempo_curve_pearson": curve_correlation(pred_tempo, human_tempo),
        "onset_w1": onset_w1,
        "onset_ratio_w1_check": onset_ratio_w1,
        "duration_w1": duration_w1,
        "velocity_w1": velocity_w1,
        "onset_coverage": onset["coverage"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only evaluate these run directory names")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/exp1115_beat/pn_metrics.json")
    parser.add_argument("--run-root", type=Path,
                        default=PROJECT_ROOT / "results/exp1115_beat")
    args = parser.parse_args()
    run_root = args.run_root
    sidecar_root = PROJECT_ROOT / "data/ASAP_processed"
    output_path = args.output
    sidecars = test_sidecars(sidecar_root)
    mapper = load_beat_mapper()

    score_by_work = {}
    human_by_work = {}
    aligned_humans_by_work = {}
    for sidecar in sidecars:
        score = score_data(sidecar, mapper[0])
        work = str(score["path"].relative_to(sidecar_root.resolve()).parent)
        score_by_work[work] = score

    runs = []
    skipped = []
    for run_dir in sorted(run_root.glob("*_seed*")):
        if not run_dir.is_dir():
            continue
        if args.only is not None and run_dir.name not in set(args.only):
            continue
        summary_path = run_dir / "test/metrics.json"
        if not summary_path.is_file():
            skipped.append({"run": run_dir.name, "reason": "missing test/metrics.json"})
            continue
        test_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sample_rows = defaultdict(list)
        work_rows = []
        for work_index, work_row in enumerate(test_summary["work_rows"]):
            work = work_row["work"]
            if work not in score_by_work:
                raise KeyError(f"No score sidecar matched test work {work!r}")
            score = score_by_work[work]
            work_dir = summary_path.parent / f"work_{work_index:02d}"
            if work not in human_by_work:
                references = [align_to_score(score, path)
                              for path in sorted(work_dir.glob("reference_*.mid"))]
                if not references:
                    raise FileNotFoundError(f"No reference MIDI under {work_dir}")
                aligned_humans_by_work[work] = references
                human_by_work[work] = (
                    [onset_dev(score, item, mapper[0], mapper[1]) for item in references],
                    [duration_dev(item, score) for item in references],
                    [tempo_curve(score, item, mapper[0], mapper[1]) for item in references],
                )
            references = aligned_humans_by_work[work]
            ref_pitches = np.sort(np.asarray([x["pitch"] for x in midi_events(
                next(work_dir.glob("reference_*.mid")))], dtype=np.int64))
            if not np.array_equal(ref_pitches, np.sort(score["pitches"])):
                raise ValueError(f"Reference/score pitch mismatch for {work}")
            sample_metrics = []
            for sample in range(int(test_summary["samples"])):
                pred_path = work_dir / f"autoregressive_{sample:02d}_history_000.mid"
                pred = align_to_score(score, pred_path)
                metrics = event_metrics(score, pred, references, mapper, human_by_work[work])
                sample_metrics.append(metrics)
                sample_rows[sample].append(metrics)
            work_rows.append({"work": work, "samples": sample_metrics,
                              "reference_count": len(references)})

        sample_summary = {}
        for sample, rows in sorted(sample_rows.items()):
            sample_summary[str(sample)] = {
                key: float(np.nanmean([row[key] for row in rows]))
                for key in ("tempo_curve_pearson", "onset_w1", "onset_ratio_w1_check",
                            "duration_w1", "velocity_w1", "onset_coverage")
            }
        metrics = {}
        for key in next(iter(sample_summary.values())):
            values = np.asarray([row[key] for row in sample_summary.values()], dtype=np.float64)
            metrics[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "sampling_values": values.tolist(),
            }
        runs.append({"run": run_dir.name, "setting": run_dir.name.rsplit("_seed", 1)[0],
                     "training_seed": run_dir.name.rsplit("_seed", 1)[1],
                     "works": len(work_rows), "samples": len(sample_summary),
                     "metrics": metrics, "per_sample": sample_summary,
                     "per_work": work_rows})
        print(f"evaluated {run_dir.name}", flush=True)

    grouped = defaultdict(list)
    for run in runs:
        grouped[run["setting"]].append(run)
    settings = []
    metric_names = ("tempo_curve_pearson", "onset_w1", "duration_w1", "velocity_w1")
    for setting, members in sorted(grouped.items()):
        flattened = {name: [value for run in members
                            for value in run["metrics"][name]["sampling_values"]]
                     for name in metric_names}
        metrics = {}
        for name, values in flattened.items():
            values = np.asarray(values, dtype=np.float64)
            metrics[name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_sampling_draws": int(len(values)),
                "n_training_runs": len(members),
            }
        settings.append({"setting": setting, "training_runs": len(members),
                         "sampling_draws": sum(run["samples"] for run in members),
                         "metrics": metrics})

    result = {
        "protocol": {
            "metric_set": ["tempo_curve_pearson", "onset_w1", "duration_w1", "velocity_w1"],
            "onset_w1": "Per-note Wasserstein: for each score event, compare its predicted value with human performances of the same event, then macro-average notes and works; uses onset deviation. Onset ratio W1 is computed as an invariance check and should match exactly.",
            "duration_w1": "Per-note Wasserstein on log(performance duration / score duration), macro-averaged by note and work.",
            "velocity_w1": "Per-note Wasserstein on raw MIDI velocity, macro-averaged by note and work.",
            "tempo_curve_pearson": "Pearson correlation of beat-level log tempo against per-work median human curve.",
            "beat_mapper": "onset-group mean and local linear interpolation; no threshold, monotonic filter, or extrapolation.",
            "aggregation": "Within each sample, macro-average works; setting summaries pool completed training-run sampling draws. Training-run and sampling counts are recorded separately.",
        },
        "settings": settings,
        "runs": runs,
        "skipped": skipped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    for row in settings:
        print(row["setting"], "runs", row["training_runs"], "draws", row["sampling_draws"])
        for name, value in row["metrics"].items():
            print(f"  {name}: {value['mean']:.6f} ± {value['std']:.6f}")


if __name__ == "__main__":
    main()
