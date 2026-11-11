"""Recompute pooled PN Wasserstein metrics from DINR-Absolute samples.

This keeps the median-scaled absolute-time values and only aligns aggregation
with the outer evaluator: three predictions are pooled at each note, and
pedal values are thresholded to binary 0/1 before Wasserstein computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from src.evaluate.evaluate_dinr_absolute import midi_note_arrays


def work_metrics(work_dir: Path, include_pedal: bool):
    predictions = [midi_note_arrays(path) for path in sorted(work_dir.glob("sample_*/prediction.mid"))]
    references = [midi_note_arrays(path) for path in sorted(work_dir.glob("gt_*.mid"))]
    if len(predictions) != 3 or not references:
        raise ValueError(f"Expected 3 predictions and at least one reference in {work_dir}")
    usable = min(len(item) for item in predictions + references)
    metrics = {}
    for name, column in (("ioi_wass", 0), ("duration_wass", 1), ("velocity_wass", 2)):
        metrics[name] = float(np.mean([
            wasserstein_distance(
                [item[note, column] for item in predictions],
                [item[note, column] for item in references],
            )
            for note in range(usable)
        ]))
    if include_pedal:
        pedal_values = []
        for column in range(3, 7):
            value = float(np.mean([
                wasserstein_distance(
                    [float(item[note, column] >= 64.0) for item in predictions],
                    [float(item[note, column] >= 64.0) for item in references],
                )
                for note in range(usable)
            ]))
            pedal_values.append(value)
        metrics["pedal_wass"] = float(np.mean(pedal_values))
    keys = ["ioi_wass", "duration_wass", "velocity_wass"]
    if include_pedal:
        keys.append("pedal_wass")
    metrics["pn_wass_mean"] = float(np.mean([metrics[key] for key in keys]))
    metrics["pn_shared3_mean"] = float(np.mean([metrics[key] for key in keys[:3]]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pedal-mode", choices=["ce", "bce", "none"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for work_dir in sorted(args.root.glob("score_*")):
        rows.append({"score": work_dir.name, "pn_wass": work_metrics(work_dir, args.pedal_mode != "none")})
    keys = rows[0]["pn_wass"].keys() if rows else []
    aggregate = {key: float(np.mean([row["pn_wass"][key] for row in rows])) for key in keys}
    result = {
        "root": str(args.root.resolve()),
        "pedal_mode": args.pedal_mode,
        "works": len(rows),
        "num_samples": 3,
        "timing_protocol": "median-scaled absolute milliseconds",
        "aggregation": "pooled 3 predictions and all references at each aligned note",
        "pedal_protocol": "threshold 64, binary 0/1" if args.pedal_mode != "none" else "not included",
        "aggregate": aggregate,
        "scores": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"works": len(rows), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
