#!/usr/bin/env python3
"""Merge batched exp1115 PN metric evaluations and aggregate by training seed."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/exp1115_beat"
PARTS = sorted(ANALYSIS.glob("pn_official_*.json"))
OUTPUT = ANALYSIS / "pn_metrics.json"
METRICS = ("tempo_curve_pearson", "onset_w1", "duration_w1", "velocity_w1")
OFFICIAL_RUNS = (
    "A1_seed42", "A2_seed42", "A3_seed42", "B1_seed42", "B2_seed42",
    "D1_seed42", "D2_seed42", "D3_seed45", "D4_seed42", "D5_seed45",
)


def main():
    available = []
    for path in PARTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        available.extend(payload["runs"])
    by_name = {run["run"]: run for run in available}
    runs = [by_name[name] for name in OFFICIAL_RUNS if name in by_name]
    missing = [name for name in OFFICIAL_RUNS if name not in by_name]

    grouped = defaultdict(list)
    for run in runs:
        grouped[run["setting"]].append(run)
    settings = []
    for setting, members in sorted(grouped.items()):
        metrics = {}
        for name in METRICS:
            by_run = np.asarray([run["metrics"][name]["mean"] for run in members], dtype=np.float64)
            sampling_std = np.asarray([run["metrics"][name]["std"] for run in members], dtype=np.float64)
            metrics[name] = {
                "mean": float(np.nanmean(by_run)),
                "training_seed_std": float(np.nanstd(by_run, ddof=1)) if len(by_run) > 1 else None,
                "mean_sampling_std": float(np.nanmean(sampling_std)),
                "training_runs": len(members),
                "sampling_draws_per_run": [run["samples"] for run in members],
            }
        settings.append({"setting": setting, "training_runs": len(members),
                         "sampling_draws": sum(run["samples"] for run in members),
                         "metrics": metrics})

    first = json.loads(PARTS[0].read_text(encoding="utf-8"))
    output = {
        "protocol": first["protocol"],
        "aggregation_note": "Official exp1115 set: one completed training run per setting, each with three existing AR MIDI draws. Setting mean is the run's mean across sampling draws; sampling std is reported separately. Training-seed std is not estimable for the official single-run setting.",
        "settings": settings,
        "runs": sorted(runs, key=lambda run: run["run"]),
        "missing_official_runs": missing,
        "excluded_additional_runs": sorted(run["run"] for run in available
                                            if run["run"] not in OFFICIAL_RUNS),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for row in settings:
        print(f"{row['setting']}: {row['training_runs']} runs, {row['sampling_draws']} draws")
        for name, values in row["metrics"].items():
            seed_std = values['training_seed_std']
            seed_text = "n/a" if seed_std is None else f"{seed_std:.6f}"
            print(f"  {name}: {values['mean']:.6f} ± seed {seed_text}; "
                  f"sampling SD {values['mean_sampling_std']:.6f}")
    print(f"Wrote {OUTPUT}; official runs missing: {missing}")


if __name__ == "__main__":
    main()
