#!/usr/bin/env python3
"""Validate 19 x 2 generations and report the shared DINR HR definition."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate.evaluate_dinr_absolute import midi_note_arrays

ANCHORS = {"ioi_wass": 22.772456164440005, "duration_wass": 75.08565114668764, "velocity_wass": 10.580359191566208}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["pt", "dinr"], required=True)
    args = parser.parse_args()
    output = args.run_dir / "test_sampling_2"
    if args.kind == "dinr":
        metrics = json.loads((output / "metrics.json").read_text())
        if (metrics["works"], metrics["num_samples"], metrics["num_predictions"]) != (19, 2, 38):
            raise ValueError("Expected 19 works x 2 samples")
        rows = metrics["rows"]
        if len(rows) != 38 or any(row.get("error") for row in rows):
            raise ValueError("Inference contains failed or missing generations")
        if {(r["work"], r["sample"]) for r in rows} != {(w, s) for w in range(19) for s in range(2)}:
            raise ValueError("Unexpected work/sample coverage")
        raw = {key: metrics["pn_wass"][key] for key in ANCHORS}
    else:
        manifest = json.loads((output / "prediction_manifest.json").read_text())
        items = manifest["items"]
        if len(items) != 19 or any(len(item["prediction_paths"]) != 2 for item in items):
            raise ValueError("Expected 19 works x 2 samples")
        rows = []
        for item in items:
            predictions = [midi_note_arrays(p) for p in item["prediction_paths"]]
            references = [midi_note_arrays(p) for p in item["ground_truth_paths"]]
            usable = min(map(len, predictions + references))
            if usable == 0:
                raise ValueError(f"Empty MIDI: {item['score_source']}")
            row = {}
            for key, column in (("ioi_wass", 0), ("duration_wass", 1), ("velocity_wass", 2)):
                row[key] = float(np.mean([
                    np.mean([wasserstein_distance([pred[note, column]], [ref[note, column] for ref in references])
                             for pred in predictions]) for note in range(usable)
                ]))
            rows.append(row)
        raw = {key: float(np.mean([row[key] for row in rows])) for key in ANCHORS}
    if not all(np.isfinite(list(raw.values()))):
        raise ValueError("Non-finite Wasserstein metric")
    hr = {key.removesuffix("_wass") + "_hr": raw[key] / anchor for key, anchor in ANCHORS.items()}
    hr["pn_hr"] = sum(hr.values()) / 3
    result = dict(protocol="DINR note-index W1 / fixed human anchor; 19 works x 2 samples",
                  raw_wass=raw, hr=hr, human_pn_anchors=ANCHORS, works=19, samples_per_work=2, predictions=38)
    (args.run_dir / "hr.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
