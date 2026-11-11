"""Compare saved outer DINR outputs against the current median-scaled ASAP GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_process.dinr_dataset import _performance_tokens
from src.evaluate.evaluate_inr_saved_midis import cached_note_arrays
from src.evaluate.evaluate_dinr_absolute import expand_grouped_ioi, raw_to_midi


SIDECAR_NAME = "score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt"


def score_metric(prediction_paths, references, include_pedal=True):
    predictions = [cached_note_arrays(str(path.resolve())) for path in prediction_paths]
    gt = [cached_note_arrays(str(path.resolve())) for path in references]
    usable = min([len(x["ioi"]) for x in predictions + gt], default=0)
    result = {}
    for feature in ("ioi", "duration", "velocity"):
        result[f"{feature}_wass"] = float(np.mean([
            wasserstein_distance(
                [item[feature][idx] for item in predictions],
                [item[feature][idx] for item in gt],
            )
            for idx in range(usable)
        ]))
    if include_pedal:
        columns = ("pedal_0", "pedal_25", "pedal_50", "pedal_75")
        result["pedal_wass"] = float(np.mean([
            np.mean([
                wasserstein_distance(
                    [float(item[key][idx] >= 64.0) for item in predictions],
                    [float(item[key][idx] >= 64.0) for item in gt],
                )
                for idx in range(usable)
            ])
            for key in columns
        ]))
    return result


def aggregate(rows, include_pedal):
    keys = ["ioi_wass", "duration_wass", "velocity_wass"]
    if include_pedal:
        keys.append("pedal_wass")
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    result["pn_shared3_mean"] = float(np.mean([result[key] for key in keys[:3]]))
    if include_pedal:
        result["pn_wass_mean"] = float(np.mean([result[key] for key in keys]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, default=ROOT / "data/ASAP_processed")
    parser.add_argument(
        "--outer-manifest",
        type=Path,
        default=ROOT / "results/inr_epr_pipeline/dinr_slot_coord_matrix_20261026/slot_sum__no_coord/sampling/prediction_manifest.json",
    )
    parser.add_argument("--absolute-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    outer = json.loads(args.outer_manifest.read_text(encoding="utf-8"))
    outer_by_source = {item["score_source"]: item for item in outer["items"]}
    sidecars = sorted(args.sidecar_root.resolve().rglob(SIDECAR_NAME))
    sidecar_index = {path: idx for idx, path in enumerate(sidecars)}
    rows = {"outer_slot_sum_no_coord": [], "dinr_absolute_final": []}
    score_sources = []
    reference_root = args.output.parent / "median_scaled_gt_midis"

    for sidecar in sidecars:
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        score_source = payload.get("meta", {}).get("score_source")
        if score_source not in outer_by_source:
            continue
        performances = [p for p in payload.get("performances", []) if p.get("split") == "test"]
        if not performances:
            continue
        score_sources.append(score_source)
        score = payload["score"]
        score_raw = np.asarray(score["score_raw"], dtype=np.float64)
        score_tokens, _, zero_mask = _performance_tokens(score, performances[0], include_pedal=True)
        work_ref_dir = reference_root / f"{sidecar_index[sidecar]:02d}"
        gt_paths = []
        for ref_index, performance in enumerate(performances):
            shared = np.asarray(performance["label_shared_raw"], dtype=np.float64)
            grouped = np.asarray(performance.get("grouped_ioi_raw", shared[:, 0]), dtype=np.float64)
            ioi = expand_grouped_ioi(score_raw, grouped, zero_mask)
            pedals = np.asarray(performance.get("label_pedal4_raw", np.zeros((len(shared), 4))), dtype=np.float64)
            raw = np.column_stack([ioi, shared[:, 1], shared[:, 2], pedals])
            gt_path = work_ref_dir / f"gt_{ref_index:03d}.mid"
            raw_to_midi(score["pitch"], raw, gt_path)
            gt_paths.append(gt_path)

        work_index = sidecar_index[sidecar]
        outer_paths = [Path(p) for p in outer_by_source[score_source]["prediction_paths"]]
        abs_paths = [
            args.absolute_root / f"score_{work_index:02d}" / f"sample_{sample:03d}" / "prediction.mid"
            for sample in range(3)
        ]
        if any(not path.exists() for path in outer_paths + abs_paths):
            raise FileNotFoundError(f"Missing predictions for {score_source}")
        rows["outer_slot_sum_no_coord"].append({
            "score_source": score_source,
            "references": len(gt_paths),
            **score_metric(outer_paths, gt_paths, include_pedal=True),
        })
        rows["dinr_absolute_final"].append({
            "score_source": score_source,
            "references": len(gt_paths),
            **score_metric(abs_paths, gt_paths, include_pedal=True),
        })
        print(f"evaluated {len(score_sources)}/17: {score_source}", flush=True)

    result = {
        "timing_protocol": "direct comparison to the already grouped-median-scaled performance labels; no score-span normalization",
        "test_scores": len(score_sources),
        "test_performances": int(sum(row["references"] for row in rows["outer_slot_sum_no_coord"])),
        "num_samples_per_score": 3,
        "score_sources": score_sources,
        "outer": {
            "aggregation": aggregate(rows["outer_slot_sum_no_coord"], include_pedal=True),
            "scores": rows["outer_slot_sum_no_coord"],
        },
        "dinr_absolute": {
            "aggregation": aggregate(rows["dinr_absolute_final"], include_pedal=True),
            "scores": rows["dinr_absolute_final"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "test_scores": len(score_sources),
        "outer": result["outer"]["aggregation"],
        "dinr_absolute": result["dinr_absolute"]["aggregation"],
    }, indent=2))


if __name__ == "__main__":
    main()
