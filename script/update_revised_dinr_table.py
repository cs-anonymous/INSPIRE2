#!/usr/bin/env python3
"""Replace superseded DINR mask rows with the six corrected reruns."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "analysis/dinr_six_model_hr_comparison/all_experiments.csv"
DOC_PATH = ROOT / "docs/EXPERIMENTS_COMPLETE.md"
RUN_ROOT = ROOT / "models/dinr_mask_revised_heldout12_8ep"
ANCHORS = {
    "ioi_hr": 22.772456164440005,
    "duration_hr": 75.08565114668764,
    "velocity_hr": 10.580359191566208,
}
RUNS = {
    ("DINR separate", "attribute", "abs 500 token"): "separate_attribute_abs500",
    ("DINR shared", "attribute", "abs 500 token"): "shared_attribute_abs500",
    ("DINR separate", "timestep retain pitch", "abs 500 token"): "separate_retain_pitch_abs500",
    ("DINR shared", "timestep retain pitch", "abs 500 token"): "shared_retain_pitch_abs500",
    ("DINR separate", "timestep retain pitch", "log abs"): "separate_retain_pitch_log_abs",
    ("DINR separate", "timestep retain pitch", "log dev"): "separate_retain_pitch_log_dev",
}
FIELDS = [
    "method", "training_mask", "time_representation", "best_valid_epoch",
    "best_valid_loss", "ioi_hr", "duration_hr", "velocity_hr", "pn_hr",
]


def load_run(run_name: str) -> dict[str, str]:
    run_dir = RUN_ROOT / run_name
    history_path = run_dir / "trainer_log_history.json"
    metrics_path = run_dir / "test_eval_sampling_w8_s2/metrics.json"
    if not history_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing completed training or inference output for {run_name}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    eval_rows = [row for row in history if "eval_loss" in row and "epoch" in row]
    if len(eval_rows) != 8:
        raise ValueError(f"Expected 8 validation epochs for {run_name}, found {len(eval_rows)}")
    best = min(eval_rows, key=lambda row: row["eval_loss"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics.get("works"), metrics.get("num_samples"), metrics.get("num_predictions")) != (19, 2, 38):
        raise ValueError(f"Unexpected inference protocol for {run_name}: {metrics}")
    w1 = metrics["pn_wass"]
    ioi_hr = w1["ioi_wass"] / ANCHORS["ioi_hr"]
    duration_hr = w1["duration_wass"] / ANCHORS["duration_hr"]
    velocity_hr = w1["velocity_wass"] / ANCHORS["velocity_hr"]
    return {
        "best_valid_epoch": str(int(round(best["epoch"]))),
        "best_valid_loss": f"{best['eval_loss']:.10f}",
        "ioi_hr": f"{ioi_hr:.10f}",
        "duration_hr": f"{duration_hr:.10f}",
        "velocity_hr": f"{velocity_hr:.10f}",
        "pn_hr": f"{(ioi_hr + duration_hr + velocity_hr) / 3:.10f}",
    }


def render_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| method | 训练mask | time表示 | Best valid epoch | Best valid loss | IOI HR | Dur HR | Vel HR | PN HR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        epoch = row["best_valid_epoch"]
        loss = f"{float(row['best_valid_loss']):.6f}"
        ioi = f"{float(row['ioi_hr']):.4f}"
        duration = f"{float(row['duration_hr']):.4f}"
        velocity = f"{float(row['velocity_hr']):.4f}"
        pn = f"**{float(row['pn_hr']):.4f}**"
        lines.append(
            f"| {row['method']} | {row['training_mask']} | {row['time_representation']} | "
            f"{epoch} | {loss} | {ioi} | {duration} | {velocity} | {pn} |"
        )
    return lines


def main() -> None:
    with TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    obsolete = [row for row in rows if row["training_mask"] == "timestep"]
    if len(obsolete) != 2:
        raise ValueError(f"Expected exactly two superseded timestep rows, found {len(obsolete)}")
    rows = [row for row in rows if row["training_mask"] != "timestep"]

    for key, run_name in RUNS.items():
        matches = [
            row for row in rows
            if (row["method"], row["training_mask"], row["time_representation"]) == key
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one existing unified-table row for {key}, found {len(matches)}")
        matches[0].update(load_run(run_name))

    if len(rows) != 10:
        raise ValueError(f"Expected 10 unified result rows after replacement, found {len(rows)}")
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    doc = DOC_PATH.read_text(encoding="utf-8")
    lines = doc.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("| method |")), None)
    if start is None:
        raise ValueError("Could not find unified results table in the experiment document")
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[start:end] = render_table(rows)
    doc = "\n".join(lines) + "\n"
    doc = doc.replace(
        "共 12 个实验：PT 两个实验固定放在前两行，之后是原六组 DINR 和四个新增 DINR 实验。",
        "共 10 个实验：PT 两个实验固定放在前两行；DINR 包含本次修正后的 6 组 mask/time 实验和 2 个 no-mask 基线。",
    )
    DOC_PATH.write_text(doc, encoding="utf-8")
    print(f"Updated {TABLE_PATH} and the unified table in {DOC_PATH}")


if __name__ == "__main__":
    main()
