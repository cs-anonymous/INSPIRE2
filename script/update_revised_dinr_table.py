#!/usr/bin/env python3
"""Refresh DINR results while retaining the legacy full-timestep baseline."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "analysis/dinr_six_model_hr_comparison/all_experiments.csv"
DOC_PATH = ROOT / "docs/epx1111_DINR_VAR.md"
RUN_ROOT = ROOT / "models/dinr_mask_revised_heldout12_8ep"
HISTORICAL_SUMMARY_PATH = ROOT / "analysis/dinr_six_model_hr_comparison/summary.json"
ANCHORS = {
    "ioi_hr": 22.772456164440005,
    "duration_hr": 75.08565114668764,
    "velocity_hr": 10.580359191566208,
}
RUNS = {
    ("DINR separate", "attribute", "abs 500 token"): "separate_attribute_abs500",
    ("DINR shared", "attribute", "abs 500 token"): "shared_attribute_abs500",
    ("DINR separate", "timestep 3+pitch", "abs 500 token"): "separate_retain_pitch_abs500",
    ("DINR shared", "timestep 3+pitch", "abs 500 token"): "shared_retain_pitch_abs500",
    ("DINR separate", "timestep 3+pitch", "log abs"): "separate_retain_pitch_log_abs",
    ("DINR separate", "timestep 3+pitch", "log dev"): "separate_retain_pitch_log_dev",
    ("DINR separate", "timestep 1+pitch", "abs 500 token"): "separate_retain_pitch_single_vector_abs500",
    ("DINR shared", "timestep 1+pitch", "abs 500 token"): "shared_retain_pitch_single_vector_abs500",
    ("DINR separate", "timestep 1+pitch", "log abs"): "separate_retain_pitch_single_vector_log_abs",
    ("DINR separate", "timestep 1+pitch", "log dev"): "separate_retain_pitch_single_vector_log_dev",
}
FOLLOWUP_RUNS = {
    "shared_retain_pitch_single_vector_abs500",
    "separate_retain_pitch_single_vector_log_abs",
    "separate_retain_pitch_single_vector_log_dev",
}
FIELDS = [
    "method", "training_mask", "time_representation", "best_valid_epoch",
    "best_valid_loss", "ioi_hr", "duration_hr", "velocity_hr", "pn_hr",
]
METHOD_ORDER = {"PT base-1": 0, "PT scratch": 1, "DINR separate": 2, "DINR shared": 3}
MASK_ORDER = {
    "attribute": 0,
    "timestep": 1,
    "none": 2,
    "timestep 3+pitch": 3,
    "timestep 1+pitch": 4,
    "non-pitch token": 0,
}
TIME_ORDER = {"abs 500 token": 0, "log abs": 1, "log dev": 2, "abs 5000 token": 0}


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
    predictions = metrics.get("rows", [])
    if len(predictions) != 38 or any(row.get("error") for row in predictions):
        raise ValueError(f"Incomplete or failed test predictions for {run_name}")
    if {(row["work"], row["sample"]) for row in predictions} != {
        (work, sample) for work in range(19) for sample in range(2)
    }:
        raise ValueError(f"Unexpected test work/sample coverage for {run_name}")
    w1 = metrics["pn_wass"]
    ioi_hr = w1["ioi_wass"] / ANCHORS["ioi_hr"]
    duration_hr = w1["duration_wass"] / ANCHORS["duration_hr"]
    velocity_hr = w1["velocity_wass"] / ANCHORS["velocity_hr"]
    if not all(math.isfinite(value) for value in (ioi_hr, duration_hr, velocity_hr)):
        raise ValueError(f"Non-finite HR for {run_name}")
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
        epoch = row["best_valid_epoch"] or "—"
        loss = f"{float(row['best_valid_loss']):.6f}" if row["best_valid_loss"] else "—"
        ioi = f"{float(row['ioi_hr']):.4f}" if row["ioi_hr"] else "—"
        duration = f"{float(row['duration_hr']):.4f}" if row["duration_hr"] else "—"
        velocity = f"{float(row['velocity_hr']):.4f}" if row["velocity_hr"] else "—"
        pn = f"**{float(row['pn_hr']):.4f}**" if row["pn_hr"] else "—"
        lines.append(
            f"| {row['method']} | {row['training_mask']} | {row['time_representation']} | "
            f"{epoch} | {loss} | {ioi} | {duration} | {velocity} | {pn} |"
        )
    return lines


def load_legacy_timestep_rows() -> list[dict[str, str]]:
    summary = json.loads(HISTORICAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    ranking = {row["model"]: row for row in summary["ranking"]}
    rows = []
    for model, method in (
        ("separate_timestep", "DINR separate"),
        ("shared_timestep", "DINR shared"),
    ):
        result = ranking[model]
        if (result.get("works"), result.get("samples_per_work"), result.get("predictions")) != (19, 2, 38):
            raise ValueError(f"Unexpected legacy timestep test protocol for {model}: {result}")
        rows.append({
            "method": method,
            "training_mask": "timestep",
            "time_representation": "abs 500 token",
            "best_valid_epoch": str(result["best_epoch"]),
            "best_valid_loss": f"{result['best_valid_loss']:.10f}",
            "ioi_hr": f"{result['ioi_hr']:.10f}",
            "duration_hr": f"{result['duration_hr']:.10f}",
            "velocity_hr": f"{result['velocity_hr']:.10f}",
            "pn_hr": f"{result['pn_hr']:.10f}",
        })
    return rows


def main() -> None:
    with TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["training_mask"] = {
            "timestep retain pitch": "timestep 3+pitch",
            "timestep retain pitch (single vector)": "timestep 1+pitch",
        }.get(row["training_mask"], row["training_mask"])

    for legacy in load_legacy_timestep_rows():
        key = (legacy["method"], legacy["training_mask"], legacy["time_representation"])
        matches = [
            row for row in rows
            if (row["method"], row["training_mask"], row["time_representation"]) == key
        ]
        if len(matches) > 1:
            raise ValueError(f"Found duplicate legacy timestep rows for {key}")
        if matches:
            matches[0].update(legacy)
        else:
            rows.append(legacy)

    for key, run_name in RUNS.items():
        matches = [
            row for row in rows
            if (row["method"], row["training_mask"], row["time_representation"]) == key
        ]
        if len(matches) > 1:
            raise ValueError(f"Found duplicate unified-table rows for {key}")
        if not matches and key[1] == "timestep 1+pitch":
            row = dict.fromkeys(FIELDS, "")
            row.update({"method": key[0], "training_mask": key[1], "time_representation": key[2]})
            rows.append(row)
            matches = [row]
        if len(matches) != 1:
            raise ValueError(f"Expected one existing unified-table row for {key}, found {len(matches)}")
        metrics_path = RUN_ROOT / run_name / "test_eval_sampling_w8_s2/metrics.json"
        if run_name in FOLLOWUP_RUNS and not metrics_path.is_file():
            continue
        matches[0].update(load_run(run_name))

    rows.sort(key=lambda row: (
        METHOD_ORDER[row["method"]] if row["method"].startswith("PT ") else 2,
        MASK_ORDER.get(row["training_mask"], 9),
        TIME_ORDER.get(row["time_representation"], 9),
        0 if row["method"] == "DINR separate" else 1,
    ))
    if len(rows) != 16:
        raise ValueError(f"Expected 16 unified result rows, found {len(rows)}")
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
    completed = sum(bool(row["pn_hr"]) for row in rows)
    pending = len(rows) - completed
    doc = re.sub(
        r"共 \d+ 个实验[^。]*。",
        f"共 16 个实验配置，已完成 {completed} 个，待完成 {pending} 个：PT 两个实验固定放在前两行；"
        "DINR 包含 2 组 attribute、2 组 legacy timestep、2 组 no-mask、4 组 timestep 3+pitch 和 4 组 timestep 1+pitch。",
        doc,
        count=1,
    )
    # Keep the validation table in section 4 in sync as follow-up runs finish.
    lines = doc.splitlines()
    validation_start = next(i for i, line in enumerate(lines) if line.startswith("| Mask | Table |"))
    validation_end = validation_start + 1
    while validation_end < len(lines) and lines[validation_end].startswith("|"):
        validation_end += 1
    for key, run_name in RUNS.items():
        if key[1] != "timestep 1+pitch":
            continue
        table = key[0].removeprefix("DINR ")
        prefix = f"| {key[1]} | {table} | {key[2]} |"
        history_path = RUN_ROOT / run_name / "trainer_log_history.json"
        values = ["—", "—", "—"]
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            eval_rows = [row for row in history if "eval_loss" in row and "epoch" in row]
            if len(eval_rows) != 8:
                raise ValueError(f"Expected 8 validation epochs for {run_name}")
            best = min(eval_rows, key=lambda row: row["eval_loss"])
            values = [str(int(round(best["epoch"]))), f"{best['eval_loss']:.6f}", f"{eval_rows[-1]['eval_loss']:.6f}"]
        line = prefix + " " + " | ".join(values) + " |"
        position = next((i for i in range(validation_start, validation_end) if lines[i].startswith(prefix)), None)
        if position is None:
            lines.insert(validation_end, line)
            validation_end += 1
        else:
            lines[position] = line
    doc = "\n".join(lines) + "\n"
    DOC_PATH.write_text(doc, encoding="utf-8")
    print(f"Updated {TABLE_PATH} and the unified table in {DOC_PATH}")


if __name__ == "__main__":
    main()
