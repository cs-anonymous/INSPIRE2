#!/usr/bin/env python3
"""Plot normalized train/validation loss trends for PT and DINR mask runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PT_CSV = ROOT / "analysis/pt_original_baseline_8ep/losses.csv"
DINR_CSV = ROOT / "analysis/dinr_stage1_mask_ablation/losses.csv"
OUT_DIR = ROOT / "analysis/pt_original_baseline_8ep"

RUN_LABELS = {
    "pt": "Original PT (eval unmasked; incomplete: 5/8 epochs)",
    "attribute_random": "DINR attribute-random mask",
    "timestep_random": "DINR timestep-random mask",
    "no_mask": "DINR no mask",
    "nonpitch_coordinates": "DINR coordinates (all non-pitch attributes)",
}
COORDINATE_CSV = ROOT / "analysis/dinr_nonpitch_coordinates_heldout12_8ep.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def relative_change(values: list[float]) -> list[float]:
    baseline = values[0]
    return [(value / baseline - 1.0) * 100.0 for value in values]


def main() -> None:
    pt_rows = read_rows(PT_CSV)
    dinr_rows = read_rows(DINR_CSV)
    coordinate_rows = read_rows(COORDINATE_CSV)
    series: dict[str, dict[str, list[float]]] = {
        "pt": {
            "train": [float(row["train_loss"]) for row in pt_rows],
            "valid": [float(row["valid_loss"]) for row in pt_rows],
        }
    }
    for run in ("attribute_random", "timestep_random", "no_mask"):
        rows = sorted((row for row in dinr_rows if row["run"] == run), key=lambda row: int(row["epoch"]))
        series[run] = {
            "train": [float(row["train_loss_step_mean"]) for row in rows],
            "valid": [float(row["valid_loss"]) for row in rows],
        }
    coordinate_rows.sort(key=lambda row: int(row["epoch"]))
    series["nonpitch_coordinates"] = {
        "train": [float(row["train_loss"]) for row in coordinate_rows],
        "valid": [float(row["valid_loss"]) for row in coordinate_rows],
    }

    output_rows: list[dict[str, object]] = []
    for run, values in series.items():
        for split in ("train", "valid"):
            for epoch, (raw, relative) in enumerate(
                zip(values[split], relative_change(values[split])), start=1
            ):
                output_rows.append({
                    "run": run,
                    "label": RUN_LABELS[run],
                    "split": split,
                    "epoch": epoch,
                    "raw_loss": raw,
                    "change_from_epoch1_percent": relative,
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_csv = OUT_DIR / "pt_dinr_8ep_loss_trends.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    colors = {
        "pt": "#222222",
        "attribute_random": "#2878B5",
        "timestep_random": "#2E8B57",
        "no_mask": "#D9534F",
        "nonpitch_coordinates": "#9A6B24",
    }
    markers = {
        "pt": "o", "attribute_random": "s", "timestep_random": "^",
        "no_mask": "D", "nonpitch_coordinates": "P",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, split, title in zip(axes, ("train", "valid"), ("Training loss", "Validation loss")):
        for run, values in series.items():
            axis.plot(
                range(1, len(values[split]) + 1),
                relative_change(values[split]),
                color=colors[run], marker=markers[run], linewidth=2, markersize=5,
                label=RUN_LABELS[run],
            )
        axis.axhline(0, color="#777777", linewidth=0.8, alpha=0.6)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, 9))
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Loss change from epoch 1 (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Loss trends: PT, DINR mask ablations, and non-pitch coordinates")
    output_png = OUT_DIR / "pt_dinr_8ep_loss_stability.png"
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(output_png)
    print(merged_csv)


if __name__ == "__main__":
    main()
