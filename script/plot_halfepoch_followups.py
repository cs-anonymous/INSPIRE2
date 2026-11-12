#!/usr/bin/env python3
"""Plot follow-up campaigns separately from Section 7's original 12 runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUTPUT = ROOT / "analysis/epx1111_halfepoch"
BASE_CAMPAIGN = PROJECT / "results/halfepoch_earlystop_20261112"
PT_FULL8 = PROJECT / "results/pt_scratch_halfepoch_full8_2gpu"
ADD_COORD = PROJECT / "results/dinr_single_pitch_coord_queue_gpu2"
PURE_COORD = PROJECT / "results/dinr_single_pitch_pure_coord_queue_gpu2"
PURE_LINEAR_EXT = PROJECT / "results/dinr_single_pitch_pure_linear_16ep_gpu2"
PURE_MLP_EXT = PROJECT / "results/dinr_single_pitch_pure_mlp_16ep_gpu01"
GROUPS = [
    "PT scratch: early stop vs full 8ep",
    "Lookup + coordinate",
    "Pure coordinate replacement",
    "Pure MLP: budget and width",
]


def read_manifest(path):
    path = Path(path)
    if not (path / "manifest.json").exists():
        return []
    manifest = json.loads((path / "manifest.json").read_text())
    return manifest.get("jobs", [])


def add(jobs, group, job, label=None, run_dir=None):
    item = dict(job)
    item["followup_group"] = group
    item["label"] = label or job.get("label", job.get("id", "run"))
    if run_dir:
        item["run_dir"] = str(run_dir)
    jobs.append(item)


def collect():
    jobs = []
    base_jobs = {j["id"]: j for j in read_manifest(BASE_CAMPAIGN)}
    full_jobs = {j["id"]: j for j in read_manifest(PT_FULL8)}
    if "pt_scratch" in base_jobs:
        add(jobs, GROUPS[0], base_jobs["pt_scratch"], "PT scratch (early stop)")
    if "pt_scratch_full8" in full_jobs:
        add(jobs, GROUPS[0], full_jobs["pt_scratch_full8"], "PT scratch (full 8ep)")

    for j in base_jobs.values():
        if j["id"] == "dinr_single_pitch_separate_abs500":
            add(jobs, GROUPS[1], j, "Lookup only")
            add(jobs, GROUPS[2], j, "Lookup only baseline")
            add(jobs, GROUPS[3], j, "Lookup only (8ep, 3e-4 linear)")
    for j in read_manifest(ADD_COORD):
        add(jobs, GROUPS[1], j)
    for j in read_manifest(PURE_COORD):
        if j.get("status") in {"failed", "cancelled"}:
            continue
        add(jobs, GROUPS[2], j)
    pure_by_id = {j["id"]: j for j in read_manifest(PURE_COORD)}
    if "dinr_single_pitch_separate_abs500_pure_coord_mlp" in pure_by_id:
        add(jobs, GROUPS[3], pure_by_id["dinr_single_pitch_separate_abs500_pure_coord_mlp"],
            "Pure MLP-32 (8ep, 3e-4 linear)")
    for j in read_manifest(PURE_LINEAR_EXT):
        add(jobs, GROUPS[2], j, "Pure linear (16ep, 3e-4 linear)")
    for j in read_manifest(PURE_MLP_EXT):
        add(jobs, GROUPS[3], j, f"Pure MLP-{j.get('hidden_size')} (16ep, 5e-4 cosine)")
    return jobs


def read_history(job):
    paths = list(Path(job["run_dir"]).rglob("loss_history.jsonl"))
    if len(paths) != 1:
        return []
    rows = []
    for line in paths[0].read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    jobs = collect()
    histories = {j["id"] + "::" + j["followup_group"]: read_history(j) for j in jobs}
    rows = []
    max_epoch = 8
    for j in jobs:
        key = j["id"] + "::" + j["followup_group"]
        events = histories[key]
        for event in events:
            max_epoch = max(max_epoch, int(float(event.get("epoch", 0)) + 0.999999))
            for metric, split in (("loss", "train"), ("eval_loss", "valid")):
                if metric in event:
                    rows.append(dict(group=j["followup_group"], run=j["id"], label=j["label"],
                                     split=split, epoch=event.get("epoch"),
                                     step=event.get("step"), loss=event[metric]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "followup_all_losses.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "run", "label", "split", "epoch", "step", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "followup_all_losses.csv.tmp").replace(OUTPUT / "followup_all_losses.csv")
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(5.1 * len(GROUPS), 9),
                             squeeze=False, constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for col, group in enumerate(GROUPS):
        grouped = [j for j in jobs if j["followup_group"] == group]
        for index, job in enumerate(grouped):
            events = histories[job["id"] + "::" + group]
            color = palette(index)
            train = [r for r in events if "loss" in r and "eval_loss" not in r]
            valid = [r for r in events if "eval_loss" in r]
            if train:
                axes[0, col].plot([r["epoch"] for r in train], [r["loss"] for r in train],
                                  color=color, linewidth=0.85, label=job["label"])
            if valid:
                axes[1, col].plot([r["epoch"] for r in valid], [r["eval_loss"] for r in valid],
                                  color=color, linewidth=1.6, marker="o", markersize=3,
                                  label=job["label"])
                best = min(valid, key=lambda r: r["eval_loss"])
                axes[1, col].scatter([best["epoch"]], [best["eval_loss"]], marker="*",
                                     s=90, color=color, edgecolors="black", linewidths=0.5)
        for row in range(2):
            axis = axes[row, col]
            axis.set_title(group if row == 0 else "")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Train loss" if row == 0 else "Valid loss")
            axis.set_xlim(0, max_epoch)
            axis.set_xticks(range(0, max_epoch + 1, 2 if max_epoch > 8 else 1))
            axis.grid(alpha=0.2)
            if grouped and any(histories[j["id"] + "::" + group] for j in grouped):
                axis.legend(fontsize=7.5, frameon=False)
            else:
                axis.text(0.5, 0.5, "No measurements yet", ha="center", va="center",
                          transform=axis.transAxes, color="#777777")
    fig.suptitle("Follow-up experiments: grouped train and validation curves", fontsize=16)
    fig.savefig(OUTPUT / "followup_groups.png", dpi=170)
    fig.savefig(OUTPUT / "followup_groups.pdf")
    plt.close(fig)
    print(OUTPUT / "followup_groups.png")


if __name__ == "__main__":
    main()
