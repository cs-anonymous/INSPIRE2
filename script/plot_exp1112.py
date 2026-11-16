#!/usr/bin/env python3
"""Plot exp1112 DINR train and valid curves in four experiment groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def history(job):
    path = Path(job["run_dir"]) / "loss_history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def table_fields(job):
    """Return the exp1112 table's three configuration columns."""
    if job.get("coordinate_encoding"):
        method = f"DINR {job['table']}"
        encoding = {"mlp": "MLP", "linear": "Linear", "fourier": "Fourier"}[job["coordinate_encoding"]]
        width = job.get("hidden_size")
        mapping = f"{encoding}{width}" if encoding == "MLP" and width else encoding
        time = f"pure coordinate {mapping}"
    elif job.get("method"):
        method = job["method"]
    elif job.get("table"):
        method = f"DINR {job['table']}"
    else:
        method = job.get("label", job["id"])
    mask = job.get("mask", "")
    if not mask and " / " in job.get("label", ""):
        mask = job["label"].split(" / ", 1)[1]
    if not job.get("coordinate_encoding"):
        time = {"log_absolute": "log abs", "score_relative_log": "log dev"}.get(
            job.get("target"), job.get("time", ""))
    return method, mask, time


def curve_label(job):
    if job.get("coordinate_encoding"):
        _, mask, time = table_fields(job)
        return f"{time.removeprefix('pure coordinate ')} / {mask}"
    return job["label"]


def render(campaign: Path, output: Path, groups=None, stem="dinr_groups", include_source_controls=False):
    manifest = json.loads((campaign / "manifest.json").read_text())
    groups = list(groups or manifest["protocol"]["groups"])
    group_order = {group: index for index, group in enumerate(groups)}
    jobs = sorted(manifest["jobs"], key=lambda job: group_order.get(job["group"], len(group_order)))
    plotted = [job for job in jobs if job["group"] in groups]
    if include_source_controls:
        by_id = {job["id"]: job for job in jobs}
        controls = set()
        for job in list(plotted):
            if job.get("source_config") and (job["group"], job["source_config"]) not in controls:
                control = dict(by_id[job["source_config"]], group=job["group"])
                plotted.append(control)
                controls.add((job["group"], job["source_config"]))
    histories = {job["id"]: history(job) for job in jobs}
    rows = []
    for job in jobs:
        for event in histories[job["id"]]:
            for key, split in (("loss", "train"), ("eval_loss", "valid")):
                if key in event:
                    rows.append({"group": job["group"], "run": job["id"], "label": curve_label(job),
                                 "split": split, "step": event.get("step"),
                                 "epoch": event.get("epoch"), "loss": event[key]})
    output.mkdir(parents=True, exist_ok=True)
    loss_name = "all_losses" if stem == "dinr_groups" else stem + "_losses"
    with (output / f"{loss_name}.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "run", "label", "split", "step", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{loss_name}.csv.tmp").replace(output / f"{loss_name}.csv")

    fig, axes = plt.subplots(2, len(groups), figsize=(5.3 * len(groups), 9),
                             squeeze=False, constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for col, group in enumerate(groups):
        grouped = [job for job in plotted if job["group"] == group]
        max_epoch = max((event["epoch"] for job in grouped
                         for event in histories[job["id"]]
                         if event.get("epoch") is not None
                         and ("loss" in event or "eval_loss" in event)), default=1)
        for index, job in enumerate(grouped):
            events = histories[job["id"]]
            color = palette(index)
            train = [row for row in events if "loss" in row and "eval_loss" not in row]
            valid = [row for row in events if "eval_loss" in row]
            if train:
                axes[0, col].plot([r["epoch"] for r in train], [r["loss"] for r in train],
                                  linewidth=0.8, color=color, label=curve_label(job))
            if valid:
                axes[1, col].plot([r["epoch"] for r in valid], [r["eval_loss"] for r in valid],
                                  marker="o", markersize=3, linewidth=1.5,
                                  color=color, label=curve_label(job))
                best = min(valid, key=lambda r: r["eval_loss"])
                axes[1, col].scatter([best["epoch"]], [best["eval_loss"]], marker="*", s=95,
                                     color=color, edgecolors="black", linewidths=0.5)
        for row in range(2):
            axis = axes[row, col]
            axis.set_title(group if row == 0 else "")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Train loss" if row == 0 else "Valid loss")
            axis.set_xlim(0, max(max_epoch, 0.1) * 1.03)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.grid(alpha=0.2)
            if grouped:
                axis.legend(fontsize=7.5, frameon=False)
            else:
                axis.text(0.5, 0.5, "No measurements", ha="center", va="center", transform=axis.transAxes)
    fig.suptitle("exp1112 DINR abs5000: training and validation", fontsize=15)
    fig.savefig(output / f"{stem}.png", dpi=170)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)

    summary = []
    for job in jobs:
        valid = [r for r in histories[job["id"]] if "eval_loss" in r]
        best = min(valid, key=lambda r: r["eval_loss"]) if valid else {}
        hr = job.get("hr", {}).get("hr", {})
        method, mask, time = table_fields(job)
        summary.append({"run": job["id"], "group": job["group"], "label": curve_label(job),
                        "method": method, "mask": mask, "time": time,
                        "status": job["status"], "best_epoch": best.get("epoch", job.get("best_epoch", "")),
                        "best_valid": best.get("eval_loss", job.get("best_valid_loss", "")),
                        "final_epoch": job.get("final_epoch", ""), "validation_count": len(valid),
                        "ioi_hr": hr.get("ioi_hr", ""), "duration_hr": hr.get("duration_hr", ""),
                        "velocity_hr": hr.get("velocity_hr", ""), "pn_hr": hr.get("pn_hr", "")})
    with (output / "all_experiments.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "all_experiments.csv.tmp").replace(output / "all_experiments.csv")
    print(output / f"{stem}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--stem", default="dinr_groups")
    parser.add_argument("--include-source-controls", action="store_true")
    args = parser.parse_args()
    render(args.campaign, args.output_dir, args.groups, args.stem, args.include_source_controls)


if __name__ == "__main__":
    main()
