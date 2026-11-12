#!/usr/bin/env python3
"""Refresh five groups of raw train/validation curves from the live campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {
    "PT": "pt",
    "DINR attribute": "dinr_attribute",
    "DINR timestep": "dinr_timestep",
    "DINR none": "dinr_none",
    "DINR timestep 1+pitch": "dinr_timestep_1_pitch",
}


def history(job):
    paths = list(Path(job["run_dir"]).rglob("loss_history.jsonl"))
    if len(paths) > 1:
        raise ValueError(f"Multiple loss histories for {job['id']}: {paths}")
    if not paths:
        return []
    lines = paths[0].read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # The training process may be appending the final line right now.
    return rows


def write_csv(path, rows, fields):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def draw(axes, jobs, histories, group):
    colors = plt.get_cmap("tab10")
    observed_epochs = [
        float(row["epoch"])
        for job in jobs for row in histories[job["id"]]
        if "epoch" in row
    ]
    max_epoch = max(8, int(max(observed_epochs, default=8) + 0.999999))
    tick_step = 1 if max_epoch <= 8 else 2
    for index, job in enumerate(jobs):
        rows = histories[job["id"]]
        for axis, key in zip(axes, ("loss", "eval_loss")):
            points = [row for row in rows if key in row]
            if points:
                axis.plot([r["epoch"] for r in points], [r[key] for r in points],
                          color=colors(index), linewidth=0.85 if key == "loss" else 1.5,
                          marker=None if key == "loss" else "o", markersize=3,
                          label=job["label"])
                if key == "eval_loss":
                    best = min(points, key=lambda r: r[key])
                    axis.scatter([best["epoch"]], [best[key]], marker="*", s=105,
                                 color=colors(index), edgecolors="black", linewidths=0.5, zorder=4)
    for axis, split in zip(axes, ("Train: every optimizer step", "Valid: twice per epoch")):
        axis.set_title(f"{group} — {split}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Token CE" if group == "PT" else "DINR attribute loss")
        axis.set_xlim(0, max_epoch)
        axis.set_xticks(range(0, max_epoch + 1, tick_step))
        axis.grid(alpha=0.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8, frameon=False)
        else:
            axis.text(0.5, 0.5, "Queued — no measurements yet", ha="center", va="center", transform=axis.transAxes, color="#777777")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".refresh.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        render(args)


def render(args):
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    # Section 7's five-by-two figure is reserved for its original 12 jobs.
    # Follow-up experiments receive separate figures in their own sections.
    if any(job["id"] == "pt_scratch_full8" for job in manifest["jobs"]):
        args.campaign = Path(__file__).resolve().parents[2] / "results/halfepoch_earlystop_20261112"
        manifest = json.loads((args.campaign / "manifest.json").read_text())
    jobs = manifest["jobs"]
    histories = {j["id"]: history(j) for j in jobs}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    losses, summary = [], []
    for job in jobs:
        events = histories[job["id"]]
        for event in events:
            for key, split in (("loss", "train"), ("eval_loss", "valid")):
                if key in event:
                    losses.append(dict(run=job["id"], group=job["group"], label=job["label"],
                                       split=split, step=event["step"], epoch=event["epoch"], loss=event[key]))
        valid = [r for r in events if "eval_loss" in r]
        best = min(valid, key=lambda r: r["eval_loss"]) if valid else {}
        hr = job.get("hr", {}).get("hr", {})
        summary.append(dict(run=job["id"], group=job["group"], label=job["label"], status=job["status"],
                            best_valid_epoch=best.get("epoch", ""), best_valid_step=best.get("step", ""),
                            best_valid_loss=best.get("eval_loss", ""),
                            final_epoch=job.get("final_epoch", events[-1]["epoch"] if events else ""),
                            validation_count=len(valid), checkpoint=job.get("checkpoint", ""),
                            ioi_hr=hr.get("ioi_hr", ""), duration_hr=hr.get("duration_hr", ""),
                            velocity_hr=hr.get("velocity_hr", ""), pn_hr=hr.get("pn_hr", "")))
    loss_fields = ["run", "group", "label", "split", "step", "epoch", "loss"]
    write_csv(output / "all_losses.csv", losses, loss_fields)
    write_csv(output / "all_experiments.csv", summary, list(summary[0]))
    # One comparison figure: columns are experiment groups, rows are train/valid.
    # Keep each group's own loss scale visible within the combined figure.
    overview, overview_axes = plt.subplots(2, 5, figsize=(28, 10), squeeze=False,
                                           constrained_layout=True)
    for index, (group, slug) in enumerate(GROUPS.items()):
        grouped = [j for j in jobs if j["group"] == group]
        axes = overview_axes[:, index]
        draw(axes, grouped, histories, group)
        write_csv(output / f"{slug}_losses.csv", [r for r in losses if r["group"] == group], loss_fields)
    overview.suptitle("Half-epoch validation rerun: five groups, train and validation", fontsize=18)
    for extension in ("png", "pdf"):
        overview.savefig(output / f"all_groups.{extension}", dpi=160)
    plt.close(overview)
    lines = ["# Half-epoch validation rerun", "",
             "Train loss is the raw mean for each optimizer step (including accumulated microbatches), without smoothing. "
             "Valid loss is evaluated on the complete fixed held-out split at the first optimizer boundary at/after half an epoch, and at epoch end. "
             "Strict improvement over the historical minimum resets patience; four consecutive non-improvements stop training"
             + (", except PT scratch full 8ep, which disables early stopping and runs all eight epochs. "
                if manifest.get("protocol", {}).get("scratch_full8_exception") else ". ") +
             "Only the lowest-valid checkpoint is saved. Stars identify the best observed validation.", "",
             "PT uses token CE and DINR uses attribute loss; their absolute values differ in meaning. "
             "All curves use recorded measurements. Queued panels are empty until actual measurements arrive.", "",
             "| Run | Status | Best epoch | Best valid | Last epoch | PN HR |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        def fmt(value):
            return f"{value:.6f}" if isinstance(value, (float, int)) else value or "—"
        lines.append(f"| {row['run']} | {row['status']} | {fmt(row['best_valid_epoch'])} | {fmt(row['best_valid_loss'])} | {fmt(row['final_epoch'])} | {fmt(row['pn_hr'])} |")
    lines.extend(["", "## Combined figure", "", "The five groups are columns; the first row is train loss and the second row is validation loss.", "", "![Five groups, two rows](all_groups.png)", "", "[PDF](all_groups.pdf)"])
    for group, slug in GROUPS.items():
        lines.extend(["", f"## {group} data", "", f"[Raw loss CSV]({slug}_losses.csv)"])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
