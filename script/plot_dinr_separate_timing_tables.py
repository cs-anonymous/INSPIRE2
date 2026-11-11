#!/usr/bin/env python3
"""Compare shared/separate timing tables and time-target variants."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "models/dinr_stage1_separate_ioi_duration_heldout12_8ep"
OUT = ROOT / "analysis/dinr_separate_ioi_duration_heldout12_8ep"
VARIANTS = {
    "attribute_mask": ("attribute_random", "Attribute mask", "#2878B5"),
    "timestep_mask": ("timestep_random", "Timestep mask", "#2E8B57"),
    "no_mask": ("no_mask", "No mask", "#D9534F"),
}
TIME_RUNS = {
    "absolute_linear": ("Linear absolute", "#7B2CBF"),
    "log_absolute": ("Log absolute", "#F77F00"),
    "log_deviation": ("Log deviation", "#0077B6"),
}
TIME_ROOT = ROOT / "models/dinr_timestep_separate_logtime_heldout12_8ep"
NOPITCH_ROOT = ROOT / "models/dinr_timestep_shared_nopitch_heldout12_8ep"


def main():
    with (ROOT / "analysis/dinr_stage1_mask_ablation/losses.csv").open() as stream:
        baseline = list(csv.DictReader(stream))
    rows = []
    summaries = []
    for variant, (old_run, label, _) in VARIANTS.items():
        run = RUN_ROOT / variant
        history = json.loads((run / "trainer_log_history.json").read_text())
        evals = [entry for entry in history if "eval_loss" in entry]
        if len(evals) != 8 or float(evals[-1]["epoch"]) != 8.0:
            raise ValueError(f"{variant} has not completed 8 epochs")
        for entry in baseline:
            if entry["run"] == old_run:
                rows.append({
                    "table": "shared", "mask": variant, "epoch": int(entry["epoch"]),
                    "train_loss": float(entry["train_loss_step_mean"]),
                    "valid_loss": float(entry["valid_loss"]),
                })
        for entry in evals:
            epoch = int(entry["epoch"])
            logs = [float(item["loss"]) for item in history
                    if "loss" in item and epoch - 1 < float(item["epoch"]) <= epoch]
            rows.append({
                "table": "separate", "mask": variant, "epoch": epoch,
                "train_loss": sum(logs) / len(logs),
                "valid_loss": float(entry["eval_loss"]),
            })
        for table in ("shared", "separate"):
            subset = [row for row in rows if row["mask"] == variant and row["table"] == table]
            best = min(subset, key=lambda row: row["valid_loss"])
            last = max(subset, key=lambda row: row["epoch"])
            summaries.append({
                "mask": variant, "table": table, "best_epoch": best["epoch"],
                "best_valid_loss": best["valid_loss"], "epoch8_valid_loss": last["valid_loss"],
                "epoch8_train_loss": last["train_loss"],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in (("losses.csv", rows), ("summary.csv", summaries)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=content[0].keys())
            writer.writeheader()
            writer.writerows(content)

    time_rows = []
    for run, (label, _) in TIME_RUNS.items():
        history = json.loads((TIME_ROOT / run / "trainer_log_history.json").read_text())
        evals = [entry for entry in history if "eval_loss" in entry]
        if len(evals) != 8 or float(evals[-1]["epoch"]) != 8.0:
            raise ValueError(f"{run} has not completed 8 epochs")
        for entry in evals:
            epoch = int(entry["epoch"])
            logs = [float(item["loss"]) for item in history
                    if "loss" in item and epoch - 1 < float(item["epoch"]) <= epoch]
            time_rows.append({
                "run": run, "label": label, "epoch": epoch,
                "train_loss": sum(logs) / len(logs),
                "valid_loss": float(entry["eval_loss"]),
            })
    with (OUT / "logtime_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "label", "epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        writer.writerows(time_rows)

    nopitch_history = json.loads((NOPITCH_ROOT / "trainer_log_history.json").read_text())
    nopitch_evals = [entry for entry in nopitch_history if "eval_loss" in entry]
    if len(nopitch_evals) != 8 or float(nopitch_evals[-1]["epoch"]) != 8.0:
        raise ValueError("shared pitch-preserving timestep run has not completed 8 epochs")
    nopitch_rows = []
    for entry in nopitch_evals:
        epoch = int(entry["epoch"])
        logs = [float(item["loss"]) for item in nopitch_history
                if "loss" in item and epoch - 1 < float(item["epoch"]) <= epoch]
        nopitch_rows.append({
            "epoch": epoch,
            "train_loss": sum(logs) / len(logs),
            "valid_loss": float(entry["eval_loss"]),
        })
    with (OUT / "timestep_shared_nopitch_losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        writer.writerows(nopitch_rows)

    fig, axes = plt.subplots(
        2, 5, figsize=(25, 8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1.25, 1]},
    )
    for col, (variant, (_, label, color)) in enumerate(VARIANTS.items()):
        for row_index, split in enumerate(("train", "valid")):
            axis = axes[row_index, col]
            for table, style, marker in (("shared", "--", "o"), ("separate", "-", "s")):
                subset = sorted((row for row in rows
                                 if row["mask"] == variant and row["table"] == table),
                                key=lambda row: row["epoch"])
                axis.plot([r["epoch"] for r in subset], [r[f"{split}_loss"] for r in subset],
                          color=color, linestyle=style, marker=marker, linewidth=2,
                          label=f"{table.capitalize()} IOI/duration tables")
            axis.set_title(f"{label}: {'training' if split == 'train' else 'validation'}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss (sum of attribute cross-entropies)")
            axis.set_xticks(range(1, 9))
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=9)
    for row_index, split in enumerate(("train", "valid")):
        axis = axes[row_index, 3]
        for run, (label, color) in TIME_RUNS.items():
            subset = sorted((row for row in time_rows if row["run"] == run),
                            key=lambda row: row["epoch"])
            axis.plot([r["epoch"] for r in subset], [r[f"{split}_loss"] for r in subset],
                      color=color, marker="o", linewidth=2, label=label)
        axis.set_title(f"Timestep + separate: {'training' if split == 'train' else 'validation'}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss (sum of attribute cross-entropies)")
        axis.set_xticks(range(1, 9))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    for row_index, split in enumerate(("train", "valid")):
        axis = axes[row_index, 4]
        axis.plot([r["epoch"] for r in nopitch_rows],
                  [r[f"{split}_loss"] for r in nopitch_rows],
                  color="#6A4C93", marker="o", linewidth=2,
                  label="Shared table; pitch retained")
        axis.set_title(f"Timestep + shared (pitch kept): {'training' if split == 'train' else 'validation'}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss (sum of attribute cross-entropies)")
        axis.set_xticks(range(1, 9))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("DINR: shared vs. separate IOI/duration tables and time-target variants (8 epochs)")
    fig.savefig(OUT / "loss_curves.png", dpi=180)
    plt.close(fig)

    report = [
        "# Independent IOI and duration embedding tables", "",
        "All runs use the fixed 12-work split, 512-note windows, batch size 32, "
        "learning rate 3e-4 with linear decay, seed 42, PT initialization, "
        "8 epochs, no pedal heads, and no coordinate embeddings. "
        "Masking is enabled only during training. "
        "Separate tables each start from the same PT timing weights and train independently. "
        "The encoder and decoder use the same IOI table and the same duration table.", "",
        "Train loss uses the mean of logged interval losses within each epoch, "
        "matching the existing mask-ablation report; intervals may cross an epoch boundary.", "",
        "| Mask | Table | Best epoch | Best valid | Epoch 8 valid | Epoch 8 train |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in summaries:
        report.append(f"| {entry['mask']} | {entry['table']} | {entry['best_epoch']} | "
                      f"{entry['best_valid_loss']:.6f} | {entry['epoch8_valid_loss']:.6f} | "
                      f"{entry['epoch8_train_loss']:.6f} |")
    report.extend(["", f"Checkpoints and training logs: `{RUN_ROOT.relative_to(ROOT)}`.", ""])
    (OUT / "README.md").write_text("\n".join(report))
    print(json.dumps(summaries, indent=2))
    print(OUT / "loss_curves.png")


if __name__ == "__main__":
    main()
