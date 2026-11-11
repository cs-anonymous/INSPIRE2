"""Summarize stage-1 decoder masking runs from Trainer log history."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = ("attribute_random", "timestep_random", "no_mask")


def load_curve(path: Path):
    history = json.loads(path.read_text(encoding="utf-8"))
    train = defaultdict(list)
    valid = {}
    for record in history:
        epoch = record.get("epoch")
        if epoch is None:
            continue
        epoch_index = int(math.ceil(float(epoch) - 1e-9))
        if "loss" in record:
            train[epoch_index].append(float(record["loss"]))
        if "eval_loss" in record:
            valid[epoch_index] = float(record["eval_loss"])
    return {
        epoch: {
            "train_loss": sum(train[epoch]) / len(train[epoch]) if train[epoch] else None,
            "valid_loss": valid.get(epoch),
        }
        for epoch in sorted(set(train) | set(valid))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("models/dinr_stage1_mask_ablation_heldout12"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/dinr_stage1_mask_ablation"))
    args = parser.parse_args()

    curves = {}
    for run in RUNS:
        path = args.root / run / "trainer_log_history.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed run log: {path}")
        curves[run] = load_curve(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "losses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "epoch", "train_loss_step_mean", "valid_loss"])
        for run, curve in curves.items():
            for epoch, values in curve.items():
                writer.writerow([run, epoch, values["train_loss"], values["valid_loss"]])

    colors = {
        "attribute_random": "#2563eb",
        "timestep_random": "#dc2626",
        "no_mask": "#15803d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for run, curve in curves.items():
        epochs = list(curve)
        color = colors[run]
        axes[0].plot(epochs, [curve[e]["train_loss"] for e in epochs],
                     marker="o", linewidth=2, color=color, label=run)
        axes[1].plot(epochs, [curve[e]["valid_loss"] for e in epochs],
                     marker="o", linewidth=2, color=color, label=run)
    axes[0].set_title("Train loss (mean of logged steps per epoch)")
    axes[1].set_title("Held-out validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        axis.set_xticks(range(1, 9))
    fig.tight_layout()
    png_path = args.output_dir / "loss_curves.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "plot": str(png_path)}, indent=2))


if __name__ == "__main__":
    main()
