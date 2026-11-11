"""Plot epoch-level train and validation losses for DINR-Absolute runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(mode: str, root: Path):
    state = root / f"models/dinr_absolute_{mode}_validlast/checkpoint-4928/trainer_state.json"
    payload = json.loads(state.read_text(encoding="utf-8"))
    history = payload["log_history"]
    rows = []
    for epoch in range(1, 17):
        train_values = [
            float(item["loss"])
            for item in history
            if "loss" in item and epoch - 1 < float(item.get("epoch", 0.0)) <= epoch
        ]
        valid_values = [
            float(item["eval_loss"])
            for item in history
            if "eval_loss" in item and abs(float(item.get("epoch", -1.0)) - epoch) < 1e-6
        ]
        rows.append({
            "epoch": epoch,
            "train_loss": sum(train_values) / len(train_values),
            "valid_loss": valid_values[-1],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("analysis/dinr_absolute_train_valid_loss.png"))
    args = parser.parse_args()
    curves = {mode: load_curve(mode, args.root) for mode in ("ce", "bce", "none")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        for mode, rows in curves.items():
            for row in rows:
                writer.writerow({"mode": mode, **row})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    colors = {"ce": "#1f77b4", "bce": "#d62728", "none": "#2ca02c"}
    labels = {"ce": "CE pedal", "bce": "BCE pedal", "none": "No pedal"}
    for mode, rows in curves.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o",
                     color=colors[mode], label=labels[mode])
        axes[1].plot(epochs, [row["valid_loss"] for row in rows], marker="o",
                     color=colors[mode], label=labels[mode])
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_xticks(range(1, 17))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("DINR-Absolute: epoch-level loss curves")
    fig.savefig(args.output, dpi=180)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
