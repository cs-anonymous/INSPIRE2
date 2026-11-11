"""Plot the heldout-12-work DINR CE 6-epoch comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "No coordinate": ROOT / "models/dinr_absolute_ce_heldout12_nocoord_6ep",
    "Coordinate": ROOT / "models/dinr_absolute_ce_heldout12_coord_6ep",
}


def read_curve(run_dir):
    state_path = run_dir / "checkpoint-1932/trainer_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    history = payload["log_history"]
    rows = []
    for epoch in range(1, 7):
        train_logs = [
            float(item["loss"])
            for item in history
            if "loss" in item and epoch - 1 < float(item.get("epoch", 0.0)) <= epoch
        ]
        valid_logs = [
            float(item["eval_loss"])
            for item in history
            if "eval_loss" in item and abs(float(item.get("epoch", -1.0)) - epoch) < 1e-6
        ]
        rows.append({
            "epoch": epoch,
            "train_loss": sum(train_logs) / len(train_logs),
            "valid_loss": valid_logs[-1],
        })
    return rows


def main():
    output_dir = ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = {label: read_curve(path) for label, path in RUNS.items()}
    csv_path = output_dir / "dinr_heldout12_6ep_loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        for label, rows in curves.items():
            for row in rows:
                writer.writerow({"variant": label, **row})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"No coordinate": "#187b67", "Coordinate": "#d66a2c"}
    for label, rows in curves.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o",
                     color=colors[label], label=label)
        axes[1].plot(epochs, [row["valid_loss"] for row in rows], marker="o",
                     color=colors[label], label=label)
        best = min(rows, key=lambda row: row["valid_loss"])
        axes[1].scatter([best["epoch"]], [best["valid_loss"]], color=colors[label],
                        edgecolor="white", linewidth=0.8, zorder=3)
    axes[0].set_title("Train loss")
    axes[1].set_title("Held-out-work validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("DINR-Absolute CE: fixed 12-work split")
    png_path = output_dir / "dinr_heldout12_6ep_loss.png"
    fig.savefig(png_path, dpi=180)
    print(png_path)
    print(csv_path)


if __name__ == "__main__":
    main()
