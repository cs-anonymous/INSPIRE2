#!/usr/bin/env python3
"""Summarize the INSPIRE2 controlled overfitting runs and plot valid-loss deltas."""

import json
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Absolute baseline": "dinr_absolute_ce_heldout12_nocoord_6ep",
    "Absolute + mask": "overfit_factorial_20261030/decoder_mask",
    "Absolute + batch64": "overfit_factorial_20261030/batch64_only",
    "Absolute + cosine/warmup": "overfit_factorial_20261030/cosine_warmup_only",
    "Absolute + batch64/cosine": "overfit_factorial_20261030/outer_optimizer",
    "Absolute + batch64/cosine/mask": "overfit_factorial_20261030/batch64_cosine_warmup_mask",
    "Absolute, random init": "overfit_factorial_20261030/random_init",
    "Relative-log baseline": "overfit_factorial_20261030/relative_target_base",
    "Relative-log + mask": "overfit_factorial_20261030/relative_target_mask",
    "Relative-log + outer protocol": "overfit_factorial_20261030/relative_target_outer_protocol",
}
PANELS = {
    "No decoder mask": [
        "Absolute baseline", "Relative-log baseline", "Absolute + batch64",
        "Absolute + cosine/warmup", "Absolute + batch64/cosine", "Absolute, random init",
    ],
    "With decoder mask": [
        "Absolute + mask", "Absolute + batch64/cosine/mask",
        "Relative-log + mask", "Relative-log + outer protocol",
    ],
}


def read_run(path):
    candidates = list(path.glob("checkpoint-*/trainer_state.json"))
    history = []
    if candidates:
        checkpoint = max(candidates, key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))
        history = json.loads(checkpoint.read_text(encoding="utf-8")).get("log_history", [])
    log = path / "train.log"
    if log.exists():
        log_history = []
        for line in log.open(encoding="utf-8", errors="ignore"):
            match = re.search(r"'eval_loss': ([0-9.eE+-]+).*?'epoch': ([0-9.eE+-]+)", line)
            if match:
                log_history.append({"eval_loss": float(match.group(1)), "epoch": float(match.group(2))})
            match = re.search(r"\{'loss': ([0-9.eE+-]+).*?'epoch': ([0-9.eE+-]+)\}", line)
            if match:
                log_history.append({"loss": float(match.group(1)), "epoch": float(match.group(2))})
        if log_history:
            history = log_history
    if not history:
        return {"eval": {}, "train": {}}

    eval_by_epoch = {}
    train_by_epoch = {epoch: [] for epoch in range(1, 7)}
    for row in history:
        epoch = float(row.get("epoch", 0.0))
        if "eval_loss" in row:
            eval_by_epoch[int(round(epoch))] = float(row["eval_loss"])
        elif "loss" in row:
            bucket = math.ceil(epoch - 1e-12)
            if 0 < epoch <= 6 and 1 <= bucket <= 6:
                train_by_epoch[bucket].append(float(row["loss"]))

    return {
        "eval": eval_by_epoch,
        "train": {
            epoch: statistics.mean(values) if values else None
            for epoch, values in train_by_epoch.items()
        },
    }


def main():
    runs = {name: read_run(ROOT / "models" / relative) for name, relative in RUNS.items()}
    summary = {}
    for name, values in runs.items():
        epochs = sorted(values["eval"])
        evals = [values["eval"][epoch] for epoch in epochs]
        best = min(values["eval"].items(), key=lambda item: item[1]) if values["eval"] else (None, None)
        summary[name] = {
            "valid_loss_by_epoch": {str(epoch): values["eval"][epoch] for epoch in epochs},
            "train_loss_logged_mean_by_epoch_approx": {
                str(epoch): values["train"][epoch] for epoch in values["train"]
            },
            "best_valid_epoch": best[0],
            "best_valid_loss": best[1],
        }

    out = ROOT / "models" / "overfit_factorial_20261030"
    (out / "loss_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, (title, names) in zip(axes, PANELS.items()):
        for name in names:
            values = runs[name]["eval"]
            if not values:
                continue
            epochs = sorted(values)
            baseline = values[epochs[0]]
            axis.plot(epochs, [values[epoch] - baseline for epoch in epochs], marker="o", label=name)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=title, xlabel="Epoch", xticks=range(1, 7))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Valid loss change from epoch 1")
    fig.suptitle("INSPIRE2 DINR controlled experiments (within-run loss change)")
    fig.tight_layout()
    fig.savefig(out / "valid_loss_delta.png", dpi=180)

    selected = [
        "Absolute baseline", "Absolute + mask", "Absolute + cosine/warmup",
        "Relative-log baseline", "Relative-log + mask", "Relative-log + outer protocol",
    ]
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, name in zip(axes2.flat, selected):
        values = runs[name]
        epochs = range(1, 7)
        train = [values["train"].get(epoch) for epoch in epochs]
        valid = [values["eval"].get(epoch) for epoch in epochs]
        axis.plot(list(epochs), train, marker="o", label="train log mean")
        axis.plot(list(epochs), valid, marker="o", label="valid")
        axis.set_title(name, fontsize=9)
        axis.set_xticks(list(epochs))
        axis.grid(alpha=0.3)
    axes2[0, 0].set_ylabel("Loss")
    axes2[1, 0].set_ylabel("Loss")
    axes2[1, 1].set_xlabel("Epoch")
    axes2[1, 2].set_xlabel("Epoch")
    axes2[0, 0].legend(fontsize=8)
    fig2.suptitle("INSPIRE2 train/valid loss logs (train is step-log mean approximation)")
    fig2.tight_layout()
    fig2.savefig(out / "train_valid_curves.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
