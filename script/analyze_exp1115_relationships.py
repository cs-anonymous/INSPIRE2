#!/usr/bin/env python3
"""Plot 1115 loss/HR relationships and compare the closest 1112 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def correlations(frame, x, y):
    frame = frame[[x, y]].dropna()
    if len(frame) < 3:
        return {"n": int(len(frame)), "pearson": None, "spearman": None}
    return {"n": int(len(frame)),
            "pearson": float(pearsonr(frame[x], frame[y]).statistic),
            "spearman": float(spearmanr(frame[x], frame[y]).statistic)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("analysis/exp1115_beat/metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/exp1115_beat/relationships"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.metrics)
    data = data[data["status"] == "completed"].copy()
    data["label"] = data["setting"] + " s" + data["seed"].astype(str)
    colors = {name: plt.get_cmap("tab10")(i % 10)
              for i, name in enumerate(sorted(data["setting"].unique()))}

    pairs = {
        "valid_vs_tf_hr": ("best_valid_loss", "tf_pn_hr", "Best valid CE", "Teacher-forced PN HR"),
        "tf_vs_ar_hr": ("tf_pn_hr", "ar_pn_hr", "Teacher-forced PN HR", "Autoregressive PN HR"),
    }
    summary = {"completed_rows": int(len(data)), "pairs": {}}
    for name, (xkey, ykey, xlabel, ylabel) in pairs.items():
        fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
        for setting, group in data.groupby("setting", sort=True):
            ax.scatter(group[xkey], group[ykey], s=42, alpha=.8, color=colors[setting],
                       label=setting)
            if len(group) >= 2:
                ax.scatter(group[xkey].mean(), group[ykey].mean(), marker="X", s=85,
                           color=colors[setting], edgecolor="black", linewidth=.5)
        corr = correlations(data, xkey, ykey)
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=f"exp1115 completed training seeds: {xlabel} vs {ylabel}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.text(.02, .98, f"n={corr['n']}  Pearson={corr['pearson']:.3f}  Spearman={corr['spearman']:.3f}",
                transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        fig.savefig(args.output_dir / f"{name}.png", dpi=180)
        fig.savefig(args.output_dir / f"{name}.pdf")
        plt.close(fig)
        setting_means = data.groupby("setting")[[xkey, ykey]].mean(numeric_only=True)
        main = data[~data["setting"].isin(["C2", "C3", "D4"])]
        same_target = data[data["setting"].isin(["A1", "A3", "B1", "D3"])]
        summary["pairs"][name] = {"all_seed_rows": corr,
                                   "setting_means": correlations(setting_means.reset_index(), xkey, ykey),
                                   "separate_head_main_settings": correlations(main, xkey, ykey),
                                   # These four settings share absolute-log targets and
                                   # separate duration heads, so CE is comparable here.
                                   "same_absolute_log_separate_head": correlations(
                                       same_target, xkey, ykey)}

    # Same HR definition and same 19-work/3-draw protocol. The 1115 beat
    # event model is not a one-variable reproduction of 1112, so label this
    # comparison as a protocol-level baseline rather than an ablation.
    legacy = []
    legacy_root = Path("../results/exp1112_sampling3")
    for path in sorted(legacy_root.glob("*/hr_mean_std.json")):
        item = json.loads(path.read_text())
        if "hr" not in item or "pn_hr" not in item["hr"]:
            continue
        legacy.append({"setting": path.parent.name,
                       "ar_pn_hr": item["hr"]["pn_hr"]["mean"],
                       "ar_pn_hr_std": item["hr"]["pn_hr"]["std"]})
    pd.DataFrame(legacy).to_csv(args.output_dir / "exp1112_hr.csv", index=False)
    current = data.groupby("setting")["ar_pn_hr"].agg(["mean", "std", "count"]).reset_index()
    current.rename(columns={"mean": "exp1115_ar_pn_hr_mean", "std": "exp1115_ar_pn_hr_std",
                            "count": "exp1115_training_seeds"}, inplace=True)
    comparable = current[current["setting"].isin(["A1", "B1", "D3"])].copy()
    baseline = pd.DataFrame([{"setting": "1112 separate/timestep/abs5000",
                              "exp1112_ar_pn_hr_mean": next((r["ar_pn_hr"] for r in legacy
                                                               if r["setting"] == "separate_timestep_abs5000"), np.nan),
                              "exp1112_ar_pn_hr_std": next((r["ar_pn_hr_std"] for r in legacy
                                                             if r["setting"] == "separate_timestep_abs5000"), np.nan)}])
    baseline.to_csv(args.output_dir / "exp1112_comparison.csv", index=False)
    # The 1112 reference is a note-only model, so this figure is a protocol
    # comparison, not a causal ablation of adding BEAT events.
    plot = pd.concat([
        current.assign(protocol="exp1115 beat-aware"),
        pd.DataFrame([{"setting": "1112 separate/timestep/abs5000",
                       "exp1115_ar_pn_hr_mean": baseline.iloc[0]["exp1112_ar_pn_hr_mean"],
                       "exp1115_ar_pn_hr_std": baseline.iloc[0]["exp1112_ar_pn_hr_std"],
                       "exp1115_training_seeds": 0,
                       "protocol": "exp1112 note-only"}])
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(plot))
    colors_plot = ["#3977a8" if p == "exp1115 beat-aware" else "#d47b36" for p in plot["protocol"]]
    # matplotlib does not accept a list of RGBA values for ``ecolor`` in
    # errorbar; draw each bar explicitly so the reference is visibly distinct.
    for idx, color in enumerate(colors_plot):
        ax.errorbar([x[idx]], [plot.iloc[idx]["exp1115_ar_pn_hr_mean"]],
                    yerr=[plot.iloc[idx]["exp1115_ar_pn_hr_std"]], fmt="none",
                    ecolor=color, capsize=3, alpha=.8)
    ax.scatter(x, plot["exp1115_ar_pn_hr_mean"], c=colors_plot, s=48)
    ax.axhline(baseline.iloc[0]["exp1112_ar_pn_hr_mean"], color="#d47b36", linestyle="--", linewidth=1)
    ax.set_xticks(x, plot["setting"], rotation=45, ha="right")
    ax.set_ylabel("Autoregressive PN HR (lower is better)")
    ax.set_title("exp1115 beat-aware results vs exp1112 note-only reference")
    ax.grid(axis="y", alpha=.25)
    fig.savefig(args.output_dir / "exp1112_vs_exp1115_ar_hr.png", dpi=180)
    fig.savefig(args.output_dir / "exp1112_vs_exp1115_ar_hr.pdf")
    plt.close(fig)
    summary["legacy_1112"] = legacy
    summary["comparison_note"] = "1115 beat-aware event topology is not a strict one-variable reproduction of 1112 note-only DINR; compare AR PN HR as a protocol-level reference."
    (args.output_dir / "relationships.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["pairs"], indent=2))


if __name__ == "__main__":
    main()
