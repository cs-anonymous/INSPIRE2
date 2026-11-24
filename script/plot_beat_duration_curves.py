#!/usr/bin/env python3
"""Plot score and human-performance beat-duration curves by work.

Each panel is one work, in natural-log milliseconds (the training coordinate).
Score beat durations are shown as a dashed black
curve; each aligned performance is a translucent colored curve.  Alternating
measure bands make the score measure structure visible without changing the
beat index used on the x-axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_curves(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    grouped: dict[str, list[dict]] = {}
    for row in payload["performances"]:
        grouped.setdefault(row["work"], []).append(row)

    result = []
    for work, rows in sorted(grouped.items()):
        first_start, first_end = rows[0]["span"]
        first_events = payload["events"][first_start:first_end]
        beat_mask = first_events[:, 0].eq(0)
        score = first_events[beat_mask, 2].numpy().astype(float)
        measure_start = first_events[beat_mask, 5].numpy().astype(int)
        performances = []
        for row in rows:
            start, end = row["span"]
            events = payload["events"][start:end]
            if not torch.equal(events, first_events):
                raise ValueError(f"{work}: performance event topology differs from score topology")
            targets = payload["targets"][start:end]
            valid = payload.get("target_valid")
            valid_duration = (valid[start:end, 0] if valid is not None
                              else torch.ones(len(events), dtype=torch.bool))
            values = np.full(len(score), np.nan, dtype=float)
            values[valid_duration[beat_mask].numpy().astype(bool)] = (
                targets[beat_mask, 0].numpy().astype(float)[valid_duration[beat_mask].numpy().astype(bool)]
            )
            performances.append({
                "id": row["performance"],
                "duration_ms": values.tolist(),
            })
        result.append({
            "work": work,
            "performance_count": len(performances),
            "beat_count": len(score),
            "score_duration_ms": score.tolist(),
            "measure_start": measure_start.tolist(),
            "performances": performances,
        })
    return result


def short_work_name(work: str) -> str:
    parts = work.split("/")
    composer = parts[0].split(",")[0].replace("_", " ")
    piece = parts[1].replace("_", " ") if len(parts) > 1 else parts[-1]
    movement = parts[-1].split(".", 1)[0] if len(parts) > 2 else ""
    prefix = f" / mvt {movement}" if movement.isdigit() else ""
    name = f"{composer}{prefix}: {piece}"
    return textwrap.shorten(name, width=48, placeholder="...")


def select_works(curves: list[dict], count: int) -> list[dict]:
    """Cover composers, then fill with the most multi-performance works."""
    if len(curves) < count:
        raise ValueError(f"Requested {count} works but found {len(curves)}")
    ranked = sorted(curves, key=lambda item: (-item["performance_count"], item["work"]))
    selected = {}
    composers = set()
    for item in ranked:
        composer = item["work"].split("/")[0]
        if composer not in composers and len(selected) < count:
            selected[item["work"]] = item
            composers.add(composer)
    for item in ranked:
        if len(selected) == count:
            break
        selected.setdefault(item["work"], item)
    return [selected[work] for work in sorted(selected)]


def measure_spans(measure_start: np.ndarray, length: int) -> list[tuple[int, int, int]]:
    starts = np.flatnonzero(measure_start == 1).tolist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts = sorted(set(starts))
    return [(start, starts[i + 1] if i + 1 < len(starts) else length, i)
            for i, start in enumerate(starts) if start < length]


def plot_matrix(curves: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(curves) != 25:
        raise ValueError("The combined matrix must contain exactly 25 works")
    fig, axes = plt.subplots(5, 5, figsize=(22, 17.5), squeeze=False)
    measure_colors = ("#dbeafe", "#ffedd5")
    perf_colors = plt.get_cmap("tab20")
    for index, ax in enumerate(axes.flat):
        if index >= len(curves):
            ax.axis("off")
            continue
        item = curves[index]
        score = np.log(np.maximum(np.asarray(item["score_duration_ms"], dtype=float), 1.0))
        measures = np.asarray(item["measure_start"], dtype=int)
        x = np.arange(len(score))
        for left, right, measure_index in measure_spans(measures, len(score)):
            ax.axvspan(left - 0.5, right - 0.5,
                       color=measure_colors[measure_index % 2], alpha=0.5,
                       linewidth=0, zorder=0)
        ax.plot(x, score, color="black", linestyle="--", linewidth=1.25,
                label="score", zorder=3)
        performances = item["performances"]
        for perf_index, perf in enumerate(performances):
            values = np.log(np.maximum(np.asarray(perf["duration_ms"], dtype=float), 1.0))
            color = perf_colors(perf_index % 20)
            ax.plot(x, values, color=color, linewidth=0.75, alpha=0.78,
                    label=perf["id"], zorder=2)
        ax.set_title(f"{item['panel']}: {short_work_name(item['work'])}\n"
                     f"{item['performance_count']} perf., {item['beat_count']} beats",
                     fontsize=8)
        ax.set_xlabel("Score beat index", fontsize=8)
        ax.set_ylabel(r"$\ln(D_{\mathrm{beat}}\,/\,\mathrm{ms})$", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlim(-0.5, max(len(score) - 0.5, 0.5))
        ax.set_facecolor("#f5faff" if item["split"] == "test" else "#fffcf5")
        ax.grid(alpha=0.18, linewidth=0.5)
    fig.suptitle("Score and human-performance log beat-duration curves (natural log, milliseconds)\n"
                 "Rows 1–3: TEST (15 works)     |     Rows 4–5: VALID (10 works)",
                 fontsize=15, y=0.992)
    fig.legend(handles=[Line2D([], [], color="black", linestyle="--", label="Score"),
                        Line2D([], [], color="#2563eb", label="Each color = one performance"),
                        Patch(facecolor=measure_colors[0], label="Measure color A"),
                        Patch(facecolor=measure_colors[1], label="Measure color B (alternating)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.956), ncol=4,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.932), h_pad=2.6, w_pad=1.8)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=170)
    temporary.replace(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=ROOT.parent / "data/ASAP_processed_dinr_beat_dev_e512_v2")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "analysis/beat_duration_curves")
    args = parser.parse_args()
    combined = []
    for split, count in (("test", 15), ("valid", 10)):
        sidecar = args.data_root / f"dinr_beat_{split}_e512_v2.pt"
        curves = select_works(load_curves(sidecar), count)
        for index, item in enumerate(curves, 1):
            item.update(split=split, panel=f"{'T' if split == 'test' else 'V'}{index:02d}")
        combined.extend(curves)
        print(f"{split}: selected {len(curves)} works, "
              f"{sum(item['performance_count'] for item in curves)} performances", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Use JSON null for unsupported intervals; do not serialize nonstandard NaN.
    for item in combined:
        for perf in item["performances"]:
            perf["duration_ms"] = [value if np.isfinite(value) else None
                                    for value in perf["duration_ms"]]
    (args.output_dir / "test15_valid10_curves.json").write_text(
        json.dumps({"selection": "Cover composers, then prioritize performance count; sort works within each split.",
                    "plot_coordinate": "ln(max(beat_duration_ms, 1)); continuous values before quantization and without [3, 8] clipping",
                    "stored_duration_unit": "milliseconds; transform using plot_coordinate to reproduce the figure",
                    "curves": combined}, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    plot_matrix(combined, args.output_dir / "test15_valid10_beat_duration_5x5.png")
    print(args.output_dir / "test15_valid10_beat_duration_5x5.png", flush=True)


if __name__ == "__main__":
    main()
