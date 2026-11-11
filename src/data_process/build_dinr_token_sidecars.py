"""Precompute DINR note tokens and windows once for fast training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data_process.dinr_dataset import (
    ASAP_SIDECAR_NAME,
    DINRTokenSpec,
    _performance_tokens,
    _sidecar_paths,
)


def canonical_windows(total: int, block_notes: int, overlap_ratio: float, min_notes: int = 8):
    total = int(total)
    if total < min_notes:
        return []
    if total <= block_notes:
        return [(0, total)]
    stride = max(1, int(block_notes * (1.0 - overlap_ratio)))
    windows = []
    start = 0
    while start + block_notes <= total:
        windows.append((start, start + block_notes))
        start += stride
    if windows[-1][1] != total and total - start >= min_notes:
        windows.append((total - block_notes, total))
    return list(dict.fromkeys(windows))


def load_valid_works(summary_path: Path):
    if summary_path is None:
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {}
    prefix = "data/ASAP_processed/"
    for row in payload.get("work_summaries", []):
        if not row.get("selected") or not row.get("valid_window"):
            continue
        path = str(row["path"])
        if prefix in path:
            rel = path.split(prefix, 1)[1]
        else:
            rel = path
        rel = rel.rsplit("/score_ASAP_refined.json", 1)[0]
        result[rel] = True
    return result


def load_heldout_work_manifest(manifest_path: Path):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = payload.get("valid_work_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{manifest_path} must contain a non-empty valid_work_paths list")
    result = set()
    for row in paths:
        work_path = row.get("work_path") if isinstance(row, dict) else row
        if not isinstance(work_path, str) or not work_path:
            raise ValueError(f"Invalid work path in {manifest_path}: {row!r}")
        result.add(work_path)
    return result


def build(root: Path, output: Path, split: str, block_notes: int, overlap_ratio: float,
          valid_windows=None, window_split: str = "train", heldout_works=None,
          valid_work_manifest: Path | None = None):
    score_parts = []
    target_parts = []
    zero_parts = []
    samples = []
    offset = 0
    performance_count = 0
    windowed_performance_count = 0
    windowed_works = set()
    canonical_window_count = 0
    excluded_window_count = 0
    excluded_heldout_work_window_count = 0
    for sidecar in _sidecar_paths(root):
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        score = payload.get("score") or {}
        for performance in payload.get("performances", []):
            if str(performance.get("split", "")) != split:
                continue
            score_tokens, target_tokens, zero_mask = _performance_tokens(
                score, performance, include_pedal=True
            )
            n = len(score_tokens)
            if n == 0:
                continue
            score_parts.append(torch.tensor(score_tokens, dtype=torch.int16))
            target_parts.append(torch.tensor(target_tokens, dtype=torch.int16))
            zero_parts.append(torch.tensor(zero_mask, dtype=torch.bool))
            windows = canonical_windows(n, block_notes, overlap_ratio)
            work_rel = sidecar.parent.relative_to(root).as_posix()
            canonical_window_count += len(windows)
            if heldout_works is not None:
                is_valid_work = work_rel in heldout_works
                if window_split == "valid":
                    windows = windows if is_valid_work else []
                elif is_valid_work:
                    excluded_heldout_work_window_count += len(windows)
                    windows = []
            else:
                valid_window = windows[-1] if windows and work_rel in (valid_windows or {}) else None
                if window_split == "valid":
                    windows = [valid_window] if valid_window in windows else []
                elif window_split == "train" and valid_window in windows:
                    original_count = len(windows)
                    windows = [
                        window for window in windows
                        if window != valid_window and not (
                            window[0] < valid_window[1] and valid_window[0] < window[1]
                        )
                    ]
                    excluded_window_count += original_count - len(windows) - 1
            for start, end in windows:
                samples.append((offset + start, offset + end))
            if windows:
                windowed_performance_count += 1
                windowed_works.add(work_rel)
            offset += n
            performance_count += 1
        del payload
    if not score_parts:
        raise ValueError(f"No performances found for split={split}")
    result = {
        "format": "dinr_absolute_tokens_v1",
        "split": window_split if split == "train" else split,
        "source_split": split,
        "window_split": window_split,
        "block_notes": int(block_notes),
        "overlap_ratio": float(overlap_ratio),
        "vocab_size": DINRTokenSpec.vocab_size,
        "source_name": ASAP_SIDECAR_NAME,
        "performances": performance_count,
        "windowed_performances": windowed_performance_count,
        "windowed_works": len(windowed_works),
        "canonical_windows_before_split": canonical_window_count,
        "excluded_overlapping_windows": excluded_window_count,
        "valid_work_selection": (
            "outer fixed-window summary work set; last canonical window per selected work"
            if valid_windows else "not applicable"
        ),
        "score_tokens": torch.cat(score_parts),
        "target_tokens": torch.cat(target_parts),
        "score_zero_ioi_mask": torch.cat(zero_parts),
        "samples": samples,
    }
    if heldout_works is not None:
        result["valid_work_selection"] = "fixed held-out works; all canonical windows"
        result["valid_work_manifest"] = str(valid_work_manifest) if valid_work_manifest else None
        result["valid_work_paths"] = sorted(heldout_works)
        result["excluded_heldout_work_windows"] = excluded_heldout_work_window_count
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"wrote {output} performances={performance_count} notes={offset} windows={len(samples)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, default=Path("../data/ASAP_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("../data/ASAP_processed_dinr_tokens"))
    parser.add_argument("--fixed-window-summary", type=Path, default=None)
    parser.add_argument("--valid-work-manifest", type=Path, default=None,
                        help="Fixed work-level validation split; all windows from these works are held out.")
    parser.add_argument("--block-notes", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    args = parser.parse_args()
    valid_windows = load_valid_works(args.fixed_window_summary) if args.fixed_window_summary else {}
    heldout_works = (
        load_heldout_work_manifest(args.valid_work_manifest)
        if args.valid_work_manifest else None
    )
    for split in ("train", "test"):
        if split == "train" and (args.fixed_window_summary or args.valid_work_manifest):
            variants = ("train", "valid")
        else:
            variants = ("train",)
        for window_split in variants:
            suffix = f"_{window_split}" if split == "train" else ""
            output = args.output_dir / f"dinr_absolute_{split}{suffix}_b{args.block_notes}_o{int(args.overlap_ratio * 100)}.pt"
            build(
                args.sidecar_root, output, split, args.block_notes, args.overlap_ratio,
                valid_windows=valid_windows, window_split=window_split,
                heldout_works=heldout_works if split == "train" else None,
                valid_work_manifest=args.valid_work_manifest if split == "train" else None,
            )


if __name__ == "__main__":
    main()
