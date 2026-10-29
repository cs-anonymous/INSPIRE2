"""Greedy DINR-absolute test inference, MIDI reconstruction, and PN W1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_process.dinr_dataset import (  # noqa: E402
    DINRTokenSpec,
    _performance_tokens,
)
from src.model.dinr import DINRAbsoluteModel  # noqa: E402
from src.model.pianoformer import PianoT5GemmaConfig  # noqa: E402
from src.utils.inr_midi import midi_to_note_features, note_features_to_midi  # noqa: E402


def load_state_dict(path):
    path = Path(path)
    if path.is_dir():
        if (path / "model.safetensors").exists():
            path = path / "model.safetensors"
        elif (path / "pytorch_model.bin").exists():
            path = path / "pytorch_model.bin"
        else:
            candidates = sorted(path.glob("*.safetensors"))
            if candidates:
                path = candidates[0]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=False)


def make_model(args, checkpoint):
    config = PianoT5GemmaConfig(
        encoder_layers_num=args.encoder_layers,
        decoder_layers_num=args.decoder_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.key_value_heads,
        head_dim=args.head_dim,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    model = DINRAbsoluteModel(config, pedal_mode=args.pedal_mode)
    state = load_state_dict(checkpoint)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing checkpoint tensors={len(missing)}")
    if unexpected:
        print(f"warning: unexpected checkpoint tensors={len(unexpected)}")
    return model.eval()


def predict_notes(model, score_tokens, score_zero_mask, device, include_pedal):
    score = torch.tensor(score_tokens, dtype=torch.long, device=device).unsqueeze(0)
    n = score.shape[1]
    generated = []
    with torch.no_grad():
        score_embeds = model._sum_note_tokens(score)
        score_mask = torch.ones((1, n), dtype=torch.long, device=device)
        encoder_outputs = model.backbone.encoder(
            inputs_embeds=score_embeds, attention_mask=score_mask
        )
        past_key_values = None
        for index in range(n):
            current = ([DINRTokenSpec.bos_id] * 8) if index == 0 else generated[-1]
            decoder = torch.tensor(current, dtype=torch.long, device=device).view(1, 1, 8)
            decoder_mask = torch.ones((1, index + 1), dtype=torch.long, device=device)
            output = model(
                score_tokens=score,
                decoder_tokens=decoder,
                attention_mask=score_mask,
                decoder_attention_mask=decoder_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=torch.tensor([index], device=device),
            )
            past_key_values = output["past_key_values"]
            logits = output["logits"]
            head = "chord_ioi" if score_zero_mask[index] else "grouped_ioi"
            ioi_class = int(logits[head][0, -1].argmax())
            duration_class = int(logits["duration"][0, -1].argmax())
            velocity_class = int(logits["velocity"][0, -1].argmax())
            if model.pedal_mode == "ce":
                pedal = [int(logits["pedal"][0, -1, j].argmax()) for j in range(4)]
            elif model.pedal_mode == "bce":
                pedal = [int(logits["pedal"][0, -1, j] >= 0.0) for j in range(4)]
            else:
                pedal = [0, 0, 0, 0]
            token = DINRTokenSpec.note_to_tokens(
                score_tokens[index][0] - DINRTokenSpec.pitch_start,
                ioi_class * DINRTokenSpec.timing_bin_ms,
                duration_class * DINRTokenSpec.timing_bin_ms,
                velocity_class,
                pedal,
            )
            if not include_pedal:
                token[4:] = [DINRTokenSpec.pad_id] * 4
            generated.append(token)
    return generated


def expand_grouped_ioi(score_raw, grouped_or_chord, zero_mask):
    """Convert grouped interval totals plus chord gaps to note IOIs."""
    score_ioi = np.asarray(score_raw, dtype=np.float64)[:, 0]
    grouped_or_chord = np.asarray(grouped_or_chord, dtype=np.float64)
    out = np.zeros(len(score_ioi), dtype=np.float64)
    previous_positive = None
    for index, score_value in enumerate(score_ioi):
        if score_value <= 0.0:
            out[index] = max(float(grouped_or_chord[index]), 0.0) if index > 0 else 0.0
            continue
        start = 1 if previous_positive is None else previous_positive + 1
        internal = float(out[start:index].sum()) if index > start else 0.0
        out[index] = max(float(grouped_or_chord[index]) - internal, 0.0)
        previous_positive = index
    return out


def tokens_to_raw(score_raw, score_tokens, pred_tokens, score_zero_mask, pedal_mode):
    pred_tokens = np.asarray(pred_tokens, dtype=np.int64)
    score_raw = np.asarray(score_raw, dtype=np.float64)
    grouped = (pred_tokens[:, 1] - DINRTokenSpec.timing_start) * DINRTokenSpec.timing_bin_ms
    duration = (pred_tokens[:, 3] - DINRTokenSpec.timing_start) * DINRTokenSpec.timing_bin_ms
    velocity = pred_tokens[:, 2] - DINRTokenSpec.velocity_start
    ioi = expand_grouped_ioi(score_raw, grouped, score_zero_mask)
    if pedal_mode == "none":
        pedals = np.zeros((len(pred_tokens), 4), dtype=np.float64)
    else:
        pedals = (pred_tokens[:, 4:8] - DINRTokenSpec.pedal_start) * 127.0
    return np.column_stack([ioi, duration, velocity, pedals])


def raw_to_midi(pitch, raw, path):
    midi = note_features_to_midi(
        pitch, raw.tolist(), target_ticks_per_beat=500,
        target_tempo=120, normalized=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


def midi_note_arrays(path):
    midi = MidiFile(str(path))
    features = midi_to_note_features(
        midi, max_time_ms=10000.0, normalize=False, force_monotonic_starts=False
    )
    rows = np.asarray(features["continuous"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"Unexpected MIDI feature shape in {path}: {rows.shape}")
    return rows


def pn_metrics(prediction_path, reference_paths, include_pedal):
    pred = midi_note_arrays(prediction_path)
    refs = [midi_note_arrays(path) for path in reference_paths]
    feature_names = [("ioi", 0), ("duration", 1), ("velocity", 2)]
    pedal_names = [(f"pedal_{index}", 3 + index) for index in range(4)]
    if include_pedal:
        feature_names.extend(pedal_names)
    usable = min([len(pred)] + [len(item) for item in refs])
    values = {}
    for name, column in feature_names:
        values[f"{name}_wass"] = float(np.mean([
            wasserstein_distance([pred[index, column]], [item[index, column] for item in refs])
            for index in range(usable)
        ])) if usable else float("nan")
    if include_pedal:
        values["pedal_wass"] = float(np.mean([values.pop(f"pedal_{index}_wass") for index in range(4)]))
    categories = [values["ioi_wass"], values["duration_wass"], values["velocity_wass"]]
    if include_pedal:
        categories.append(values["pedal_wass"])
    values["pn_wass_mean"] = float(np.mean(categories))
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sidecar-root", type=Path, default=Path("../data/ASAP_processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pedal-mode", choices=["ce", "bce", "none"], default="ce")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--encoder-layers", type=int, default=10)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--limit-works", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = make_model(args, args.checkpoint).to(device)
    sidecars = sorted(args.sidecar_root.resolve().rglob("score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt"))
    rows = []
    include_pedal = args.pedal_mode != "none"
    for work_index, sidecar in enumerate(sidecars):
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        performances = [p for p in payload.get("performances", []) if p.get("split") == "test"]
        if not performances:
            continue
        score = payload["score"]
        score_raw = np.asarray(score["score_raw"], dtype=np.float64)
        score_tokens, _, zero_mask = _performance_tokens(
            score, performances[0], include_pedal=include_pedal
        )
        prediction_tokens = predict_notes(model, score_tokens, zero_mask, device, include_pedal)
        pred_raw = tokens_to_raw(score_raw, score_tokens, prediction_tokens, zero_mask, args.pedal_mode)
        work_dir = args.output_dir / f"score_{work_index:02d}"
        raw_to_midi(score["pitch"], pred_raw, work_dir / "prediction.mid")
        gt_raw = []
        gt_paths = []
        for gt_index, performance in enumerate(performances):
            _, _, _ = _performance_tokens(score, performance, include_pedal=include_pedal)
            shared = np.asarray(performance["label_shared_raw"], dtype=np.float64)
            grouped = np.asarray(performance.get("grouped_ioi_raw", shared[:, 0]), dtype=np.float64)
            gt_ioi = expand_grouped_ioi(score_raw, grouped, zero_mask)
            pedals = np.asarray(performance.get("label_pedal4_raw", np.zeros((len(shared), 4))), dtype=np.float64)
            if not include_pedal:
                pedals = np.zeros_like(pedals)
            gt = np.column_stack([gt_ioi, shared[:, 1], shared[:, 2], pedals])
            gt_raw.append(gt)
            gt_path = work_dir / f"gt_{gt_index:03d}.mid"
            raw_to_midi(score["pitch"], gt, gt_path)
            gt_paths.append(str(gt_path))
        metrics = pn_metrics(work_dir / "prediction.mid", gt_paths, include_pedal)
        rows.append({"sidecar": str(sidecar), "prediction": str(work_dir / "prediction.mid"),
                     "ground_truth_count": len(gt_raw), "pn_wass": metrics})
        print(json.dumps({"work": work_index, "pn_wass_mean": metrics["pn_wass_mean"],
                          "references": len(gt_raw)}), flush=True)
        if args.limit_works is not None and len(rows) >= args.limit_works:
            break
    aggregate = {}
    if rows:
        keys = rows[0]["pn_wass"].keys()
        aggregate = {key: float(np.mean([row["pn_wass"][key] for row in rows])) for key in keys}
    result = {"pedal_mode": args.pedal_mode, "works": len(rows), "pn_wass": aggregate, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"works": len(rows), "pn_wass": aggregate}, indent=2))


if __name__ == "__main__":
    main()
