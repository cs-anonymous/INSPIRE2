"""Greedy DINR-absolute test inference, MIDI reconstruction, and PN W1."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import traceback
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
    state = load_state_dict(checkpoint)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    slot_projection = state.get("note_slot_projection")
    num_input_slots = int(slot_projection.shape[0]) if slot_projection is not None else None
    separate_ioi_duration_tables = "ioi_token_embedding.weight" in state
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
    model = DINRAbsoluteModel(
        config,
        pedal_mode=args.pedal_mode,
        numerical_coordinates=args.numerical_coordinates,
        coordinate_frequencies=args.coordinate_frequencies,
        num_input_slots=num_input_slots,
        separate_ioi_duration_tables=separate_ioi_duration_tables,
        decoder_mask_mode="timestep",
        timing_target_mode=args.timing_target_mode or "absolute",
        time_input_mode=args.time_input_mode or "linear",
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing checkpoint tensors={len(missing)}")
    if unexpected:
        print(f"warning: unexpected checkpoint tensors={len(unexpected)}")
    return model.eval()


def stable_seed(base_seed, work_index, sample_index):
    payload = f"{int(base_seed)}::{int(work_index)}::{int(sample_index)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def sample_top_p(logits, temperature, top_p):
    """Sample one categorical value with temperature scaling and nucleus truncation."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    logits = logits.float() / float(temperature)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative - sorted_probs > float(top_p)
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return int(torch.multinomial(torch.softmax(filtered, dim=-1), 1).item())


def predict_notes(model, score_tokens, score_raw, score_zero_mask, device, include_pedal,
                  temperature=1.0, top_p=0.9, seed=42):
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    score = torch.tensor(score_tokens, dtype=torch.long, device=device).unsqueeze(0)
    n = score.shape[1]
    generated = []
    decoder_history = []
    with torch.no_grad():
        score_embeds = model._sum_note_tokens(score)
        score_mask = torch.ones((1, n), dtype=torch.long, device=device)
        encoder_outputs = model.backbone.encoder(
            inputs_embeds=score_embeds, attention_mask=score_mask
        )
        past_key_values = None
        for index in range(n):
            current = ([DINRTokenSpec.bos_id] * model.num_input_slots) if index == 0 else decoder_history[-1]
            decoder = torch.tensor(current, dtype=torch.long, device=device).view(
                1, 1, model.num_input_slots
            )
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
            ioi_class = sample_top_p(logits[head][0, -1], temperature, top_p)
            duration_class = sample_top_p(logits["duration"][0, -1], temperature, top_p)
            velocity_class = sample_top_p(logits["velocity"][0, -1], temperature, top_p)
            if model.pedal_mode == "ce":
                pedal = [sample_top_p(logits["pedal"][0, -1, j], temperature, top_p)
                         for j in range(4)]
            elif model.pedal_mode == "bce":
                pedal = [
                    sample_top_p(torch.stack((logits["pedal"][0, -1, j].new_zeros(()),
                                              logits["pedal"][0, -1, j])),
                                 temperature, top_p)
                    for j in range(4)
                ]
            else:
                pedal = [0, 0, 0, 0]
            token = [
                int(score_tokens[index][0]),
                DINRTokenSpec.timing_start + ioi_class,
                DINRTokenSpec.velocity_start + velocity_class,
                DINRTokenSpec.timing_start + duration_class,
                *[DINRTokenSpec.pedal_start + int(value) for value in pedal],
            ]
            if not include_pedal:
                token = token[:4]
                if model.num_input_slots == 8:
                    token.extend([DINRTokenSpec.pad_id] * 4)
            generated.append(token)
            feedback = list(token)
            if model.timing_target_mode == "score_relative_log":
                if score_zero_mask[index]:
                    grouped_ms = np.exp(ioi_class / 499.0 * 5.0)
                else:
                    grouped_ms = max(float(score_raw[index, 0]), 1.0) * np.exp(-3.0 + ioi_class / 499.0 * 5.0)
                duration_ms = max(float(score_raw[index, 1]), 1.0) * np.exp(-3.0 + duration_class / 499.0 * 5.0)
                feedback[1] = DINRTokenSpec.timing_start + DINRTokenSpec.log_time_bin(grouped_ms)
                feedback[3] = DINRTokenSpec.timing_start + DINRTokenSpec.log_time_bin(duration_ms)
            decoder_history.append(feedback)
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


def tokens_to_raw(score_raw, score_tokens, pred_tokens, score_zero_mask, pedal_mode,
                  timing_target_mode="absolute"):
    pred_tokens = np.asarray(pred_tokens, dtype=np.int64)
    score_raw = np.asarray(score_raw, dtype=np.float64)
    ioi_class = pred_tokens[:, 1] - DINRTokenSpec.timing_start
    duration_class = pred_tokens[:, 3] - DINRTokenSpec.timing_start
    grouped = ioi_class * DINRTokenSpec.timing_bin_ms
    duration = duration_class * DINRTokenSpec.timing_bin_ms
    if timing_target_mode == "log_absolute":
        grouped = np.asarray([DINRTokenSpec.log_time_ms(x) for x in ioi_class])
        duration = np.asarray([DINRTokenSpec.log_time_ms(x) for x in duration_class])
    elif timing_target_mode == "score_relative_log":
        score_ioi = score_raw[:, 0]
        score_duration = score_raw[:, 1]
        decoded_ioi = np.zeros(len(ioi_class), dtype=np.float64)
        for index, category in enumerate(ioi_class):
            if score_zero_mask[index]:
                decoded_ioi[index] = np.exp(category / 499.0 * 5.0)
            else:
                decoded_ioi[index] = max(score_ioi[index], 1.0) * np.exp(-3.0 + category / 499.0 * 5.0)
        grouped = decoded_ioi
        duration = np.maximum(score_duration, 1.0) * np.exp(-3.0 + duration_class / 499.0 * 5.0)
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
    parser.add_argument("--numerical-coordinates", action="store_true")
    parser.add_argument("--coordinate-frequencies", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--encoder-layers", type=int, default=10)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--limit-works", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--timing-target-mode", choices=["absolute", "log_absolute", "score_relative_log"], default=None)
    parser.add_argument("--time-input-mode", choices=["linear", "log_absolute"], default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def test_sidecars(sidecar_root):
    """Return one usable sidecar per metadata test work, largest first.

    The original grouped-median preprocessing has no grouped sidecar for the
    two MS-score test works.  Those works still have a MUSICAL51 sidecar with
    the same score/performance labels, so use it as a fallback.  Selecting via
    metadata also makes the expected 19-work test protocol explicit.
    """
    sidecar_root = Path(sidecar_root).resolve()
    metadata_path = sidecar_root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv under {sidecar_root}")
    import csv
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    score_sources = sorted({
        row["refined_score_midi_path"] for row in rows
        if row.get("split") == "test" and row.get("refined_score_midi_path")
    })
    specs = []
    for source in score_sources:
        directory = sidecar_root / Path(source).parent
        stem = Path(source).stem
        grouped = directory / f"{stem}.ASAP_GROUPED_MEDIAN.pt"
        musical = directory / f"{stem}.ASAP_MUSICAL51.pt"
        sidecar = grouped if grouped.is_file() else musical
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"No grouped-median or MUSICAL51 sidecar for test work {source}"
            )
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        if not any(p.get("split") == "test" for p in payload.get("performances", [])):
            raise ValueError(f"Selected sidecar has no test performances: {sidecar}")
        specs.append((len(payload["score"]["pitch"]), sidecar))
    # The requested inference order is largest work to smallest work.
    specs.sort(key=lambda item: (-item[0], str(item[1])))
    if len(specs) != 19:
        raise ValueError(f"Expected 19 test works, found {len(specs)}")
    return [path for _, path in specs]


def prepare_work(sidecar, work_index, output_dir, include_pedal, num_input_slots, time_input_mode="linear"):
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    performances = [p for p in payload.get("performances", []) if p.get("split") == "test"]
    if not performances:
        return None
    score = payload["score"]
    score_raw = np.asarray(score["score_raw"], dtype=np.float64)
    score_tokens, _, zero_mask = _performance_tokens(
        score, performances[0], include_pedal=include_pedal
    )
    if num_input_slots == 8 and not include_pedal:
        score_tokens = [tokens + [DINRTokenSpec.pad_id] * 4 for tokens in score_tokens]
    if time_input_mode == "log_absolute":
        for note in score_tokens:
            for slot in (1, 3):
                ms = max(int(note[slot] - DINRTokenSpec.timing_start), 0) * DINRTokenSpec.timing_bin_ms
                note[slot] = DINRTokenSpec.timing_start + DINRTokenSpec.log_time_bin(ms)
    work_dir = output_dir / f"score_{work_index:02d}"
    gt_paths = []
    for gt_index, performance in enumerate(performances):
        shared = np.asarray(performance["label_shared_raw"], dtype=np.float64)
        grouped = np.asarray(performance.get("grouped_ioi_raw", shared[:, 0]), dtype=np.float64)
        gt_ioi = expand_grouped_ioi(score_raw, grouped, zero_mask)
        pedals = np.asarray(
            performance.get("label_pedal4_raw", np.zeros((len(shared), 4))), dtype=np.float64
        )
        if not include_pedal:
            pedals = np.zeros_like(pedals)
        gt = np.column_stack([gt_ioi, shared[:, 1], shared[:, 2], pedals])
        gt_path = work_dir / f"gt_{gt_index:03d}.mid"
        raw_to_midi(score["pitch"], gt, gt_path)
        gt_paths.append(str(gt_path))
    return {
        "work": int(work_index),
        "sidecar": str(sidecar),
        "score_pitch": score["pitch"],
        "score_raw": score_raw.tolist(),
        "score_tokens": score_tokens,
        "score_zero_mask": zero_mask,
        "gt_paths": gt_paths,
    }


def worker_loop(worker_index, args, job_queue, result_queue):
    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = make_model(args, args.checkpoint).to(device)
        model.eval()
        print(f"Worker {worker_index} using {device}", flush=True)
        while True:
            job = job_queue.get()
            if job is None:
                break
            spec, sample_index = job
            try:
                sample_seed = stable_seed(args.seed, spec["work"], sample_index)
                prediction_tokens = predict_notes(
                    model, spec["score_tokens"], np.asarray(spec["score_raw"], dtype=np.float64),
                    spec["score_zero_mask"], device,
                    args.pedal_mode != "none", args.temperature, args.top_p, sample_seed,
                )
                pred_raw = tokens_to_raw(
                    np.asarray(spec["score_raw"], dtype=np.float64), spec["score_tokens"],
                    prediction_tokens, spec["score_zero_mask"], args.pedal_mode,
                    args.timing_target_mode,
                )
                work_dir = args.output_dir / f"score_{spec['work']:02d}" / f"sample_{sample_index:03d}"
                pred_path = work_dir / "prediction.mid"
                raw_to_midi(spec["score_pitch"], pred_raw, pred_path)
                metrics = pn_metrics(pred_path, spec["gt_paths"], args.pedal_mode != "none")
                result_queue.put({
                    "work": spec["work"], "sample": int(sample_index),
                    "seed": sample_seed, "prediction": str(pred_path),
                    "ground_truth_count": len(spec["gt_paths"]), "pn_wass": metrics,
                    "error": None,
                })
            except Exception:
                result_queue.put({"work": spec["work"], "sample": int(sample_index),
                                  "error": traceback.format_exc()})
    except Exception:
        result_queue.put({"worker": worker_index, "error": traceback.format_exc(), "fatal": True})


def run_pool(args, specs):
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    jobs = [(spec, sample) for spec in specs for sample in range(args.num_samples)]
    workers = [ctx.Process(target=worker_loop, args=(i, args, job_queue, result_queue))
               for i in range(args.num_workers)]
    for worker in workers:
        worker.start()
    for job in jobs:
        job_queue.put(job)
    for _ in workers:
        job_queue.put(None)
    rows = []
    with __import__("tqdm").tqdm(total=len(jobs), desc="DINR sampling inference") as progress:
        for _ in jobs:
            row = result_queue.get()
            if row.get("error"):
                for worker in workers:
                    worker.terminate()
                raise RuntimeError(row["error"])
            rows.append(row)
            progress.update(1)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"worker {worker.pid} exited with code {worker.exitcode}")
    return rows


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    include_pedal = args.pedal_mode != "none"
    checkpoint_path = Path(args.checkpoint).resolve()
    config_paths = [checkpoint_path / "dinr_config.json"]
    config_paths.extend(parent / "dinr_config.json" for parent in checkpoint_path.parents)
    config = next((json.loads(path.read_text(encoding="utf-8")) for path in config_paths if path.is_file()), {})
    args.timing_target_mode = args.timing_target_mode or config.get("timing_target_mode", "absolute")
    args.time_input_mode = args.time_input_mode or config.get("time_input_mode", "linear")
    num_input_slots = int(config.get("num_input_slots") or 8)
    sidecars = test_sidecars(args.sidecar_root)
    specs = []
    for work_index, sidecar in enumerate(sidecars):
        spec = prepare_work(sidecar, work_index, args.output_dir, include_pedal, num_input_slots,
                            args.time_input_mode)
        if spec is not None:
            specs.append(spec)
        if args.limit_works is not None and len(specs) >= args.limit_works:
            break
    print(f"DINR sampling: {len(specs)} scores x {args.num_samples} samples, "
          f"{args.num_workers} workers, temperature={args.temperature}, top_p={args.top_p}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for spec in specs:
        for sample_index in range(args.num_samples):
            run_records.append({
                "gpu_visible_device": args.device,
                "checkpoint": str(args.checkpoint.resolve()),
                "training_seed": 42,
                "sampling_seed": stable_seed(args.seed, spec["work"], sample_index),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "score_index": spec["work"],
                "sample_index": sample_index,
                "output_dir": str(args.output_dir / f"score_{spec['work']:02d}" / f"sample_{sample_index:03d}"),
                "status": "planned",
            })
    (args.output_dir / "runs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in run_records), encoding="utf-8"
    )
    rows = run_pool(args, specs)
    rows.sort(key=lambda row: (row["work"], row["sample"]))
    aggregate = {}
    if rows:
        keys = rows[0]["pn_wass"].keys()
        aggregate = {key: float(np.mean([row["pn_wass"][key] for row in rows])) for key in keys}
    result = {
        "pedal_mode": args.pedal_mode, "works": len(specs), "num_samples": args.num_samples,
        "num_predictions": len(rows), "num_workers": args.num_workers,
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
        "pn_wass": aggregate, "rows": rows,
    }
    completed = {(row["work"], row["sample"]): row for row in rows}
    for record in run_records:
        row = completed.get((record["score_index"], record["sample_index"]))
        record["status"] = "complete" if row else "failed"
        if row:
            record["pn_wass_mean"] = row["pn_wass"]["pn_wass_mean"]
    (args.output_dir / "runs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in run_records), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"works": len(specs), "num_predictions": len(rows), "pn_wass": aggregate}, indent=2))


if __name__ == "__main__":
    main()
