#!/usr/bin/env python3
"""Train one design1125 timing variant and export the minimum-loss checkpoint."""

# ruff: noqa: E402

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from transformers import get_linear_schedule_with_warmup, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.prepare_exp1125_chord import DATA, RUN, save, sha256, write_json
from src.data_process.dinr_chord_dataset import COARSE_TIMING_BINS, TIMING_BINS, ChordBatches, MODES
from src.model.dinr_chord import DINRChordModel
from src.model.dinr_chord_ar import DINRChordARModel
from src.model.dinr_chord_ar_input import (DINRChordARInputModel, FIELD_HEAD_INPUTS, LOG_SCORE_INPUTS, SCORE_INPUTS,
                                          VELOCITY_SCORE_INPUTS, uses_raw_score)
from src.model.dinr_chord_ar_mlp128_output import (DINRChordARMLP128DLMModel, DINRChordARMLP128OutputModel,
                                                  MLP128_DLM_LEARNED_MASKS, MLP128_DLM_MASK_MODES, MLP128_DLM_MODES,
                                                  MLP128_OUTPUT_MODES,
                                                  MLP128_SCORE_INPUT, RPS_WEIGHT)
from src.model.dinr_chord_ar_output import DIRECT_DLM_MODES, DINRChordAROutputModel, OUTPUT_MODES

SEED = 42
EPOCHS = 4
ACCUMULATION = 3
MICRO_BATCH = 16
EVAL_STEPS = 50
LR = 3e-4
HEADS = ('ioi', 'duration', 'velocity')
INPUT_RUN = RUN.parent / 'exp1125_chord_ar_inputs'
LOG_INPUT_RUN = RUN.parent / 'exp1125_chord_ar_log_inputs'
VELOCITY_INPUT_RUN = RUN.parent / 'exp1125_chord_ar_velocity_inputs'
FIELD_HEAD_RUN = RUN.parent / 'exp1125_chord_ar_field_heads'
OUTPUT_RUN = RUN.parent / 'exp1125_chord_ar_outputs'
DLM_DIRECT_RUN = RUN.parent / 'exp1125_chord_ar_dlm_direct'
MLP128_OUTPUT_RUN = RUN.parent / 'exp1125_chord_ar_mlp128_outputs'
MLP128_DLM_RUN = RUN.parent / 'exp1125_chord_ar_mlp128_dlm'
MLP128_DLM_MASK_RUN = RUN.parent / 'exp1125_chord_ar_mlp128_dlm_mask50'
MLP128_DLM_LEARNED_MASK_RUN = RUN.parent / 'exp1125_chord_ar_mlp128_dlm_mask_embeddings'


def loss_parts(model):
    return torch.stack([model.loss_parts[slot] for slot in range(len(HEADS))])


def loss_part_report(parts):
    return {key: dict(zip(HEADS, values, strict=True))
            for key, values in zip(('ce', 'rps'), parts.T.tolist(), strict=True)}


@torch.inference_mode()
def validation(model, data, batch_size, track_parts=False):
    model.eval()
    sums = torch.zeros(3, device=data.device)
    counts = torch.zeros_like(sums)
    parts = torch.zeros((3, 2), device=data.device)
    for indices in torch.arange(len(data)).split(batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            chunk, labels = model.losses(data(indices))
        sums += chunk
        counts += labels
        if track_parts:
            parts += loss_parts(model)
    result = (sums / counts).tolist()
    model.train()
    report = loss_part_report(parts / counts[:, None]) if track_parts else None
    return sum(result), dict(zip(HEADS, result, strict=True)), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--autoregressive', action='store_true')
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--score-input', choices=(*SCORE_INPUTS, *LOG_SCORE_INPUTS, *VELOCITY_SCORE_INPUTS,
                                                 *FIELD_HEAD_INPUTS))
    variant.add_argument('--output-mode', choices=(*OUTPUT_MODES, *DIRECT_DLM_MODES))
    variant.add_argument('--mlp128-output', choices=(*MLP128_OUTPUT_MODES, *MLP128_DLM_MODES, *MLP128_DLM_MASK_MODES))
    args = parser.parse_args()
    if (args.score_input or args.output_mode or args.mlp128_output) and (
            not args.autoregressive or args.mode != 'log_dev'):
        parser.error('Score-input and output variants require log_dev --autoregressive')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    set_seed(SEED)

    model_class = DINRChordARModel if args.autoregressive else DINRChordModel
    output = (RUN.parent / 'exp1125_chord_ar' if args.autoregressive else RUN)
    mask_history = args.mlp128_output in MLP128_DLM_MASK_MODES
    mlp128_dlm = mask_history or args.mlp128_output in MLP128_DLM_MODES
    if args.mlp128_output:
        model_class = DINRChordARMLP128DLMModel if mlp128_dlm else DINRChordARMLP128OutputModel
        output = MLP128_DLM_RUN if mlp128_dlm else MLP128_OUTPUT_RUN
        if mask_history:
            output = MLP128_DLM_MASK_RUN
        if args.mlp128_output in MLP128_DLM_LEARNED_MASKS:
            output = MLP128_DLM_LEARNED_MASK_RUN
        if args.smoke:
            output /= 'smoke'
        output /= args.mlp128_output
    elif args.score_input:
        model_class = DINRChordARInputModel
        if args.score_input in FIELD_HEAD_INPUTS:
            output = FIELD_HEAD_RUN
        elif args.score_input in VELOCITY_SCORE_INPUTS:
            output = VELOCITY_INPUT_RUN
        else:
            output = LOG_INPUT_RUN if args.score_input in LOG_SCORE_INPUTS else INPUT_RUN
        if args.smoke:
            output = output / 'smoke'
        output = output / args.score_input
    elif args.output_mode:
        model_class = DINRChordAROutputModel
        output = DLM_DIRECT_RUN if args.output_mode in DIRECT_DLM_MODES else OUTPUT_RUN
        if args.smoke:
            output = output / 'smoke'
        output = output / args.output_mode
    else:
        output = output / ('smoke' if args.smoke else args.mode)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / 'manifest.json').read_text())
    timing_bins = COARSE_TIMING_BINS if args.output_mode else TIMING_BINS
    if args.mlp128_output:
        timing_bins = DINRChordARMLP128OutputModel.timing_bins(args.mlp128_output)
    train_payload = torch.load(DATA / 'train.pt', map_location='cpu', weights_only=True, mmap=True)
    score_input = MLP128_SCORE_INPUT if args.mlp128_output else args.score_input
    raw_score = uses_raw_score(score_input)
    train = ChordBatches(train_payload, args.mode, device, raw_score=raw_score, timing_bins=timing_bins)
    valid = ChordBatches(torch.load(DATA / 'test.pt', map_location='cpu', weights_only=True), args.mode, device,
                         raw_score=raw_score, timing_bins=timing_bins)
    if args.smoke:
        valid.spans = valid.spans[:MICRO_BATCH]
    setting = args.score_input or args.output_mode or args.mlp128_output
    model = model_class(setting) if setting else model_class()
    initialization = None
    if args.output_mode in DIRECT_DLM_MODES or mlp128_dlm:
        initialization = model.initialize_from_training(train_payload)
    del train_payload
    model.to(device)
    if mask_history:
        model.history_mask_rng = torch.Generator(device=device).manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0., fused=True)
    per_epoch = math.ceil(len(train) / (MICRO_BATCH * ACCUMULATION))
    steps = 1 if args.smoke else EPOCHS * per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    protocol = dict(mode=args.mode, gpu=os.getenv('CUDA_VISIBLE_DEVICES'), seed=SEED, epochs=EPOCHS,
                    micro_batch=MICRO_BATCH, accumulation=ACCUMULATION, eval_steps=EVAL_STEPS,
                    training_windows=len(train), validation_windows=len(valid), optimizer_steps=steps,
                    lr=LR, model=model_class.__name__, train_sha256=meta['train']['sha256'],
                    validation_sha256=meta['test']['sha256'])
    if score_input:
        protocol.update(score_input=score_input, score_timing_transform='floor_log_ms' if model.log_timing else 'ms',
                        normalized_score_input=model.normalized, parameters=sum(p.numel() for p in model.parameters()),
                        score_mlp_width=model.score_mlp_width, velocity_input='lookup' if model.velocity_lookup else 'mlp',
                        velocity_input_output_tied=model.velocity_tied, field_head_width=model.field_head_width,
                        score_encoding='lookup' if model.score_lookup else 'mlp')
    if args.output_mode:
        protocol.update(score_input='lookup', output_mode=args.output_mode, timing_bins=timing_bins,
                        parameters=sum(p.numel() for p in model.parameters()))
    track_parts = args.mlp128_output == 'cat5000_rps'
    if args.mlp128_output:
        protocol.update(mlp128_output=args.mlp128_output, timing_bins=timing_bins, velocity_bins=128,
                        rps_weight=RPS_WEIGHT if track_parts else 0.0)
    if initialization is not None:
        protocol.update(dlm_initialization=initialization, location_parameterization='direct',
                        dlm_components=model.components, timing_scale_floor=0.005, velocity_scale_floor=0.5)
    if mask_history:
        protocol.update(history_mask_probability=model.history_mask_probability, history_mask_seed=SEED,
                        history_mask_granularity='chord', history_mask_value=MLP128_DLM_MASK_MODES[args.mlp128_output])
    write_json(output / 'run_config.json', protocol)

    step = 0
    best = float('inf')
    best_step = None
    rng = torch.Generator().manual_seed(SEED)
    start_time = time.monotonic()
    history = (output / 'loss_history.jsonl').open('w')
    with history:
        def record(row):
            history.write(json.dumps(row, allow_nan=False) + '\n')
            history.flush()
            print(row, flush=True)

        if score_input:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                record(dict(step=0, score_embedding_rms=model.score_rms(train(torch.arange(MICRO_BATCH)))))

        for epoch in range(1, EPOCHS + 1):
            order = torch.randperm(len(train), generator=rng)
            for block in order.split(MICRO_BATCH * ACCUMULATION):
                optimizer.zero_grad(set_to_none=True)
                microbatches = block.split(MICRO_BATCH)
                accumulated = torch.zeros(3, device=device)
                if track_parts:
                    parts = torch.zeros((3, 2), device=device)
                lr = scheduler.get_last_lr()[0]
                for indices in microbatches:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        sums, counts = model.losses(train(indices))
                        losses = sums / counts.clamp_min(1)
                    if track_parts:
                        parts += loss_parts(model) / counts[:, None].clamp_min(1) / len(microbatches)
                    (losses.sum() / len(microbatches)).backward()
                    accumulated += losses.detach() / len(microbatches)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                step += 1
                heads = accumulated.tolist()
                row = dict(step=step, epoch=epoch, train_loss=sum(heads), lr=lr,
                           train_heads=dict(zip(HEADS, heads, strict=True)), elapsed_s=time.monotonic() - start_time)
                if track_parts:
                    row['train_loss_parts'] = loss_part_report(parts)
                record(row)
                if step % EVAL_STEPS == 0 or step == steps:
                    loss, heads, report = validation(model, valid, MICRO_BATCH, track_parts)
                    if loss < best:
                        best, best_step = loss, step
                        save(output / 'best.pt', model.state_dict())
                    row = dict(step=step, epoch=epoch, valid_loss=loss, valid_heads=heads, best_step=best_step,
                               elapsed_s=time.monotonic() - start_time)
                    if track_parts:
                        row['valid_loss_parts'] = report
                    if score_input:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            row['score_embedding_rms'] = model.score_rms(train(torch.arange(MICRO_BATCH)))
                    record(row)
                if args.smoke:
                    break
            if args.smoke:
                break
    checkpoint = output / 'best.pt'
    assert step == steps and best_step is not None
    completion = dict(optimizer_steps=step, epochs=1 if args.smoke else EPOCHS, best_step=best_step,
                      best_valid_loss=best, sha256=sha256(checkpoint), checkpoint=str(checkpoint))
    write_json(output / 'training_complete.json', completion)
    print(completion, flush=True)


if __name__ == '__main__':
    main()
