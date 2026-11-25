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
from src.model.dinr_chord_ar_input import DINRChordARInputModel, SCORE_INPUTS
from src.model.dinr_chord_ar_output import DIRECT_DLM_MODES, DINRChordAROutputModel, OUTPUT_MODES

SEED = 42
EPOCHS = 4
ACCUMULATION = 3
MICRO_BATCH = 16
EVAL_STEPS = 50
LR = 3e-4
HEADS = ('ioi', 'duration', 'velocity')
INPUT_RUN = RUN.parent / 'exp1125_chord_ar_inputs'
OUTPUT_RUN = RUN.parent / 'exp1125_chord_ar_outputs'
DLM_DIRECT_RUN = RUN.parent / 'exp1125_chord_ar_dlm_direct'


@torch.inference_mode()
def validation(model, data, batch_size):
    model.eval()
    sums = torch.zeros(3, device=data.device)
    counts = torch.zeros_like(sums)
    for indices in torch.arange(len(data)).split(batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            chunk, labels = model.losses(data(indices))
        sums += chunk
        counts += labels
    result = (sums / counts).tolist()
    model.train()
    return sum(result), dict(zip(HEADS, result, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--autoregressive', action='store_true')
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--score-input', choices=SCORE_INPUTS)
    variant.add_argument('--output-mode', choices=(*OUTPUT_MODES, *DIRECT_DLM_MODES))
    args = parser.parse_args()
    if (args.score_input or args.output_mode) and (not args.autoregressive or args.mode != 'log_dev'):
        parser.error('Score-input and output variants require log_dev --autoregressive')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    set_seed(SEED)

    model_class = DINRChordARModel if args.autoregressive else DINRChordModel
    output = (RUN.parent / 'exp1125_chord_ar' if args.autoregressive else RUN)
    if args.score_input:
        model_class = DINRChordARInputModel
        output = INPUT_RUN / 'smoke' if args.smoke else INPUT_RUN
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
    train_payload = torch.load(DATA / 'train.pt', map_location='cpu', weights_only=True, mmap=True)
    train = ChordBatches(train_payload, args.mode, device, raw_score=bool(args.score_input), timing_bins=timing_bins)
    valid = ChordBatches(torch.load(DATA / 'test.pt', map_location='cpu', weights_only=True), args.mode, device,
                         raw_score=bool(args.score_input), timing_bins=timing_bins)
    if args.smoke:
        valid.spans = valid.spans[:MICRO_BATCH]
    setting = args.score_input or args.output_mode
    model = model_class(setting) if setting else model_class()
    initialization = None
    if args.output_mode in DIRECT_DLM_MODES:
        initialization = model.initialize_from_training(train_payload)
    del train_payload
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0., fused=True)
    per_epoch = math.ceil(len(train) / (MICRO_BATCH * ACCUMULATION))
    steps = 1 if args.smoke else EPOCHS * per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    protocol = dict(mode=args.mode, gpu=os.getenv('CUDA_VISIBLE_DEVICES'), seed=SEED, epochs=EPOCHS,
                    micro_batch=MICRO_BATCH, accumulation=ACCUMULATION, eval_steps=EVAL_STEPS,
                    training_windows=len(train), validation_windows=len(valid), optimizer_steps=steps,
                    lr=LR, model=model_class.__name__, train_sha256=meta['train']['sha256'],
                    validation_sha256=meta['test']['sha256'])
    if args.score_input:
        protocol.update(score_input=args.score_input, parameters=sum(p.numel() for p in model.parameters()))
    if args.output_mode:
        protocol.update(score_input='lookup', output_mode=args.output_mode, timing_bins=timing_bins,
                        parameters=sum(p.numel() for p in model.parameters()))
    if initialization is not None:
        protocol.update(dlm_initialization=initialization, location_parameterization='direct',
                        dlm_components=model.components, timing_scale_floor=0.005, velocity_scale_floor=0.5)
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

        if args.score_input:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                record(dict(step=0, score_embedding_rms=model.score_rms(train(torch.arange(MICRO_BATCH)))))

        for epoch in range(1, EPOCHS + 1):
            order = torch.randperm(len(train), generator=rng)
            for block in order.split(MICRO_BATCH * ACCUMULATION):
                optimizer.zero_grad(set_to_none=True)
                microbatches = block.split(MICRO_BATCH)
                accumulated = torch.zeros(3, device=device)
                lr = scheduler.get_last_lr()[0]
                for indices in microbatches:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        sums, counts = model.losses(train(indices))
                        losses = sums / counts.clamp_min(1)
                    (losses.sum() / len(microbatches)).backward()
                    accumulated += losses.detach() / len(microbatches)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                step += 1
                heads = accumulated.tolist()
                record(dict(step=step, epoch=epoch, train_loss=sum(heads), lr=lr,
                            train_heads=dict(zip(HEADS, heads, strict=True)), elapsed_s=time.monotonic() - start_time))
                if step % EVAL_STEPS == 0 or step == steps:
                    loss, heads = validation(model, valid, MICRO_BATCH)
                    if loss < best:
                        best, best_step = loss, step
                        save(output / 'best.pt', model.state_dict())
                    row = dict(step=step, epoch=epoch, valid_loss=loss, valid_heads=heads, best_step=best_step,
                               elapsed_s=time.monotonic() - start_time)
                    if args.score_input:
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
