#!/usr/bin/env python3
"""Score chord decoding with the same teacher-forced history at every position."""

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.prepare_exp1125_chord import DATA, write_json
from script.run_exp1125_chord import training_result
from script.run_exp1125_chord_ar import DRAWS, TEMPERATURE, TOP_P
from script.train_exp1125_chord import HEADS, MICRO_BATCH, MLP128_DLM_MASK_RUN, MLP128_DLM_RUN, SEED
from src.data_process.dinr_chord_dataset import COARSE_TIMING_BINS, ChordBatches, decode_tokens
from src.model.dinr_chord_ar_mlp128_output import DINRChordARMLP128DLMModel, MLP128_DLM_MASK50


@torch.inference_mode()
def audit(variant, directory):
    checkpoint = training_result(directory)
    device = torch.device('cuda:0')
    model = DINRChordARMLP128DLMModel(variant)
    state = torch.load(checkpoint['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, assign=True)
    model = model.to(device).eval()
    payload = torch.load(DATA / 'test.pt', map_location='cpu', weights_only=True)
    batches = ChordBatches(payload, 'log_dev', device, raw_score=True, timing_bins=COARSE_TIMING_BINS)
    targets = payload['targets'].to(device)
    rng = torch.Generator(device=device).manual_seed(SEED)
    counts = torch.zeros(3, device=device)
    totals = torch.zeros(3, 3, device=device, dtype=torch.float64)

    for indices in torch.arange(len(batches)).split(MICRO_BATCH):
        pitches, score, attention, advances, labels = batches(indices)
        offsets = torch.arange(score.shape[1], device=device)
        rows = (batches.spans[indices, 0].to(device)[:, None] + offsets)[attention]
        target = targets[rows]
        valid = labels[attention] >= 0
        valid[:, 0] &= advances[attention]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = model.teacher_forced_hidden(pitches, score, attention, advances, labels)[attention]
            picks = [model.decode_step(hidden, generator, TEMPERATURE, TOP_P)[0]
                     for generator in (None, *([rng] * DRAWS))]
        values = torch.stack([decode_tokens(pick, score[attention], 'log_dev', COARSE_TIMING_BINS)
                              for pick in picks])
        median_error = (values[0] - target).abs()
        expected_error = (values[1:] - target).abs().mean(0)
        mean_draw_error = (values[1:].mean(0) - target).abs()
        errors = torch.stack((median_error, expected_error, mean_draw_error))
        totals += errors.masked_fill(~valid, 0.).sum(1, dtype=torch.float64)
        counts += valid.sum(0)

    names = ('median_mae', 'mean_per_draw_mae', 'mae_of_draw_mean')
    metrics = {name: dict(zip(HEADS, result, strict=True))
               for name, result in zip(names, (totals / counts).tolist(), strict=True)}
    return dict(variant=variant, checkpoint_sha256=checkpoint['sha256'], windows=len(batches),
                history='complete_shifted_ground_truth', draws=DRAWS, seed=SEED,
                temperature=TEMPERATURE, top_p=TOP_P, target_counts=dict(zip(HEADS, counts.int().tolist(), strict=True)),
                **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unmasked', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = 'all_dlm_k1_mlp128'
    variant, root = (base, MLP128_DLM_RUN) if args.unmasked else (MLP128_DLM_MASK50, MLP128_DLM_MASK_RUN)
    report = audit(variant, root / variant)
    write_json(MLP128_DLM_MASK_RUN / f'{variant}_teacher_forced.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
