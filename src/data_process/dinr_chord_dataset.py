"""Chord layout, timing representation and batches for design1125."""

from typing import NamedTuple
import math

import numpy as np
import torch

LIMIT = 512
TIMING_BINS = 5000
COARSE_TIMING_BINS = 500
LOG_DEV_LOW = -3.0
LOG_DEV_HIGH = 2.0
MODES = ('abs5000', 'log_abs', 'log_dev')


class ChordLayout(NamedTuple):
    notes: np.ndarray
    onsets: np.ndarray
    boundaries: np.ndarray
    oversized_measures: int
    membership: np.ndarray


def chord_layout(geometry, keep):
    _, _, measure, ticks, group_starts, fields = geometry
    pitches = fields[keep, 1].astype(np.int64)
    durations = fields[keep, 2].astype(np.int64)
    onset_ids = np.repeat(np.arange(len(ticks)), np.diff(group_starts, append=len(keep)))[keep]
    keys, member, notes = np.unique(np.column_stack((onset_ids, durations)), axis=0,
                                    return_inverse=True, return_counts=True)
    onset, duration = keys.T
    low_pitch = np.full(len(notes), 128)
    np.minimum.at(low_pitch, member, pitches)
    order = np.lexsort((duration, low_pitch, onset))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    onset = onset[order]
    measures = measure['ids'][group_starts[onset]]
    bars = np.r_[0, np.flatnonzero(np.diff(measures)) + 1, len(notes)]
    onset_breaks = np.flatnonzero(np.diff(onset)) + 1
    oversized = np.flatnonzero(np.diff(bars) > LIMIT)
    fallback = [onset_breaks[(onset_breaks > bars[i]) & (onset_breaks < bars[i + 1])] for i in oversized]
    boundaries = np.sort(np.concatenate([bars, *fallback]))
    return ChordLayout(notes[order], onset, boundaries, len(oversized), rank[member])


def nonoverlapping_windows(boundaries):
    start = 0
    while start < boundaries[-1]:
        end = int(boundaries[np.searchsorted(boundaries, start + LIMIT, side='right') - 1])
        if end <= start:
            raise ValueError('An onset block exceeds the window cap')
        yield start, end
        start = end


def quantize(values, low, high):
    return ((values.clamp(low, high) - low) * ((TIMING_BINS - 1) / (high - low))).round().long()


def score_tokens(score, mode):
    times = score[:, :2]
    if mode == 'abs5000':
        timing = times.round().clamp(0, TIMING_BINS - 1).long()
    else:
        timing = quantize(times.clamp_min(1).log(), 0, math.log(5000))
    return torch.cat((timing, score[:, 2:].round().clamp(0, 127).long()), -1)


def log_dev_edges(device):
    return torch.linspace(LOG_DEV_LOW, LOG_DEV_HIGH, COARSE_TIMING_BINS + 1, device=device, dtype=torch.float32)


def encode_targets(score, targets, ioi_valid, mode, timing_bins=TIMING_BINS):
    if mode == 'log_dev':
        deviation = targets[:, :2].clamp_min(1).log() - score[:, :2].clamp_min(1).log()
        if timing_bins == COARSE_TIMING_BINS:
            timing = torch.bucketize(deviation.contiguous(), log_dev_edges(score.device)[1:-1], right=True)
        else:
            timing = quantize(deviation, LOG_DEV_LOW, LOG_DEV_HIGH)
        labels = torch.column_stack((timing, targets[:, 2].round().clamp(0, 127).long()))
    else:
        labels = score_tokens(targets, mode)
    labels[~ioi_valid, 0] = -100
    return labels


def decode_tokens(tokens, score, mode, timing_bins=TIMING_BINS, continuous_timing=None):
    categories = tokens[:, :2].float()
    if mode == 'abs5000':
        timing = categories
    elif mode == 'log_abs':
        timing = (categories * (math.log(5000) / (TIMING_BINS - 1))).exp()
    else:
        if continuous_timing is not None:
            deviation = continuous_timing.clamp(LOG_DEV_LOW, LOG_DEV_HIGH)
        elif timing_bins == COARSE_TIMING_BINS:
            width = (LOG_DEV_HIGH - LOG_DEV_LOW) / COARSE_TIMING_BINS
            deviation = (categories + 0.5) * width + LOG_DEV_LOW
        else:
            deviation = categories * (5 / (TIMING_BINS - 1)) - 3
        timing = score[:, :2].clamp_min(1) * deviation.exp()
    timing = timing.clamp_min(1)
    timing[score[:, 0] == 0, 0] = 0
    return torch.column_stack((timing, tokens[:, 2].clamp(1, 127)))


class ChordBatches:
    def __init__(self, payload, mode, device, raw_score=False, timing_bins=TIMING_BINS):
        self.spans = payload['spans'].long().cpu()
        self.device = device
        self.pitches = payload['pitches'].to(device)
        score = payload['score'].to(device)
        self.score = score if raw_score else score_tokens(score, mode)
        self.advances = score[:, 0] > 0
        self.labels = None
        if 'targets' in payload:
            self.labels = encode_targets(score, payload['targets'].to(device), payload['ioi_valid'].to(device), mode,
                                         timing_bins=timing_bins)

    def __len__(self):
        return len(self.spans)

    def __call__(self, sample_indices):
        spans = self.spans[sample_indices]
        lengths = spans[:, 1] - spans[:, 0]
        width = ((int(lengths.max()) + 7) // 8) * 8
        offsets = torch.arange(width, device=self.device)[None, :]
        mask = offsets < lengths.to(self.device)[:, None]
        take = spans[:, 0].to(self.device)[:, None] + offsets.masked_fill(~mask, 0)
        labels = self.labels[take] if self.labels is not None else None
        return self.pitches[take], self.score[take], mask, self.advances[take], labels
