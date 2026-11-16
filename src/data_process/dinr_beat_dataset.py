"""Beat-and-note event dataset for the two-level DINR experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


def fixed_event_spans(event_count, block_events=512):
    """Cut consecutive events without imposing BEAT/NOTE boundary rules."""
    if block_events < 1:
        raise ValueError('block_events must be positive')
    return [(start, min(start + block_events, event_count))
            for start in range(0, event_count, block_events)]


class BeatTokenSpec:
    beat_type = 0
    note_type = 1
    bos_type = 2
    pad_type = 3
    duration_bins = 500
    onset_bins = 500
    velocity_bins = 128
    pitch_bins = 128
    duration_abs_range = (3.0, 8.0)  # natural log milliseconds
    duration_dev_range = (-2.0, 3.0)
    onset_range = (-0.25, 1.0)

    @staticmethod
    def quantize(value, bounds, bins):
        low, high = bounds
        value = torch.as_tensor(value, dtype=torch.float32).clamp(low, high)
        return torch.round((value - low) * ((bins - 1) / (high - low))).long()

    @staticmethod
    def dequantize(token, bounds, bins):
        low, high = bounds
        token = torch.as_tensor(token, dtype=torch.float32).clamp(0, bins - 1)
        return low + token * ((high - low) / (bins - 1))


def _duration_target(score_ms, perf_ms, mode):
    score_log = torch.log(torch.as_tensor(score_ms, dtype=torch.float32).clamp_min(1.0))
    perf_log = torch.log(torch.as_tensor(perf_ms, dtype=torch.float32).clamp_min(1.0))
    if mode == "absolute_log":
        return BeatTokenSpec.quantize(perf_log, BeatTokenSpec.duration_abs_range,
                                      BeatTokenSpec.duration_bins)
    if mode == "log_dev":
        return BeatTokenSpec.quantize(perf_log - score_log, BeatTokenSpec.duration_dev_range,
                                      BeatTokenSpec.duration_bins)
    raise ValueError(f"Unknown duration target mode: {mode}")


class DINRBeatDataset(Dataset):
    """Quantize precomputed continuous event sidecars for one experiment setting."""

    def __init__(self, path, duration_target="absolute_log", input_duration_tables="shared",
                 input_measure_start=True, limit=None, beat_duration_target=None,
                 note_duration_target=None):
        self.path = Path(path)
        self.data = torch.load(self.path, map_location="cpu", weights_only=False)
        if self.data.get("format") != "dinr_beat_events_v1":
            raise ValueError(f"Unsupported beat event sidecar: {self.data.get('format')}")
        beat_duration_target = beat_duration_target or duration_target
        note_duration_target = note_duration_target or duration_target
        if beat_duration_target not in {"absolute_log", "log_dev"} or note_duration_target not in {
            "absolute_log", "log_dev"
        }:
            raise ValueError("duration_target must be absolute_log or log_dev")
        if input_duration_tables not in {"shared", "separate"}:
            raise ValueError("input_duration_tables must be shared or separate")
        self.duration_target = duration_target
        self.beat_duration_target = beat_duration_target
        self.note_duration_target = note_duration_target
        self.input_duration_tables = input_duration_tables
        self.input_measure_start = bool(input_measure_start)
        self.samples = self.data["samples"][:limit] if limit else self.data["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start, end = self.samples[index]["span"]
        events = self.data["events"][start:end].float()
        targets = self.data["targets"][start:end].float()
        event_type = events[:, 0].long()
        is_beat = event_type == BeatTokenSpec.beat_type
        is_note = event_type == BeatTokenSpec.note_type

        score_duration = events[:, 2].clamp_min(1.0)
        score_log = torch.log(score_duration)
        score_duration_token = BeatTokenSpec.quantize(
            score_log, BeatTokenSpec.duration_abs_range, BeatTokenSpec.duration_bins
        )
        beat_duration_token = _duration_target(
            score_duration, targets[:, 0], self.beat_duration_target
        )
        note_duration_token = _duration_target(
            score_duration, targets[:, 0], self.note_duration_target
        )
        perf_duration_input = BeatTokenSpec.quantize(
            torch.log(targets[:, 0].clamp_min(1.0)), BeatTokenSpec.duration_abs_range,
            BeatTokenSpec.duration_bins
        )
        input_duration_token = score_duration_token.clone()
        # Keep the event role separate from the value index.  The model has
        # two 500-row tables, so a note token must not become an index in a
        # hypothetical concatenated 1000-row table.
        if self.input_duration_tables == "separate":
            input_duration_token = score_duration_token

        score_onset = BeatTokenSpec.quantize(events[:, 3], BeatTokenSpec.onset_range,
                                             BeatTokenSpec.onset_bins)
        perf_onset = BeatTokenSpec.quantize(targets[:, 1], BeatTokenSpec.onset_range,
                                            BeatTokenSpec.onset_bins)
        pitch = events[:, 1].long().clamp(0, BeatTokenSpec.pitch_bins - 1)
        score_velocity = events[:, 4].long().clamp(0, BeatTokenSpec.velocity_bins - 1)
        perf_velocity = targets[:, 2].long().clamp(0, BeatTokenSpec.velocity_bins - 1)
        measure_start = events[:, 5].long().clamp(0, 1)
        if not self.input_measure_start:
            measure_start.zero_()

        # Columns: type, pitch, duration, onset, velocity, measure_start.
        encoder_tokens = torch.stack((event_type, pitch, input_duration_token,
                                      score_onset, score_velocity, measure_start), dim=-1)
        labels = torch.full((len(events), 4), -100, dtype=torch.long)
        labels[is_beat, 0] = beat_duration_token[is_beat]
        labels[is_note, 1] = note_duration_token[is_note]
        labels[is_note, 2] = perf_onset[is_note]
        labels[is_note, 3] = perf_velocity[is_note]

        # Keep the target event topology; shift only performance attributes.
        decoder = torch.empty_like(encoder_tokens)
        decoder[0] = torch.tensor([BeatTokenSpec.bos_type, 0, 0, 0, 0, 0])
        if len(events) > 1:
            previous_type = event_type[:-1]
            previous_is_note = previous_type == BeatTokenSpec.note_type
            # Decoder history is always absolute-log, even when the output
            # head predicts score-relative log deviation.
            previous_duration = perf_duration_input[:-1].clone()
            previous_onset = perf_onset[:-1]
            previous_velocity = perf_velocity[:-1]
            previous_pitch = pitch[:-1]
            previous_measure = measure_start[:-1]
            decoder[1:] = torch.stack((previous_type, previous_pitch, previous_duration,
                                       previous_onset, previous_velocity, previous_measure), dim=-1)
        return {
            "encoder_tokens": encoder_tokens,
            "decoder_tokens": decoder,
            "event_types": event_type,
            "labels": labels,
        }


class DINRBeatCollator:
    def __init__(self, fixed_length=None):
        self.fixed_length = fixed_length

    def __call__(self, examples):
        batch_size = len(examples)
        max_len = max(example["encoder_tokens"].shape[0] for example in examples)
        if self.fixed_length is not None:
            if max_len > self.fixed_length:
                raise ValueError(f'Event window {max_len} exceeds fixed length {self.fixed_length}')
            max_len = self.fixed_length
        else:
            max_len = ((max_len + 7) // 8) * 8
        encoder = torch.zeros((batch_size, max_len, 6), dtype=torch.long)
        decoder = torch.zeros_like(encoder)
        event_types = torch.full((batch_size, max_len), BeatTokenSpec.pad_type, dtype=torch.long)
        labels = torch.full((batch_size, max_len, 4), -100, dtype=torch.long)
        attention = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, example in enumerate(examples):
            length = example["encoder_tokens"].shape[0]
            encoder[row, :length] = example["encoder_tokens"]
            decoder[row, :length] = example["decoder_tokens"]
            event_types[row, :length] = example["event_types"]
            labels[row, :length] = example["labels"]
            attention[row, :length] = 1
        return {
            "encoder_tokens": encoder,
            "decoder_tokens": decoder,
            "event_types": event_types,
            "labels": labels,
            "attention_mask": attention,
            "decoder_attention_mask": attention,
        }
