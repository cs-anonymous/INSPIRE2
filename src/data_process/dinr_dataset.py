"""Note-level DINR dataset backed by the grouped-median ASAP sidecars.

The sidecar is the canonical data representation.  Absolute DINR targets are
created at load time so the same sidecar can later support log-relative CINR.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


ASAP_SIDECAR_NAME = "score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt"


class DINRTokenSpec:
    """Compact PT-style categorical vocabulary used by note-level DINR."""

    pad_id = 0
    bos_id = 2
    mask_id = 1
    pitch_start = 5
    pitch_bins = 128
    velocity_bins = 128
    timing_bins = 500
    timing_bin_ms = 10
    max_timing_ms = timing_bins * timing_bin_ms
    velocity_start = pitch_start + pitch_bins
    timing_start = velocity_start + velocity_bins
    pedal_start = timing_start + timing_bins
    vocab_size = pedal_start + 2

    @classmethod
    def note_to_tokens(cls, pitch, ioi, duration, velocity, pedal):
        timing_i = int(round(float(ioi) / cls.timing_bin_ms))
        timing_d = int(round(float(duration) / cls.timing_bin_ms))
        return [
            cls.pitch_start + min(max(int(round(float(pitch))), 0), cls.pitch_bins - 1),
            cls.timing_start + min(max(timing_i, 0), cls.timing_bins - 1),
            cls.velocity_start + min(max(int(round(float(velocity))), 0), cls.velocity_bins - 1),
            cls.timing_start + min(max(timing_d, 0), cls.timing_bins - 1),
            *[cls.pedal_start + (1 if int(value) else 0) for value in pedal[:4]],
        ]

    @classmethod
    def log_time_bin(cls, milliseconds):
        """Map log1p(milliseconds) from [0, 5000] ms to 500 categorical bins."""
        import math
        value = min(max(float(milliseconds), 0.0), float(cls.max_timing_ms))
        unit = math.log1p(value) / math.log1p(cls.max_timing_ms)
        return int(round(unit * (cls.timing_bins - 1)))

    @classmethod
    def log_time_ms(cls, category):
        import math
        unit = min(max(int(category), 0), cls.timing_bins - 1) / (cls.timing_bins - 1)
        return math.expm1(unit * math.log1p(cls.max_timing_ms))

    @classmethod
    def tokens_to_local_targets(cls, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.shape[-1] not in (4, 8):
            raise ValueError(f"Expected 4 or 8 note slots, got {tokens.shape[-1]}")
        targets = [
            tokens[..., 1] - cls.timing_start,
            tokens[..., 3] - cls.timing_start,
            tokens[..., 2] - cls.velocity_start,
        ]
        if tokens.shape[-1] == 8:
            targets.extend(tokens[..., index] - cls.pedal_start for index in range(4, 8))
        return torch.stack(targets, dim=-1)


def _sidecar_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob(ASAP_SIDECAR_NAME))
    if not paths:
        raise FileNotFoundError(
            f"No {ASAP_SIDECAR_NAME} files found under {root}. "
            "Build the grouped-median ASAP sidecars first."
        )
    return paths


def _as_float_rows(rows, width=3):
    result = []
    for row in rows:
        values = [float(value) for value in row[:width]]
        if len(values) != width:
            raise ValueError(f"Expected {width} values per raw row, got {row}")
        result.append(values)
    return result


def _performance_tokens(score, performance, include_pedal=True):
    score_pitch = score["pitch"]
    score_raw = _as_float_rows(score["score_raw"])
    shared = _as_float_rows(performance["label_shared_raw"])
    grouped = [float(value) for value in performance.get("grouped_ioi_raw", [])]
    pedal_rows = performance.get("label_pedal4_raw") or [[0, 0, 0, 0] for _ in shared]
    n = min(len(score_pitch), len(score_raw), len(shared), len(pedal_rows))
    if grouped and len(grouped) != n:
        raise ValueError("grouped_ioi_raw length does not match aligned performance rows")
    score_tokens = []
    target_tokens = []
    score_zero_ioi = []
    for index in range(n):
        s_ioi, s_dur, s_vel = score_raw[index]
        p_ioi, p_dur, p_vel = shared[index]
        # Positive score-IOI rows carry the grouped interval total; zero-score
        # rows carry the internal onset gap. This is the shared target layout
        # for DINR-absolute and the later score-relative CINR variant.
        target_ioi = grouped[index] if grouped else p_ioi
        score_note = DINRTokenSpec.note_to_tokens(
            score_pitch[index], s_ioi, s_dur, s_vel, [0, 0, 0, 0]
        )
        if not include_pedal:
            score_note = score_note[:4]
        score_tokens.append(score_note)
        score_zero_ioi.append(s_ioi <= 0.0)
        pedals = [1 if float(value) >= 64.0 else 0 for value in pedal_rows[index][:4]]
        target_note = DINRTokenSpec.note_to_tokens(
            score_pitch[index], target_ioi, p_dur, p_vel, pedals
        )
        if not include_pedal:
            target_note = target_note[:4]
        target_tokens.append(target_note)
    return score_tokens, target_tokens, score_zero_ioi


class DINRAbsoluteDataset(Dataset):
    """Windowed note-level DINR samples from normalized ASAP sidecars."""

    def __init__(self, sidecar_root, split="train", block_notes=512, overlap_ratio=0.5,
                 max_sidecars=None, max_performances=None, include_pedal=True,
                 precomputed_path=None, timing_target_mode="absolute", time_input_mode="linear"):
        self.root = Path(sidecar_root).resolve()
        self.split = str(split)
        self.block_notes = int(block_notes)
        self.overlap_ratio = float(overlap_ratio)
        self.include_pedal = bool(include_pedal)
        self.timing_target_mode = str(timing_target_mode).lower()
        if self.timing_target_mode not in {"absolute", "log_absolute", "score_relative_log"}:
            raise ValueError("timing_target_mode must be absolute, log_absolute, or score_relative_log")
        self.time_input_mode = str(time_input_mode).lower()
        if self.time_input_mode not in {"linear", "log_absolute"}:
            raise ValueError("time_input_mode must be linear or log_absolute")
        self.precomputed = None
        self._sidecar_cache = OrderedDict()
        self._token_cache = OrderedDict()
        if self.block_notes < 1:
            raise ValueError("block_notes must be positive")
        self.samples = []
        if precomputed_path is not None:
            self.precomputed = torch.load(precomputed_path, map_location="cpu", weights_only=False)
            if self.precomputed.get("split") != self.split:
                raise ValueError(
                    f"Precomputed token split {self.precomputed.get('split')} != requested {self.split}"
                )
            if self.precomputed.get("block_notes") != self.block_notes or self.precomputed.get("overlap_ratio") != self.overlap_ratio:
                raise ValueError("Precomputed token windows do not match block_notes/overlap_ratio")
            self.samples = self.precomputed["samples"]
            return
        sidecars = _sidecar_paths(self.root)
        if max_sidecars is not None:
            sidecars = sidecars[: int(max_sidecars)]
        performance_count = 0
        for sidecar in sidecars:
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            score = payload.get("score") or {}
            for performance_index, performance in enumerate(payload.get("performances", [])):
                if str(performance.get("split", "")) != self.split:
                    continue
                if max_performances is not None and performance_count >= int(max_performances):
                    break
                note_count = min(
                    len(score.get("pitch", [])),
                    len(score.get("score_raw", [])),
                    len(performance.get("label_shared_raw", [])),
                    len(performance.get("label_pedal4_raw", [])),
                )
                self._add_windows(sidecar, performance_index, note_count)
                performance_count += 1
            del payload
            if max_performances is not None and performance_count >= int(max_performances):
                break
        if not self.samples:
            raise ValueError(f"No DINR samples found for split={self.split} under {self.root}")

    def _add_windows(self, sidecar, performance_index, total):
        if total == 0:
            return
        stride = max(1, int(round(self.block_notes * (1.0 - self.overlap_ratio))))
        start = 0
        while start < total:
            end = min(total, start + self.block_notes)
            if end - start < min(8, self.block_notes) and self.samples:
                break
            self.samples.append((str(sidecar), int(performance_index), start, end))
            if end == total:
                break
            start += stride

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _quantize_unit_interval(value, low, high, bins):
        value = torch.as_tensor(value, dtype=torch.float32).clamp(low, high)
        return torch.round((value - low) * ((bins - 1) / (high - low))).long()

    def _encode_relative_targets(self, score_tokens, target_tokens, zero_mask):
        """Replace absolute timing labels with score-relative log categories."""
        if self.timing_target_mode == "absolute":
            return target_tokens
        timing_start = DINRTokenSpec.timing_start
        score_ioi_ms = (score_tokens[:, 1] - timing_start).clamp_min(0).float() * DINRTokenSpec.timing_bin_ms
        target_ioi_ms = (target_tokens[:, 1] - timing_start).clamp_min(0).float() * DINRTokenSpec.timing_bin_ms
        score_duration_ms = (score_tokens[:, 3] - timing_start).clamp_min(0).float() * DINRTokenSpec.timing_bin_ms
        target_duration_ms = (target_tokens[:, 3] - timing_start).clamp_min(0).float() * DINRTokenSpec.timing_bin_ms

        ioi_log = torch.log(target_ioi_ms.clamp_min(1.0))
        ioi_relative = ioi_log - torch.log(score_ioi_ms.clamp_min(1.0))
        positive_ioi_bins = self._quantize_unit_interval(ioi_relative, -3.0, 2.0, DINRTokenSpec.timing_bins)
        chord_ioi_bins = self._quantize_unit_interval(ioi_log, 0.0, 5.0, DINRTokenSpec.timing_bins)
        if self.timing_target_mode == "log_absolute":
            encoded = target_tokens.clone()
            encoded[:, 1] = timing_start + torch.tensor(
                [DINRTokenSpec.log_time_bin(x) for x in target_ioi_ms.tolist()], dtype=torch.long
            )
            encoded[:, 3] = timing_start + torch.tensor(
                [DINRTokenSpec.log_time_bin(x) for x in target_duration_ms.tolist()], dtype=torch.long
            )
            return encoded
        duration_relative = torch.log(target_duration_ms.clamp_min(1.0)) - torch.log(score_duration_ms.clamp_min(1.0))
        duration_bins = self._quantize_unit_interval(duration_relative, -3.0, 2.0, DINRTokenSpec.timing_bins)

        encoded = target_tokens.clone()
        encoded[:, 1] = timing_start + torch.where(zero_mask.bool(), chord_ioi_bins, positive_ioi_bins)
        encoded[:, 3] = timing_start + duration_bins
        return encoded

    @staticmethod
    def _encode_input_times(tokens, mode):
        if mode == "linear":
            return tokens
        result = tokens.clone()
        start = DINRTokenSpec.timing_start
        for slot in (1, 3):
            ms = (tokens[:, slot] - start).clamp_min(0).float() * DINRTokenSpec.timing_bin_ms
            bins = torch.tensor([DINRTokenSpec.log_time_bin(x) for x in ms.tolist()], dtype=torch.long)
            result[:, slot] = start + bins
        return result

    def __getitem__(self, index):
        if self.precomputed is not None:
            start, end = self.samples[index]
            score = self.precomputed["score_tokens"][start:end].long().clone()
            targets = self.precomputed["target_tokens"][start:end].long().clone()
            zero_mask = self.precomputed["score_zero_ioi_mask"][start:end]
            if not self.include_pedal:
                score = score[:, :4]
                targets = targets[:, :4]
            decoder_source = targets.clone()
            targets = self._encode_relative_targets(score, targets, zero_mask)
            score = self._encode_input_times(score, self.time_input_mode)
            if self.timing_target_mode == "score_relative_log":
                decoder_source = self._encode_input_times(decoder_source, "log_absolute")
            else:
                decoder_source = targets
            decoder_tokens = torch.full_like(targets, DINRTokenSpec.bos_id)
            if len(targets) > 1:
                decoder_tokens[1:] = decoder_source[:-1]
            labels = DINRTokenSpec.tokens_to_local_targets(targets)
            if not self.include_pedal:
                labels[..., 3:] = -100
            return {
                "score_tokens": score,
                "decoder_tokens": decoder_tokens,
                "labels": labels,
                "score_zero_ioi_mask": zero_mask,
                "attention_mask": torch.ones(score.shape[0], dtype=torch.long),
            }
        sidecar, performance_index, start, end = self.samples[index]
        cache_key = (sidecar, performance_index)
        if cache_key not in self._token_cache:
            if sidecar not in self._sidecar_cache:
                payload = torch.load(sidecar, map_location="cpu", weights_only=False)
                self._sidecar_cache[sidecar] = payload
                while len(self._sidecar_cache) > 2:
                    self._sidecar_cache.popitem(last=False)
            payload = self._sidecar_cache[sidecar]
            score_tokens, target_tokens, score_zero_ioi = _performance_tokens(
                payload["score"], payload["performances"][performance_index],
                include_pedal=self.include_pedal,
            )
            self._token_cache[cache_key] = (score_tokens, target_tokens, score_zero_ioi)
            while len(self._token_cache) > 4:
                self._token_cache.popitem(last=False)
        else:
            self._token_cache.move_to_end(cache_key)
        score_tokens, target_tokens, score_zero_ioi = self._token_cache[cache_key]
        score = torch.tensor(score_tokens[start:end], dtype=torch.long)
        target_tokens = torch.tensor(target_tokens[start:end], dtype=torch.long)
        zero_mask = torch.tensor(score_zero_ioi[start:end], dtype=torch.bool)
        decoder_source = target_tokens.clone()
        target_tokens = self._encode_relative_targets(score, target_tokens, zero_mask)
        score = self._encode_input_times(score, self.time_input_mode)
        if self.timing_target_mode == "score_relative_log":
            decoder_source = self._encode_input_times(decoder_source, "log_absolute")
        else:
            decoder_source = target_tokens
        slot_count = len(target_tokens[0]) if target_tokens else 4
        decoder_tokens = [[DINRTokenSpec.bos_id] * slot_count]
        decoder_tokens.extend(decoder_source[:-1].tolist())
        decoder = torch.tensor(decoder_tokens, dtype=torch.long)
        target = DINRTokenSpec.tokens_to_local_targets(target_tokens)
        if not self.include_pedal:
            target[..., 3:] = -100
        return {
            "score_tokens": score,
            "decoder_tokens": decoder,
            "labels": target,
            "score_zero_ioi_mask": zero_mask,
            "attention_mask": torch.ones(score.shape[0], dtype=torch.long),
        }


class DINRCollator:
    def __call__(self, examples):
        max_notes = max(example["score_tokens"].shape[0] for example in examples)
        batch_size = len(examples)
        slot_count = examples[0]["score_tokens"].shape[-1]
        label_count = examples[0]["labels"].shape[-1]
        score = torch.full((batch_size, max_notes, slot_count), DINRTokenSpec.pad_id, dtype=torch.long)
        decoder = torch.full_like(score, DINRTokenSpec.pad_id)
        labels = torch.full((batch_size, max_notes, label_count), -100, dtype=torch.long)
        zero_ioi = torch.zeros((batch_size, max_notes), dtype=torch.bool)
        mask = torch.zeros((batch_size, max_notes), dtype=torch.long)
        for row, example in enumerate(examples):
            length = example["score_tokens"].shape[0]
            score[row, :length] = example["score_tokens"]
            decoder[row, :length] = example["decoder_tokens"]
            labels[row, :length] = example["labels"]
            zero_ioi[row, :length] = example["score_zero_ioi_mask"]
            mask[row, :length] = 1
        return {
            "score_tokens": score,
            "decoder_tokens": decoder,
            "labels": labels,
            "score_zero_ioi_mask": zero_ioi,
            "attention_mask": mask,
            "decoder_attention_mask": mask,
        }
