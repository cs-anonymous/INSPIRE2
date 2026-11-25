"""Compact timing vocabularies and discretized logistic heads for chord AR."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.data_process.dinr_chord_dataset import (COARSE_TIMING_BINS, LOG_DEV_HIGH, LOG_DEV_LOW, TIMING_BINS,
                                                log_dev_edges)
from src.model.dinr_chord_ar import DINRChordARModel, sample_head

OUTPUT_MODES = ('cat500', 'timing_dlm', 'all_dlm')
DIRECT_DLM_MODES = ('timing_dlm_k3_direct', 'timing_dlm_k1_direct', 'all_dlm_k1_direct')


class DINRChordAROutputModel(DINRChordARModel):
    def __init__(self, output_mode):
        super().__init__()
        if output_mode not in (*OUTPUT_MODES, *DIRECT_DLM_MODES):
            raise ValueError(f'Invalid chord output: {output_mode}')
        self.output_mode = output_mode
        self.direct_location = output_mode in DIRECT_DLM_MODES
        self.components = 3 if output_mode == 'timing_dlm_k3_direct' else 1
        self.dlm_slots = {'cat500': (), 'timing_dlm': (0, 1), 'all_dlm': (0, 1, 2),
                          'timing_dlm_k3_direct': (0, 1), 'timing_dlm_k1_direct': (0, 1),
                          'all_dlm_k1_direct': (0, 1, 2)}[output_mode]
        self.register_buffer('timing_edges', log_dev_edges('cpu'), persistent=False)
        self.register_buffer('velocity_edges', torch.arange(129, dtype=torch.float32) - 0.5, persistent=False)
        representatives = ((torch.arange(COARSE_TIMING_BINS) + 0.5) * (TIMING_BINS - 1) /
                           COARSE_TIMING_BINS).round().long()
        feedback_rows = torch.cat((representatives, torch.tensor([TIMING_BINS, TIMING_BINS + 1])))
        for slot, name in enumerate(('ioi_head', 'duration_head', 'velocity_head')):
            head = getattr(self, name)
            if slot in self.dlm_slots:
                outputs = 2 if self.components == 1 else 3 * self.components
                setattr(self, name, nn.Linear(head.in_features, outputs))
            elif slot < 2:
                reduced = nn.Linear(head.in_features, COARSE_TIMING_BINS)
                with torch.no_grad():
                    reduced.weight.copy_(head.weight[representatives])
                    reduced.bias.copy_(head.bias[representatives])
                setattr(self, name, reduced)
            if slot < 2:
                selected = self.feedback_tables[slot].weight.detach()[feedback_rows]
                self.feedback_tables[slot] = nn.Embedding.from_pretrained(selected, freeze=False)

    @torch.no_grad()
    def initialize_from_training(self, payload):
        score = payload['score']
        targets = payload['targets']
        valid = payload['ioi_valid'] & (score[:, 0] > 0)
        iois = targets[valid, 0].clamp_min(1).log() - score[valid, 0].clamp_min(1).log()
        durations = targets[:, 1].clamp_min(1).log() - score[:, 1].clamp_min(1).log()
        values = (iois, durations, targets[:, 2].round().clamp(0, 127))
        fields = ('ioi', 'duration', 'velocity')
        report = {}
        for slot, head in enumerate((self.ioi_head, self.duration_head, self.velocity_head)):
            if slot not in self.dlm_slots:
                continue
            groups = values[slot].sort().values.tensor_split(self.components)
            quantiles = torch.stack([torch.quantile(group, torch.tensor([0.25, 0.5, 0.75])) for group in groups])
            minimum = scale_floor(slot)
            locations = quantiles[:, 1]
            scales = ((quantiles[:, 2] - quantiles[:, 0]) / (2 * math.log(3))).clamp_min(2 * minimum)
            positive = scales - minimum
            raw_scales = positive + torch.log(-torch.expm1(-positive))
            weights = torch.tensor([len(group) / len(values[slot]) for group in groups])
            bias = torch.cat((locations, raw_scales))
            if self.components > 1:
                bias = torch.cat((weights.log(), bias))
            head.weight.zero_()
            head.bias.copy_(bias)
            report[fields[slot]] = dict(valid_train_chords=len(values[slot]), location=locations.tolist(),
                                        scale=scales.tolist(), weight=weights.tolist())
        return report

    def field_loss(self, slot, logits, targets):
        if slot not in self.dlm_slots:
            return super().field_loss(slot, logits, targets)
        if self.components > 1:
            log_weights, locations, scales, edges = self.mixture_parameters(slot, logits)
            masses = logistic_bin_mass(locations, scales, edges, targets[:, None])
            return -torch.logsumexp(log_weights + masses, dim=-1).sum()
        loc, scale, edges = self.distribution(slot, logits)
        return -logistic_bin_mass(loc, scale, edges, targets).sum()

    def distribution(self, slot, logits):
        raw_loc, raw_scale = logits.float().unbind(-1)
        if slot < 2:
            low, high, edges = LOG_DEV_LOW, LOG_DEV_HIGH, self.timing_edges
        else:
            low, high, edges = 0., 127., self.velocity_edges
        loc = raw_loc if self.direct_location else low + (high - low) * raw_loc.sigmoid()
        return loc, F.softplus(raw_scale) + scale_floor(slot), edges

    def mixture_parameters(self, slot, logits):
        log_weights, locations, raw_scales = logits.float().chunk(3, dim=-1)
        edges = self.timing_edges if slot < 2 else self.velocity_edges
        return log_weights.log_softmax(-1), locations, F.softplus(raw_scales) + scale_floor(slot), edges

    def decode_step(self, hidden, rng, temperature, top_p):
        tokens, timing_locations = [], []
        for slot, head in enumerate((self.ioi_head, self.duration_head, self.velocity_head)):
            logits = head(hidden)
            if slot not in self.dlm_slots:
                tokens.append(sample_head(logits, rng, temperature, top_p))
                continue
            if self.components > 1:
                token, loc = self.decode_mixture(slot, logits, rng, temperature, top_p)
            else:
                loc, scale, edges = self.distribution(slot, logits)
                if rng is None:
                    token = torch.bucketize(loc.contiguous(), edges[1:-1], right=False)
                else:
                    bins = torch.arange(len(edges) - 1, device=edges.device)
                    masses = logistic_bin_mass(loc[:, None], scale[:, None], edges, bins)
                    token = sample_head(masses, rng, temperature, top_p)
            tokens.append(token)
            if slot < 2:
                timing_locations.append(loc)
        locations = torch.stack(timing_locations, dim=-1) if timing_locations else None
        return torch.stack(tokens, dim=-1), locations

    def decode_mixture(self, slot, logits, rng, temperature, top_p):
        log_weights, locs, scales, edges = self.mixture_parameters(slot, logits)
        if rng is not None:
            bins = torch.arange(len(edges) - 1, device=edges.device)
            masses = logistic_bin_mass(locs[:, None, :], scales[:, None, :], edges, bins[:, None])
            tokens = sample_head(torch.logsumexp(log_weights[:, None, :] + masses, dim=-1), rng, temperature, top_p)
            return tokens, (locs * log_weights.exp()).sum(-1)

        weights = log_weights.exp()
        cdf = ((edges[1:-1, None] - locs[:, None, :]) / scales[:, None, :]).sigmoid()
        tokens = ((cdf * weights[:, None, :]).sum(-1) < 0.5).sum(-1)
        left, right = edges[tokens], edges[tokens + 1]
        for _ in range(10):
            mid = (left + right) * 0.5
            mass = (((mid[:, None] - locs) / scales).sigmoid() * weights).sum(-1)
            lower = mass < 0.5
            left, right = torch.where(lower, mid, left), torch.where(lower, right, mid)
        return tokens, (left + right) * 0.5


def scale_floor(slot):
    return 0.005 if slot < 2 else 0.5


def logistic_bin_mass(loc, scale, edges, bins):
    """Log probability of a bin, with outer tails assigned to the edge bins."""
    left, right = edges[bins], edges[bins + 1]
    z_left = (left - loc) / scale
    z_right = (right - loc) / scale
    difference = torch.log(-torch.expm1((left - right) / scale))
    inner = F.logsigmoid(z_right) + F.logsigmoid(-z_left) + difference
    return torch.where(bins == 0, F.logsigmoid(z_right),
                       torch.where(bins == len(edges) - 2, F.logsigmoid(-z_left), inner))
