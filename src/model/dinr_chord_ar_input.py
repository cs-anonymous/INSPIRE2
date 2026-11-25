"""Continuous score inputs for the chord autoregressive timing ablation."""

import torch
from torch import nn

from src.model.dinr_chord_ar import DINRChordARModel

SCORE_INPUTS = ('raw_separate', 'normalized_separate', 'normalized_joint')
TIMING_INPUT_MAX_MS = 5000.0
VELOCITY_INPUT_MAX = 127.0


def score_mlp(in_features, width, hidden):
    return nn.Sequential(nn.Linear(in_features, width), nn.GELU(), nn.Linear(width, hidden))


class DINRChordARInputModel(DINRChordARModel):
    def __init__(self, score_input):
        super().__init__()
        if score_input not in SCORE_INPUTS:
            raise ValueError(f'Invalid score input: {score_input}')
        self.score_input = score_input
        # Build the common layers with the lookup model's seed and module order.
        del self.ioi_embedding, self.duration_embedding, self.velocity_embedding
        hidden = self.ioi_head.in_features
        if score_input == 'normalized_joint':
            self.score_mlp = score_mlp(3, 192, hidden)
        else:
            self.score_mlps = nn.ModuleList(score_mlp(1, 64, hidden) for _ in range(3))

    def score_embeddings(self, pitches, score):
        fields = score.float()
        if self.score_input != 'raw_separate':
            timing = fields[..., :2].clamp(0, TIMING_INPUT_MAX_MS) / TIMING_INPUT_MAX_MS
            velocity = fields[..., 2:].clamp(0, VELOCITY_INPUT_MAX) / VELOCITY_INPUT_MAX
            fields = torch.cat((timing, velocity), dim=-1)
        pitch = self.pitch(pitches.float())
        if self.score_input == 'normalized_joint':
            return (pitch, self.score_mlp(fields))
        return (pitch, *(mlp(value) for mlp, value in zip(self.score_mlps, fields.split(1, dim=-1), strict=True)))

    def hidden(self, pitches, score, attention):
        inputs = sum(self.score_embeddings(pitches, score))
        return self.encoder(inputs_embeds=inputs, attention_mask=attention.long()).last_hidden_state

    @torch.no_grad()
    def score_rms(self, batch):
        pitches, score, attention, _, _ = batch
        names = ('pitch', 'joint') if self.score_input == 'normalized_joint' else ('pitch', 'ioi', 'duration', 'velocity')
        vectors = self.score_embeddings(pitches[attention], score[attention])
        rms = torch.stack([x.float().square().mean().sqrt() for x in vectors]).tolist()
        return dict(zip(names, rms, strict=True))
