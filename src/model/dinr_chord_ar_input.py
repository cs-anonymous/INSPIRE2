"""Continuous score inputs for the chord autoregressive timing ablation."""

import math

import torch
from torch import nn

from src.model.dinr_chord_ar import DINRChordARModel

SCORE_INPUTS = ('raw_separate', 'normalized_separate', 'normalized_joint')
LOG_SCORE_INPUTS = ('log_raw_separate', 'log_normalized_separate', 'log_normalized_joint')
VELOCITY_SCORE_INPUTS = ('velocity_tied', 'velocity_untied', 'log_separate_wide')
FIELD_HEAD_INPUTS = ('log_separate_256', 'lookup_field_mlp', 'log_separate_256_field_mlp')
TIMING_INPUT_MAX_MS = 5000.0
VELOCITY_INPUT_MAX = 127.0


def score_mlp(in_features, width, hidden):
    return nn.Sequential(nn.Linear(in_features, width), nn.GELU(), nn.Linear(width, hidden))


def uses_raw_score(score_input):
    return bool(score_input) and score_input != 'lookup_field_mlp'


class ResidualFieldHead(nn.Module):
    def __init__(self, classifier, width):
        super().__init__()
        hidden = classifier.in_features
        self.mlp = score_mlp(hidden, width, hidden)
        self.classifier = classifier
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden):
        return self.classifier(hidden + self.mlp(hidden))


class DINRChordARInputModel(DINRChordARModel):
    def __init__(self, score_input=None):
        super().__init__()
        if score_input not in (None, *SCORE_INPUTS, *LOG_SCORE_INPUTS, *VELOCITY_SCORE_INPUTS, *FIELD_HEAD_INPUTS):
            raise ValueError(f'Invalid score input: {score_input}')
        self.score_input = score_input
        score_input = score_input or ''
        self.score_lookup = not uses_raw_score(score_input)
        self.log_timing = score_input in (*LOG_SCORE_INPUTS, *VELOCITY_SCORE_INPUTS, *FIELD_HEAD_INPUTS)
        self.normalized = ('normalized' in score_input or score_input in VELOCITY_SCORE_INPUTS or
                           (score_input in FIELD_HEAD_INPUTS and not self.score_lookup))
        self.joint = score_input.endswith('_joint')
        self.velocity_tied = score_input == 'velocity_tied'
        self.velocity_lookup = self.score_lookup or score_input in ('velocity_tied', 'velocity_untied')
        self.score_mlp_width = 192 if self.joint else (128 if score_input == 'log_separate_wide' else 64)
        if score_input in FIELD_HEAD_INPUTS:
            self.score_mlp_width = 0 if self.score_lookup else 256
        self.field_head_width = 256 if score_input.endswith('_field_mlp') else 0
        if not self.score_lookup:
            self.build_score_mlps()
        if self.field_head_width:
            self.ioi_head = ResidualFieldHead(self.ioi_head, self.field_head_width)
            self.duration_head = ResidualFieldHead(self.duration_head, self.field_head_width)
            self.velocity_head = ResidualFieldHead(self.velocity_head, self.field_head_width)

    def build_score_mlps(self):
        # Build the common layers with the lookup model's seed and module order.
        del self.ioi_embedding, self.duration_embedding
        if self.velocity_lookup:
            if self.velocity_tied:
                self.velocity_head.weight = self.velocity_embedding.weight
            else:
                with torch.no_grad():
                    self.velocity_head.weight.copy_(self.velocity_embedding.weight)
        else:
            del self.velocity_embedding
        hidden = self.ioi_head.in_features
        if self.joint:
            self.score_mlp = score_mlp(3, self.score_mlp_width, hidden)
        else:
            slots = 2 if self.velocity_lookup else 3
            self.score_mlps = nn.ModuleList(score_mlp(1, self.score_mlp_width, hidden) for _ in range(slots))

    def numerical_input(self, score):
        fields = score.float()
        timing, velocity = fields[..., :2], fields[..., 2:]
        maximum = TIMING_INPUT_MAX_MS
        if self.log_timing:
            timing = timing.clamp_min(1).log()
            maximum = math.log(maximum)
        if self.normalized:
            timing = timing.clamp(0, maximum) / maximum
            velocity = velocity.clamp(0, VELOCITY_INPUT_MAX) / VELOCITY_INPUT_MAX
        return torch.cat((timing, velocity), dim=-1)

    def score_embeddings(self, pitches, score):
        pitch = self.pitch(pitches.float())
        if self.score_lookup:
            ioi, duration, velocity = score.unbind(-1)
            return pitch, self.ioi_embedding(ioi), self.duration_embedding(duration), self.velocity_embedding(velocity)
        fields = self.numerical_input(score)
        if self.joint:
            return (pitch, self.score_mlp(fields))
        if self.velocity_lookup:
            velocity = score[..., 2].round().clamp(0, 127).long()
            return pitch, self.score_mlps[0](fields[..., :1]), self.score_mlps[1](fields[..., 1:2]), self.velocity_embedding(velocity)
        return (pitch, *(mlp(value) for mlp, value in zip(self.score_mlps, fields.split(1, dim=-1), strict=True)))

    def hidden(self, pitches, score, attention, pitch_mask=None):
        if self.score_lookup:
            return super().hidden(pitches, score, attention)
        embeddings = self.score_embeddings(pitches, score)
        if pitch_mask is not None:
            pitch, *numerical = embeddings
            embeddings = (pitch.masked_fill(pitch_mask.unsqueeze(-1), 0.), *numerical)
        inputs = sum(embeddings)
        return self.encoder(inputs_embeds=inputs, attention_mask=attention.long()).last_hidden_state

    @torch.no_grad()
    def score_rms(self, batch):
        pitches, score, attention, _, _ = batch
        names = ('pitch', 'joint') if self.joint else ('pitch', 'ioi', 'duration', 'velocity')
        vectors = self.score_embeddings(pitches[attention], score[attention])
        rms = torch.stack([x.float().square().mean().sqrt() for x in vectors]).tolist()
        return dict(zip(names, rms, strict=True))
