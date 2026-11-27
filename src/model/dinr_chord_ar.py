"""Autoregressive chord decoder conditioned on aligned score encoder states."""

import torch
from torch import nn

from src.model.dinr_chord import DINRChordModel, median_category


class DINRChordARModel(DINRChordModel):
    history_mask_probability = 0.0
    history_mask_rng = None
    mask_history_pitch = False

    def __init__(self):
        super().__init__()
        self.history_mask_embedding = None
        self.history_mask_fields = None
        hidden = self.ioi_head.in_features
        sizes = (self.ioi_head.out_features, self.duration_head.out_features, self.velocity_head.out_features)
        self.feedback_tables = nn.ModuleList(nn.Embedding(size + 2, 128) for size in sizes)
        self.feedback_projection = nn.Linear(3 * 128, hidden, bias=False)
        self.bos = nn.Parameter(torch.zeros(hidden))
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)

    def feedback(self, tokens, advances, selected=None):
        inputs = []
        for slot, table in enumerate(self.feedback_tables):
            ids = tokens[..., slot]
            missing = table.num_embeddings - 1
            if slot == 0:
                ids = torch.where(advances, ids, missing - 1)
            embedding = table(ids.masked_fill(ids < 0, missing))
            if selected is not None and self.history_mask_fields is not None:
                embedding = torch.where(selected, self.history_mask_fields[slot], embedding)
            inputs.append(embedding)
        history = self.feedback_projection(torch.cat(inputs, dim=-1))
        if selected is not None and self.history_mask_fields is None:
            if self.history_mask_embedding is None:
                return history.masked_fill(selected, 0.)
            return torch.where(selected, self.history_mask_embedding.to(dtype=history.dtype), history)
        return history

    def losses(self, batch):
        pitches, score, attention, advances, labels = batch
        hidden = self.teacher_forced_hidden(pitches, score, attention, advances, labels)
        return self.head_losses(hidden, attention, advances, labels)

    def teacher_forced_hidden(self, pitches, score, attention, advances, labels, context=None):
        selected = None
        if self.training and self.history_mask_probability:
            valid = attention[:, :-1] & attention[:, 1:]
            draw = torch.rand(valid.shape, device=pitches.device, generator=self.history_mask_rng)
            selected = valid & (draw < self.history_mask_probability)
        if selected is not None and self.mask_history_pitch:
            pitch_mask = nn.functional.pad(selected, (0, 1))
            context = self.hidden(pitches, score, attention, pitch_mask=pitch_mask)
        elif context is None:
            context = self.hidden(pitches, score, attention)
        history = self.feedback(labels[:, :-1], advances[:, :-1],
                                None if selected is None else selected.unsqueeze(-1))
        shifted = torch.cat((self.bos.expand(len(labels), 1, -1), history), dim=1)
        decoded, _ = self.decoder(context + shifted)
        return context + decoded

    @torch.inference_mode()
    def sample_windows(self, pitches, score, attention, advances, rng, temperature, top_p, return_locations=False):
        context = self.hidden(pitches, score, attention)
        outputs, locations = [], []
        history = self.bos.expand(len(score), 1, -1)
        state = None
        for index in range(score.shape[1]):
            current = context[:, index:index + 1]
            decoded, state = self.decoder(current + history, state)
            hidden = (current + decoded).squeeze(1)
            tokens, timing_locations = self.decode_step(hidden, rng, temperature, top_p, advances[:, index])
            tokens[:, 0] = tokens[:, 0].masked_fill(~advances[:, index], 0)
            outputs.append(tokens)
            history = self.feedback(tokens, advances[:, index]).unsqueeze(1)
            if return_locations:
                if timing_locations is None:
                    raise ValueError('This timing head has no continuous location')
                locations.append(timing_locations)
        if return_locations:
            return torch.stack(outputs, dim=1), torch.stack(locations, dim=1)
        return torch.stack(outputs, dim=1)

    def decode_step(self, hidden, rng, temperature, top_p, advances=None):
        tokens = torch.stack([sample_head(head(hidden), rng, temperature, top_p)
                              for head in (self.ioi_head, self.duration_head, self.velocity_head)], dim=-1)
        return tokens, None


def sample_head(logits, rng, temperature, top_p):
    if rng is None:
        return median_category(logits)
    ordered, categories = (logits.float() / temperature).sort(descending=True)
    probabilities = ordered.softmax(-1)
    valid = probabilities.cumsum(-1) - probabilities < top_p
    picked = torch.multinomial(probabilities * valid, 1, generator=rng)
    return categories.gather(-1, picked).squeeze(-1)
