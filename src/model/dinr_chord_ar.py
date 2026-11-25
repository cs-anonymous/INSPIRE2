"""Autoregressive chord decoder conditioned on aligned score encoder states."""

import torch
from torch import nn

from src.model.dinr_chord import DINRChordModel, median_category


class DINRChordARModel(DINRChordModel):
    def __init__(self):
        super().__init__()
        hidden = self.ioi_head.in_features
        sizes = (self.ioi_head.out_features, self.duration_head.out_features, self.velocity_head.out_features)
        self.feedback_tables = nn.ModuleList(nn.Embedding(size + 2, 128) for size in sizes)
        self.feedback_projection = nn.Linear(3 * 128, hidden, bias=False)
        self.bos = nn.Parameter(torch.zeros(hidden))
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)

    def feedback(self, tokens, advances):
        inputs = []
        for slot, table in enumerate(self.feedback_tables):
            ids = tokens[..., slot]
            missing = table.num_embeddings - 1
            if slot == 0:
                ids = torch.where(advances, ids, missing - 1)
            inputs.append(table(ids.masked_fill(ids < 0, missing)))
        return self.feedback_projection(torch.cat(inputs, dim=-1))

    def losses(self, batch):
        pitches, score, attention, advances, labels = batch
        context = self.hidden(pitches, score, attention)
        history = self.feedback(labels[:, :-1], advances[:, :-1])
        shifted = torch.cat((self.bos.expand(len(labels), 1, -1), history), dim=1)
        decoded, _ = self.decoder(context + shifted)
        return self.head_losses(context + decoded, attention, advances, labels)

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
            tokens, timing_locations = self.decode_step(hidden, rng, temperature, top_p)
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

    def decode_step(self, hidden, rng, temperature, top_p):
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
