"""Score-only chord model with three parallel categorical heads."""

import torch
from torch import nn
from torch.nn import functional as F

from src.data_process.dinr_chord_dataset import TIMING_BINS
from src.model.dinr import DINRNoteEncoder
from src.model.pianoformer import PianoT5GemmaConfig


def median_category(logits):
    return (logits.float().softmax(-1).cumsum(-1) < .5).sum(-1)


class DINRChordModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = PianoT5GemmaConfig(encoder_layers_num=10, attn_implementation='sdpa')
        config.encoder._attn_implementation = 'sdpa'
        self.encoder = DINRNoteEncoder(config.encoder)
        hidden = config.hidden_size
        self.pitch = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, hidden))
        self.ioi_embedding = nn.Embedding(TIMING_BINS, hidden)
        self.duration_embedding = nn.Embedding(TIMING_BINS, hidden)
        self.velocity_embedding = nn.Embedding(128, hidden)
        self.ioi_head = nn.Linear(hidden, TIMING_BINS)
        self.duration_head = nn.Linear(hidden, TIMING_BINS)
        self.velocity_head = nn.Linear(hidden, 128)
        for layer in (self.ioi_embedding, self.duration_embedding, self.velocity_embedding):
            nn.init.normal_(layer.weight, std=config.initializer_range)

    def hidden(self, pitches, score, attention):
        ioi, duration, velocity = score.unbind(-1)
        inputs = (self.pitch(pitches.float()) + self.ioi_embedding(ioi)
                  + self.duration_embedding(duration) + self.velocity_embedding(velocity))
        return self.encoder(inputs_embeds=inputs, attention_mask=attention.long()).last_hidden_state

    def losses(self, batch):
        pitches, score, attention, advances, labels = batch
        hidden = self.hidden(pitches, score, attention)
        return self.head_losses(hidden, attention, advances, labels)

    def head_losses(self, hidden, attention, advances, labels, velocity_state=None):
        ioi = attention & advances & (labels[..., 0] != -100)
        active = (ioi, attention, attention)
        sums = []
        for slot, (head, mask) in enumerate(zip((self.ioi_head, self.duration_head, self.velocity_head), active,
                                                strict=True)):
            inputs = velocity_state if slot == 2 and velocity_state is not None else hidden[mask]
            sums.append(self.field_loss(slot, head(inputs), labels[..., slot][mask]))
        return torch.stack(sums), torch.stack([mask.sum() for mask in active])

    def field_loss(self, slot, logits, targets):
        return F.cross_entropy(logits, targets, reduction='sum')

    def predict(self, batch, choose=median_category):
        pitches, score, attention, advances, _ = batch
        hidden = self.hidden(pitches, score, attention)
        pred = torch.zeros((*attention.shape, 3), device=pitches.device, dtype=torch.long)
        active = (attention & advances, attention, attention)
        for slot, (head, mask) in enumerate(zip((self.ioi_head, self.duration_head, self.velocity_head), active,
                                                strict=True)):
            pred[..., slot][mask] = choose(head(hidden[mask]))
        return pred
