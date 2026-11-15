"""Typed beat/note DINR model with one output position per event."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from src.data_process.dinr_beat_dataset import BeatTokenSpec
from src.model.dinr import DINRNoteEncoder
from src.model.pianoformer import PianoT5GemmaConfig, PianoT5GemmaModel


class DINRBeatModel(nn.Module):
    """Event model: BEAT and NOTE positions share the sequence topology."""

    def __init__(self, config=None, duration_target="absolute_log",
                 duration_input_tables="shared", duration_output_head="separate",
                 decoder_mask_probability=0.5):
        super().__init__()
        self.config = config or PianoT5GemmaConfig()
        if duration_target not in {"absolute_log", "log_dev"}:
            raise ValueError("duration_target must be absolute_log or log_dev")
        if duration_input_tables not in {"shared", "separate"}:
            raise ValueError("duration_input_tables must be shared or separate")
        if duration_output_head not in {"separate", "shared_conditioned", "shared"}:
            raise ValueError("Unknown duration_output_head")
        self.duration_target = duration_target
        self.beat_duration_target = duration_target
        self.note_duration_target = duration_target
        self.duration_input_tables = duration_input_tables
        self.duration_output_head = duration_output_head
        self.decoder_mask_probability = float(decoder_mask_probability)
        self.backbone = PianoT5GemmaModel(self.config)
        self.backbone.encoder = DINRNoteEncoder(self.config.encoder)
        hidden = self.config.hidden_size
        self.event_type_embedding = nn.Embedding(4, hidden, padding_idx=BeatTokenSpec.pad_type)
        self.pitch_embedding = nn.Embedding(BeatTokenSpec.pitch_bins, hidden)
        if duration_input_tables == "shared":
            self.duration_embedding = nn.Embedding(BeatTokenSpec.duration_bins, hidden)
            self.beat_duration_embedding = None
            self.note_duration_embedding = None
        else:
            self.duration_embedding = None
            self.beat_duration_embedding = nn.Embedding(BeatTokenSpec.duration_bins, hidden)
            self.note_duration_embedding = nn.Embedding(BeatTokenSpec.duration_bins, hidden)
        self.onset_embedding = nn.Embedding(BeatTokenSpec.onset_bins, hidden)
        self.velocity_embedding = nn.Embedding(BeatTokenSpec.velocity_bins, hidden)
        self.measure_start_embedding = nn.Embedding(2, hidden)
        decoder_hidden = self.config.decoder.hidden_size
        self.beat_duration_head = nn.Linear(decoder_hidden, BeatTokenSpec.duration_bins)
        self.note_duration_head = (
            nn.Linear(decoder_hidden, BeatTokenSpec.duration_bins)
            if duration_output_head == "separate" else None
        )
        self.shared_duration_head = (
            nn.Linear(decoder_hidden, BeatTokenSpec.duration_bins)
            if duration_output_head != "separate" else None
        )
        self.duration_type_bias = (
            nn.Parameter(torch.zeros(2, BeatTokenSpec.duration_bins))
            if duration_output_head == "shared_conditioned" else None
        )
        self.onset_head = nn.Linear(decoder_hidden, BeatTokenSpec.onset_bins)
        self.velocity_head = nn.Linear(decoder_hidden, BeatTokenSpec.velocity_bins)
        self.type_output_embedding = nn.Embedding(2, decoder_hidden)
        self.decoder_mask_embedding = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.decoder_mask_embedding, std=self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Embedding) and module not in self.backbone.modules():
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def _duration_features(self, event_type, duration_token):
        index = (duration_token % BeatTokenSpec.duration_bins).clamp(0, BeatTokenSpec.duration_bins - 1)
        if self.duration_input_tables == "shared":
            return self.duration_embedding(index)
        note = event_type == BeatTokenSpec.note_type
        result = self.beat_duration_embedding(index)
        return torch.where(note.unsqueeze(-1), self.note_duration_embedding(index), result)

    def _sum_events(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 6:
            raise ValueError("Beat event tokens must have shape [batch, events, 6]")
        event_type, pitch, duration, onset, velocity, measure = tokens.unbind(dim=-1)
        is_beat = event_type == BeatTokenSpec.beat_type
        is_note = event_type == BeatTokenSpec.note_type
        valid = is_beat | is_note | (event_type == BeatTokenSpec.bos_type)
        result = self.event_type_embedding(event_type.clamp(0, BeatTokenSpec.pad_type))
        result = result + self._duration_features(event_type, duration)
        note_features = (self.pitch_embedding(pitch.clamp(0, BeatTokenSpec.pitch_bins - 1))
                         + self.onset_embedding(onset.clamp(0, BeatTokenSpec.onset_bins - 1))
                         + self.velocity_embedding(velocity.clamp(0, BeatTokenSpec.velocity_bins - 1)))
        beat_features = self.measure_start_embedding(measure.clamp(0, 1))
        result = result + torch.where(is_note.unsqueeze(-1), note_features,
                                      torch.where(is_beat.unsqueeze(-1), beat_features,
                                                  torch.zeros_like(note_features)))
        return result * valid.unsqueeze(-1)

    @staticmethod
    def _masked_ce(logits, labels):
        valid = labels >= 0
        if not valid.any():
            return logits.new_zeros(())
        return F.cross_entropy(logits[valid], labels[valid])

    def forward(self, encoder_tokens, decoder_tokens, event_types, labels=None,
                attention_mask=None, decoder_attention_mask=None,
                encoder_outputs=None, use_cache=False, past_key_values=None,
                cache_position=None):
        if attention_mask is None:
            attention_mask = encoder_tokens.new_ones(encoder_tokens.shape[:2])
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask
        encoder_embeds = None if encoder_outputs is not None else self._sum_events(encoder_tokens)
        decoder_embeds = self._sum_events(decoder_tokens)
        if self.training and self.decoder_mask_probability:
            selected = torch.rand(decoder_tokens.shape[:2], device=decoder_tokens.device)
            selected = (selected < self.decoder_mask_probability) & (
                decoder_tokens[..., 0] < BeatTokenSpec.bos_type
            )
            decoder_embeds = torch.where(selected.unsqueeze(-1),
                                        self.decoder_mask_embedding, decoder_embeds)
        outputs = self.backbone(
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_embeds,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state
        note_hidden = hidden + self.type_output_embedding(
            torch.full_like(event_types, BeatTokenSpec.note_type)
        )
        beat_hidden = hidden + self.type_output_embedding(
            torch.full_like(event_types, BeatTokenSpec.beat_type)
        )
        # During cached generation encoder attention covers the complete score,
        # while event_types only covers the current decoder output positions.
        is_beat = event_types == BeatTokenSpec.beat_type
        is_note = event_types == BeatTokenSpec.note_type
        beat_duration = self.beat_duration_head(beat_hidden)
        if self.duration_output_head == "separate":
            note_duration = self.note_duration_head(note_hidden)
        else:
            shared = self.shared_duration_head(hidden)
            beat_duration = shared + (self.duration_type_bias[0] if self.duration_type_bias is not None else 0.0)
            note_duration = shared + (self.duration_type_bias[1] if self.duration_type_bias is not None else 0.0)
        duration_logits = torch.where(is_beat.unsqueeze(-1), beat_duration,
                                      torch.where(is_note.unsqueeze(-1), note_duration,
                                                  torch.zeros_like(beat_duration)))
        logits = {
            "duration": duration_logits,
            "beat_duration": beat_duration,
            "note_duration": note_duration,
            "onset": self.onset_head(note_hidden),
            "velocity": self.velocity_head(note_hidden),
        }
        loss = None
        if labels is not None:
            beat_labels = torch.where(is_beat, labels[..., 0], -100)
            note_duration_labels = torch.where(is_note, labels[..., 1], -100)
            onset_labels = torch.where(is_note, labels[..., 2], -100)
            velocity_labels = torch.where(is_note, labels[..., 3], -100)
            loss = torch.stack([
                self._masked_ce(duration_logits, beat_labels),
                self._masked_ce(duration_logits, note_duration_labels),
                self._masked_ce(logits["onset"], onset_labels),
                self._masked_ce(logits["velocity"], velocity_labels),
            ]).sum()
        return {"loss": loss, "logits": logits,
                "encoder_outputs": BaseModelOutput(last_hidden_state=outputs.encoder_last_hidden_state),
                "past_key_values": outputs.past_key_values}

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")
        self.config.save_pretrained(output_dir)
        (output_dir / "dinr_config.json").write_text(json.dumps({
            "model_type": "dinr_beat_events",
            "duration_target": self.duration_target,
            "beat_duration_target": self.beat_duration_target,
            "note_duration_target": self.note_duration_target,
            "duration_input_tables": self.duration_input_tables,
            "duration_output_head": self.duration_output_head,
            "decoder_mask_probability": self.decoder_mask_probability,
            "duration_bins": BeatTokenSpec.duration_bins,
            "duration_absolute_log_range": BeatTokenSpec.duration_abs_range,
            "duration_log_dev_range": BeatTokenSpec.duration_dev_range,
            "onset_bins": BeatTokenSpec.onset_bins,
            "onset_range": BeatTokenSpec.onset_range,
            "event_topology": "one BEAT followed by its aligned NOTES",
            "event_targets": "BEAT duration; NOTE duration/onset/velocity",
        }, indent=2) + "\n", encoding="utf-8")
