"""Minimal note-level DINR with absolute timing targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5gemma.modeling_t5gemma import (
    bidirectional_mask_function,
    create_causal_mask,
    create_sliding_window_causal_mask,
    sliding_window_bidirectional_mask_function,
)
from src.model.pianoformer import (
    PianoT5Gemma,
    PianoT5GemmaConfig,
    PianoT5GemmaEncoder,
    PianoT5GemmaModel,
)
from src.data_process.dinr_dataset import DINRTokenSpec


MASKED_ATTRIBUTE_SLOTS = (1, 2, 3)


class DINRNoteEncoder(PianoT5GemmaEncoder):
    """PianoT5Gemma encoder with one hidden state per input note."""

    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens
        del self.embeddings

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if attention_mask is None:
            attention_mask = inputs_embeds.new_ones(inputs_embeds.shape[:2], dtype=torch.long)
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(
                **mask_kwargs,
                or_mask_function=bidirectional_mask_function(attention_mask),
            ),
            "sliding_attention": create_sliding_window_causal_mask(
                **mask_kwargs,
                or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                and_mask_function=bidirectional_mask_function(attention_mask),
            ),
        }
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=inputs_embeds.dtype,
                                  device=inputs_embeds.device)
        hidden_states = self.dropout(inputs_embeds * normalizer)
        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states, position_embeddings,
                masks[layer_module.attention_type], position_ids, **kwargs,
            )
        return BaseModelOutput(last_hidden_state=self.dropout(self.norm(hidden_states)))


class DINRAbsoluteModel(nn.Module):
    """PT backbone at one note per time step with categorical attribute heads."""

    def __init__(self, config: PianoT5GemmaConfig | None = None, pedal_mode="ce",
                 numerical_coordinates=False, coordinate_frequencies=16,
                 decoder_mask_probability=0.0, decoder_mask_mode="note",
                 num_input_slots=None, separate_ioi_duration_tables=False,
                 timing_target_mode="absolute", time_input_mode="linear",
                 coordinate_encoding="fourier", coordinate_hidden_size=32,
                 separate_coordinate_projections=False, coordinate_combination="add"):
        super().__init__()
        self.config = config or PianoT5GemmaConfig()
        self.pedal_mode = str(pedal_mode).lower()
        self.numerical_coordinates = bool(numerical_coordinates)
        self.coordinate_frequencies = max(1, int(coordinate_frequencies))
        self.coordinate_encoding = str(coordinate_encoding)
        self.coordinate_hidden_size = int(coordinate_hidden_size)
        self.separate_coordinate_projections = bool(separate_coordinate_projections)
        self.coordinate_combination = str(coordinate_combination)
        if self.coordinate_combination not in {"add", "replace"}:
            raise ValueError("coordinate_combination must be add or replace")
        if self.coordinate_combination == "replace" and (
            not self.numerical_coordinates or not self.separate_coordinate_projections
            or self.pedal_mode != "none"
        ):
            raise ValueError("Coordinate replacement requires three independent numerical mappings and no pedals")
        if self.coordinate_encoding not in {"linear", "mlp", "fourier"}:
            raise ValueError("coordinate_encoding must be linear, mlp, or fourier")
        if self.coordinate_hidden_size < 1:
            raise ValueError("coordinate_hidden_size must be positive")
        if self.separate_coordinate_projections and not self.numerical_coordinates:
            raise ValueError("Separate coordinate projections require numerical_coordinates")
        self.decoder_mask_probability = float(decoder_mask_probability)
        self.decoder_mask_mode = str(decoder_mask_mode).lower()
        self.separate_ioi_duration_tables = bool(separate_ioi_duration_tables)
        self.timing_target_mode = str(timing_target_mode)
        self.time_input_mode = str(time_input_mode)
        expected_slots = 4 if self.pedal_mode == "none" else 8
        self.num_input_slots = expected_slots if num_input_slots is None else int(num_input_slots)
        if not 0.0 <= self.decoder_mask_probability < 1.0:
            raise ValueError("decoder_mask_probability must be in [0, 1)")
        if self.pedal_mode not in {"ce", "bce", "none"}:
            raise ValueError("pedal_mode must be ce, bce, or none")
        if self.num_input_slots not in (4, 8) or (
            self.pedal_mode != "none" and self.num_input_slots != 8
        ):
            raise ValueError("num_input_slots must be 4 or 8; pedal heads require 8 slots")
        if self.decoder_mask_mode not in {"note", "attribute", "timestep", "timestep_single"}:
            raise ValueError(
                "decoder_mask_mode must be note, attribute, timestep, or timestep_single"
            )
        self.backbone = PianoT5GemmaModel(self.config)
        self.backbone.encoder = DINRNoteEncoder(self.config.encoder)
        self.token_embedding = nn.Embedding(
            # Replacement has no learned rows for timing or velocity values.
            DINRTokenSpec.velocity_start if self.coordinate_combination == "replace"
            else DINRTokenSpec.vocab_size,
            self.config.hidden_size,
            padding_idx=DINRTokenSpec.pad_id,
        )
        self.ioi_token_embedding = (
            nn.Embedding(DINRTokenSpec.timing_bins, self.config.hidden_size)
            if self.separate_ioi_duration_tables and self.coordinate_combination != "replace" else None
        )
        self.duration_token_embedding = (
            nn.Embedding(DINRTokenSpec.timing_bins, self.config.hidden_size)
            if self.separate_ioi_duration_tables and self.coordinate_combination != "replace" else None
        )
        self.decoder_mask_embedding = nn.Parameter(torch.empty(self.config.hidden_size))
        nn.init.normal_(self.decoder_mask_embedding, mean=0.0, std=self.config.initializer_range)
        self.decoder_attribute_mask_embeddings = nn.Parameter(
            torch.empty(len(MASKED_ATTRIBUTE_SLOTS), self.config.hidden_size)
        )
        nn.init.normal_(
            self.decoder_attribute_mask_embeddings,
            mean=0.0,
            std=self.config.initializer_range,
        )
        # Token ids alone do not identify the field: IOI and duration share
        # the timing vocabulary, and the four pedal slots share two ids.
        # Slot-specific residual linear maps make each value embedding
        # field-specific before the per-note sum. Zero initialization keeps
        # PT embeddings unchanged at the start of fine-tuning.
        self.note_slot_projection = nn.Parameter(torch.zeros(
            self.num_input_slots, self.config.hidden_size, self.config.hidden_size
        ))
        hidden = self.config.decoder.hidden_size
        self.grouped_ioi_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        self.chord_ioi_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        self.velocity_head = nn.Linear(hidden, DINRTokenSpec.velocity_bins)
        self.duration_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        pedal_width = 2 if self.pedal_mode == "ce" else 1
        self.pedal_heads = nn.ModuleList(
            [nn.Linear(hidden, pedal_width) for _ in range(4)]
        ) if self.pedal_mode != "none" else nn.ModuleList()
        self.timing_coordinate_projection = self._make_coordinate_projection()
        self.velocity_coordinate_projection = self._make_coordinate_projection()
        self.duration_coordinate_projection = (
            self._make_coordinate_projection() if self.separate_coordinate_projections else None
        )

    def _make_coordinate_projection(self):
        if not self.numerical_coordinates:
            return None
        if self.coordinate_encoding == "linear":
            return nn.Linear(1, self.config.hidden_size)
        if self.coordinate_encoding == "mlp":
            return nn.Sequential(
                nn.Linear(1, self.coordinate_hidden_size), nn.GELU(),
                nn.Linear(self.coordinate_hidden_size, self.config.hidden_size),
            )
        return nn.Linear(1 + 2 * self.coordinate_frequencies,
                         self.config.hidden_size, bias=False)

    def initialize_from_pt(self, checkpoint):
        """Initialize compatible PT backbone and token embeddings when available."""
        pt = PianoT5Gemma.from_pretrained(checkpoint, torch_dtype=torch.float32)
        own = self.backbone.state_dict()
        source = pt.model.state_dict()
        compatible = {
            key: value for key, value in source.items()
            if key in own and own[key].shape == value.shape
        }
        missing, unexpected = self.backbone.load_state_dict(compatible, strict=False)
        source_embedding = pt.get_encoder().embeddings.word_embeddings.weight
        with torch.no_grad():
            target_embedding = self.token_embedding.weight
            target_embedding[DINRTokenSpec.pitch_start: DINRTokenSpec.pitch_start + DINRTokenSpec.pitch_bins].copy_(
                source_embedding[5:133]
            )
            if self.coordinate_combination != "replace":
                target_embedding[DINRTokenSpec.velocity_start: DINRTokenSpec.velocity_start + DINRTokenSpec.velocity_bins].copy_(
                    source_embedding[133:261]
                )
                timing_embeddings = []
                for timing_id in range(DINRTokenSpec.timing_bins):
                    old_start = 261 + timing_id * DINRTokenSpec.timing_bin_ms
                    old_end = old_start + DINRTokenSpec.timing_bin_ms
                    timing_embeddings.append(source_embedding[old_start:old_end].mean(dim=0))
                timing_embeddings = torch.stack(timing_embeddings)
                target_embedding[
                    DINRTokenSpec.timing_start:
                    DINRTokenSpec.timing_start + DINRTokenSpec.timing_bins
                ].copy_(timing_embeddings)
                if self.separate_ioi_duration_tables:
                    self.ioi_token_embedding.weight.copy_(timing_embeddings)
                    self.duration_token_embedding.weight.copy_(timing_embeddings)
                target_embedding[DINRTokenSpec.pedal_start].copy_(source_embedding[5261])
                target_embedding[DINRTokenSpec.pedal_start + 1].copy_(source_embedding[5325])
        return {"loaded_backbone_tensors": len(compatible),
                "numerical_lookup_initialized": self.coordinate_combination != "replace",
                "missing_backbone_tensors": len(missing),
                "unused_source_tensors": len(unexpected)}

    def _coordinate_features(self, values):
        values = values.float().unsqueeze(-1)
        if self.coordinate_encoding != "fourier":
            return values
        frequencies = torch.arange(
            1, self.coordinate_frequencies + 1,
            device=values.device, dtype=values.dtype,
        )
        angles = values * frequencies * torch.pi
        return torch.cat([values, torch.sin(angles), torch.cos(angles)], dim=-1)

    def _sum_note_tokens(self, tokens, attribute_mask=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.num_input_slots:
            raise ValueError(
                f"Expected [batch, notes, {self.num_input_slots}] token blocks, "
                f"got {tuple(tokens.shape)}"
            )
        lookup_tokens = tokens
        if self.coordinate_combination == "replace":
            # Numeric IDs only select coordinates. They never index a learned
            # numerical table; PAD's embedding is zero.
            lookup_tokens = tokens.masked_fill(tokens >= DINRTokenSpec.velocity_start,
                                                DINRTokenSpec.pad_id)
        embeds = self.token_embedding(lookup_tokens)
        if self.ioi_token_embedding is not None:
            ioi_tokens = tokens[:, :, 1]
            ioi_mask = (
                (ioi_tokens >= DINRTokenSpec.timing_start)
                & (ioi_tokens < DINRTokenSpec.timing_start + DINRTokenSpec.timing_bins)
            )
            ioi_values = (ioi_tokens - DINRTokenSpec.timing_start).clamp(
                min=0, max=DINRTokenSpec.timing_bins - 1
            )
            duration_tokens = tokens[:, :, 3]
            duration_mask = (
                (duration_tokens >= DINRTokenSpec.timing_start)
                & (duration_tokens < DINRTokenSpec.timing_start + DINRTokenSpec.timing_bins)
            )
            duration_values = (duration_tokens - DINRTokenSpec.timing_start).clamp(
                min=0, max=DINRTokenSpec.timing_bins - 1
            )
            embeds = embeds.clone()
            embeds[:, :, 1, :] = torch.where(
                ioi_mask.unsqueeze(-1),
                self.ioi_token_embedding(ioi_values),
                embeds[:, :, 1, :],
            )
            embeds[:, :, 3, :] = torch.where(
                duration_mask.unsqueeze(-1),
                self.duration_token_embedding(duration_values),
                embeds[:, :, 3, :],
            )
        if self.numerical_coordinates:
            timing_mask = (
                (tokens >= DINRTokenSpec.timing_start)
                & (tokens < DINRTokenSpec.timing_start + DINRTokenSpec.timing_bins)
            )
            timing_seconds = (
                (tokens - DINRTokenSpec.timing_start).clamp(
                    min=0, max=DINRTokenSpec.timing_bins - 1
                ).float()
                * (DINRTokenSpec.timing_bin_ms / 1000.0)
            )
            timing_coord = self.timing_coordinate_projection(
                self._coordinate_features(timing_seconds)
            )
            if self.duration_coordinate_projection is not None:
                timing_coord = timing_coord.clone()
                timing_coord[:, :, 3, :] = self.duration_coordinate_projection(
                    self._coordinate_features(timing_seconds[:, :, 3])
                )
            embeds = embeds + timing_coord * timing_mask.unsqueeze(-1)

            velocity_mask = (
                (tokens >= DINRTokenSpec.velocity_start)
                & (tokens < DINRTokenSpec.velocity_start + DINRTokenSpec.velocity_bins)
            )
            velocity = (
                (tokens - DINRTokenSpec.velocity_start).clamp(
                    min=0, max=DINRTokenSpec.velocity_bins - 1
                ).float() / (DINRTokenSpec.velocity_bins - 1)
            )
            velocity_coord = self.velocity_coordinate_projection(
                self._coordinate_features(velocity)
            )
            embeds = embeds + velocity_coord * velocity_mask.unsqueeze(-1)
        embeds = embeds + torch.einsum(
            "bnsh,shd->bnsd", embeds, self.note_slot_projection
        )
        if attribute_mask is not None:
            if attribute_mask.shape != (*tokens.shape[:2], len(MASKED_ATTRIBUTE_SLOTS)):
                raise ValueError("attribute_mask must have shape [batch, notes, 3]")
            slot_embeddings = []
            for slot, slot_embedding in enumerate(embeds.unbind(dim=2)):
                if slot in MASKED_ATTRIBUTE_SLOTS:
                    mask_index = MASKED_ATTRIBUTE_SLOTS.index(slot)
                    mask_embedding = self.decoder_attribute_mask_embeddings[mask_index]
                    slot_embedding = torch.where(
                        attribute_mask[:, :, mask_index].unsqueeze(-1),
                        mask_embedding.to(dtype=slot_embedding.dtype).view(1, 1, -1),
                        slot_embedding,
                    )
                slot_embeddings.append(slot_embedding)
            embeds = torch.stack(slot_embeddings, dim=2)
        return embeds.sum(dim=2)

    def _mask_decoder_attributes(self, decoder_tokens, decoder_attention_mask):
        if (not self.training or self.decoder_mask_mode not in {"attribute", "timestep"}
                or self.decoder_mask_probability <= 0.0 or decoder_tokens.shape[1] <= 1):
            return None
        # Both modes preserve pitch and mask IOI, velocity, and duration.
        # Attribute mode samples each slot independently; timestep mode uses
        # one shared Bernoulli decision for all three timing attributes.
        maskable = decoder_attention_mask.bool().clone()
        maskable[:, 0] = False
        slots = (1, 2, 3)
        if self.decoder_mask_mode == "timestep":
            row_drop = (torch.rand(decoder_tokens.shape[:2], device=decoder_tokens.device)
                        < self.decoder_mask_probability) & maskable
            dropped = row_drop.unsqueeze(-1).expand(-1, -1, len(slots))
        else:
            dropped = (
                torch.rand((*decoder_tokens.shape[:2], len(slots)), device=decoder_tokens.device)
                < self.decoder_mask_probability
            ) & maskable.unsqueeze(-1)
        return dropped

    def _mask_decoder_timestep_single(self, decoder_tokens, decoder_attention_mask):
        if (not self.training or self.decoder_mask_mode != "timestep_single"
                or self.decoder_mask_probability <= 0.0 or decoder_tokens.shape[1] <= 1):
            return None
        maskable = decoder_attention_mask.bool().clone()
        maskable[:, 0] = False
        return (torch.rand(maskable.shape, device=decoder_tokens.device)
                < self.decoder_mask_probability) & maskable

    def _heads(self, hidden):
        result = {
            "grouped_ioi": self.grouped_ioi_head(hidden),
            "chord_ioi": self.chord_ioi_head(hidden),
            "duration": self.duration_head(hidden),
            "velocity": self.velocity_head(hidden),
            "pedal": None,
        }
        if self.pedal_mode == "ce":
            result["pedal"] = torch.stack([head(hidden) for head in self.pedal_heads], dim=2)
        elif self.pedal_mode == "bce":
            result["pedal"] = torch.cat([head(hidden) for head in self.pedal_heads], dim=-1)
        return result

    @staticmethod
    def _masked_ce(logits, target, mask):
        valid = mask & (target >= 0)
        if not valid.any():
            return logits.new_zeros(())
        return F.cross_entropy(logits[valid], target[valid])

    @staticmethod
    def _masked_bce(logits, target, mask):
        valid = mask & (target >= 0)
        if not valid.any():
            return logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(logits[valid], target[valid].float())

    def forward(
        self,
        score_tokens,
        decoder_tokens,
        labels=None,
        attention_mask=None,
        decoder_attention_mask=None,
        score_zero_ioi_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
    ):
        score_embeds = self._sum_note_tokens(score_tokens)
        if attention_mask is None:
            attention_mask = score_tokens.new_ones(score_tokens.shape[:2])
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask
        attribute_mask = self._mask_decoder_attributes(
            decoder_tokens, decoder_attention_mask
        )
        decoder_embeds = self._sum_note_tokens(decoder_tokens, attribute_mask)
        single_timestep_mask = self._mask_decoder_timestep_single(
            decoder_tokens, decoder_attention_mask
        )
        if single_timestep_mask is not None:
            pitch_only_tokens = decoder_tokens.new_full(
                decoder_tokens.shape, DINRTokenSpec.pad_id
            )
            pitch_only_tokens[:, :, 0] = decoder_tokens[:, :, 0]
            pitch_embeds = self._sum_note_tokens(pitch_only_tokens)
            masked_embeds = pitch_embeds + self.decoder_mask_embedding.to(
                dtype=decoder_embeds.dtype
            ).view(1, 1, -1)
            decoder_embeds = torch.where(
                single_timestep_mask.unsqueeze(-1), masked_embeds, decoder_embeds
            )
        if (self.training and self.decoder_mask_mode == "note"
                and self.decoder_mask_probability > 0.0 and decoder_embeds.shape[1] > 1):
            maskable = decoder_attention_mask.bool().clone()
            maskable[:, 0] = False
            dropped = (torch.rand(maskable.shape, device=decoder_embeds.device)
                       < self.decoder_mask_probability) & maskable
            decoder_embeds = torch.where(
                dropped.unsqueeze(-1),
                self.decoder_mask_embedding.to(dtype=decoder_embeds.dtype).view(1, 1, -1),
                decoder_embeds,
            )
        outputs = self.backbone(
            inputs_embeds=None if encoder_outputs is not None else score_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_embeds,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        logits = self._heads(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            mask = attention_mask.bool()
            if score_zero_ioi_mask is None:
                raise ValueError("DINR requires score_zero_ioi_mask to route grouped/chord IOI")
            zero_mask = score_zero_ioi_mask.bool() & mask
            positive_mask = (~score_zero_ioi_mask.bool()) & mask
            ioi_loss = (
                self._masked_ce(logits["grouped_ioi"], labels[..., 0], positive_mask)
                + self._masked_ce(logits["chord_ioi"], labels[..., 0], zero_mask)
            )
            losses = [
                ioi_loss,
                self._masked_ce(logits["duration"], labels[..., 1], mask),
                self._masked_ce(logits["velocity"], labels[..., 2], mask),
            ]
            if self.pedal_mode == "ce":
                pedal_losses = [
                    self._masked_ce(logits["pedal"][..., index, :], labels[..., 3 + index], mask)
                    for index in range(4)
                ]
                losses.append(torch.stack(pedal_losses).mean())
            elif self.pedal_mode == "bce":
                pedal_losses = [
                    self._masked_bce(logits["pedal"][..., index], labels[..., 3 + index], mask)
                    for index in range(4)
                ]
                losses.append(torch.stack(pedal_losses).mean())
            loss = torch.stack(losses).sum()
        return {"loss": loss, "logits": logits,
                "encoder_outputs": BaseModelOutput(last_hidden_state=outputs.encoder_last_hidden_state),
                "past_key_values": outputs.past_key_values}

    def save_pretrained(self, output_dir, state_dict=None, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict() if state_dict is None else state_dict,
                   output_dir / "pytorch_model.bin")
        if hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(output_dir)
        (output_dir / "dinr_config.json").write_text(
            json.dumps({
                "model_type": "dinr_absolute",
                "ioi_heads": ["grouped_ioi", "chord_ioi"],
                "pedal_mode": self.pedal_mode,
                "slot_sum_projection": (
                    f"{self.num_input_slots} independent residual "
                    "hidden_size-to-hidden_size linear maps"
                ),
                "numerical_coordinates": self.numerical_coordinates,
                "coordinate_frequencies": self.coordinate_frequencies,
                "coordinate_encoding": self.coordinate_encoding,
                "coordinate_combination": self.coordinate_combination,
                "coordinate_hidden_size": self.coordinate_hidden_size,
                "separate_coordinate_projections": self.separate_coordinate_projections,
                "separate_ioi_duration_tables": self.separate_ioi_duration_tables,
                "decoder_mask_probability": self.decoder_mask_probability,
                "decoder_mask_mode": self.decoder_mask_mode,
                "decoder_attribute_mask_embeddings": "three slot-specific learned vectors",
                "decoder_attribute_mask_init_std": self.config.initializer_range,
                "timing_target_mode": self.timing_target_mode,
                "time_input_mode": self.time_input_mode,
                "num_input_slots": self.num_input_slots,
                "vocab_size": DINRTokenSpec.vocab_size,
            }, indent=2),
            encoding="utf-8",
        )
