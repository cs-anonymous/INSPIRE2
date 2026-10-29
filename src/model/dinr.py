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

    def __init__(self, config: PianoT5GemmaConfig | None = None, pedal_mode="ce"):
        super().__init__()
        self.config = config or PianoT5GemmaConfig()
        self.pedal_mode = str(pedal_mode).lower()
        if self.pedal_mode not in {"ce", "bce", "none"}:
            raise ValueError("pedal_mode must be ce, bce, or none")
        self.backbone = PianoT5GemmaModel(self.config)
        self.backbone.encoder = DINRNoteEncoder(self.config.encoder)
        self.token_embedding = nn.Embedding(
            DINRTokenSpec.vocab_size,
            self.config.hidden_size,
            padding_idx=DINRTokenSpec.pad_id,
        )
        hidden = self.config.decoder.hidden_size
        self.grouped_ioi_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        self.chord_ioi_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        self.velocity_head = nn.Linear(hidden, DINRTokenSpec.velocity_bins)
        self.duration_head = nn.Linear(hidden, DINRTokenSpec.timing_bins)
        pedal_width = 2 if self.pedal_mode == "ce" else 1
        self.pedal_heads = nn.ModuleList(
            [nn.Linear(hidden, pedal_width) for _ in range(4)]
        ) if self.pedal_mode != "none" else nn.ModuleList()

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
            target_embedding[DINRTokenSpec.velocity_start: DINRTokenSpec.velocity_start + DINRTokenSpec.velocity_bins].copy_(
                source_embedding[133:261]
            )
            for timing_id in range(DINRTokenSpec.timing_bins):
                old_start = 261 + timing_id * DINRTokenSpec.timing_bin_ms
                old_end = old_start + DINRTokenSpec.timing_bin_ms
                target_embedding[DINRTokenSpec.timing_start + timing_id].copy_(
                    source_embedding[old_start:old_end].mean(dim=0)
                )
            target_embedding[DINRTokenSpec.pedal_start].copy_(source_embedding[5261])
            target_embedding[DINRTokenSpec.pedal_start + 1].copy_(source_embedding[5325])
        return {"loaded_backbone_tensors": len(compatible),
                "missing_backbone_tensors": len(missing),
                "unused_source_tensors": len(unexpected)}

    def _sum_note_tokens(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != 8:
            raise ValueError(f"Expected [batch, notes, 8] token blocks, got {tuple(tokens.shape)}")
        embeds = self.token_embedding(tokens)
        return embeds.sum(dim=2)

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
        decoder_embeds = self._sum_note_tokens(decoder_tokens)
        if attention_mask is None:
            attention_mask = score_tokens.new_ones(score_tokens.shape[:2])
        if decoder_attention_mask is None:
            decoder_attention_mask = attention_mask
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
                "vocab_size": DINRTokenSpec.vocab_size,
            }, indent=2),
            encoding="utf-8",
        )
