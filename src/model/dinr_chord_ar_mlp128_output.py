"""First output ablations with the same normalized-log MLP 128 score input."""

import torch
from torch import nn

from src.data_process.dinr_chord_dataset import COARSE_TIMING_BINS, LOG_DEV_HIGH, LOG_DEV_LOW, TIMING_BINS
from src.model.dinr_chord_ar import sample_head
from src.model.dinr_chord_ar_input import DINRChordARInputModel, score_mlp
from src.model.dinr_chord_ar_output import DINRChordAROutputModel, compact_timing_heads

MLP128_OUTPUT_MODES = ('cat5000_rps', 'cat500_mlp128', 'cat5000_timing_velocity')
MLP128_DLM_MODES = {
    'timing_dlm_k1_mlp128': 'timing_dlm_k1_direct',
    'timing_dlm_k3_mlp128': 'timing_dlm_k3_direct',
    'all_dlm_k1_mlp128': 'all_dlm_k1_direct',
}
MLP128_DLM_MASK50 = 'all_dlm_k1_mlp128_mask50'
MLP128_DLM_LEARNED_MASKS = {
    'mask50_learned_keep_pitch': 'learned_feedback',
    'mask50_learned_drop_pitch': 'learned_feedback_and_score_pitch',
    'mask50_learned_fields': 'learned_fields',
}
MLP128_DLM_MASK_MODES = {MLP128_DLM_MASK50: 'zero_performance_feedback', **MLP128_DLM_LEARNED_MASKS}
MLP128_SCORE_INPUT = 'log_separate_wide'
RPS_WEIGHT = 1.0


def ranked_probability_sum(logits, targets):
    cdf = logits.float().softmax(-1).cumsum(-1)[..., :-1]
    boundaries = torch.arange(cdf.shape[-1], device=cdf.device)
    target_cdf = targets[:, None] <= boundaries
    return (cdf - target_cdf.to(cdf.dtype)).square().sum() / cdf.shape[-1]


class DINRChordARMLP128OutputModel(DINRChordARInputModel):
    def __init__(self, output_mode):
        super().__init__(MLP128_SCORE_INPUT)
        if output_mode not in MLP128_OUTPUT_MODES:
            raise ValueError(f'Invalid MLP 128 output: {output_mode}')
        self.output_mode = output_mode
        self.loss_parts = {}
        if output_mode == 'cat500_mlp128':
            compact_timing_heads(self)
        if output_mode == 'cat5000_timing_velocity':
            self.timing_to_velocity = score_mlp(4, 64, self.velocity_head.in_features)

    @staticmethod
    def timing_bins(output_mode):
        coarse = ('cat500_mlp128', *MLP128_DLM_MODES, *MLP128_DLM_MASK_MODES)
        return COARSE_TIMING_BINS if output_mode in coarse else TIMING_BINS

    def field_loss(self, slot, logits, targets):
        ce = super().field_loss(slot, logits, targets)
        if self.output_mode != 'cat5000_rps':
            return ce
        rps = ranked_probability_sum(logits, targets)
        self.loss_parts[slot] = torch.stack((ce.detach(), rps.detach()))
        return ce + RPS_WEIGHT * rps

    def velocity_state(self, hidden, timing, advances):
        ioi_token, duration_token = timing.unbind(-1)
        invalid = advances & (ioi_token == -100)
        step = (LOG_DEV_HIGH - LOG_DEV_LOW) / (TIMING_BINS - 1)
        ioi = (ioi_token.float() * step + LOG_DEV_LOW).masked_fill(~advances | invalid, 0.)
        duration = duration_token.float() * step + LOG_DEV_LOW
        features = torch.stack((ioi, duration, ~advances, invalid), dim=-1)
        return hidden + self.timing_to_velocity(features)

    def head_losses(self, hidden, attention, advances, labels):
        velocity_state = None
        if self.output_mode == 'cat5000_timing_velocity':
            velocity_state = self.velocity_state(hidden[attention], labels[attention][..., :2], advances[attention])
        return super().head_losses(hidden, attention, advances, labels, velocity_state=velocity_state)

    def decode_step(self, hidden, rng, temperature, top_p, advances=None):
        if self.output_mode != 'cat5000_timing_velocity':
            return super().decode_step(hidden, rng, temperature, top_p, advances)
        ioi = sample_head(self.ioi_head(hidden), rng, temperature, top_p).masked_fill(~advances, 0)
        duration = sample_head(self.duration_head(hidden), rng, temperature, top_p)
        timing = torch.stack((ioi, duration), dim=-1)
        velocity_state = self.velocity_state(hidden, timing, advances)
        velocity = sample_head(self.velocity_head(velocity_state), rng, temperature, top_p)
        return torch.cat((timing, velocity[:, None]), dim=-1), None


class DINRChordARMLP128DLMModel(DINRChordAROutputModel):
    def __init__(self, output_mode):
        mask_mode = MLP128_DLM_MASK_MODES.get(output_mode)
        dlm_mode = 'all_dlm_k1_mlp128' if mask_mode else output_mode
        super().__init__(MLP128_DLM_MODES[dlm_mode], score_input=MLP128_SCORE_INPUT)
        self.history_mask_probability = 0.5 if mask_mode else 0.0
        self.mask_history_pitch = mask_mode == 'learned_feedback_and_score_pitch'
        if mask_mode in ('learned_feedback', 'learned_feedback_and_score_pitch'):
            self.history_mask_embedding = nn.Parameter(torch.zeros(self.bos.shape))
        if mask_mode == 'learned_fields':
            self.history_mask_fields = nn.Parameter(torch.zeros(3, self.feedback_tables[0].embedding_dim))
