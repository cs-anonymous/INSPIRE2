"""One timing scale per performance, estimated from supported score beats."""
from __future__ import annotations

import copy
import numpy as np

from src.data_process.beat_mapping import (
    map_score_beats, map_score_beats_structural_earliest, valid_beat_intervals,
)


METHOD = 'beat_median_log_ratio'


def normalize_performance(performance, score_ticks, beat_edges, score_beat_ms,
                          anchor_method="onset_group_mean", method=METHOD):
    """Undo any recorded scale, then match median log beat-duration ratio to zero.

    Score tempo changes remain in the denominator at each beat. All IOIs and
    note durations receive the same factor; velocity and pedal values do not.
    """
    result = copy.deepcopy(performance)
    values = np.asarray(result['label_shared_raw'], dtype=np.float64).copy()
    prior = performance.get('global_timing_scale')
    previous_scale = float(prior['scale']) if prior else 1.0
    if not np.isfinite(previous_scale) or previous_scale <= 0:
        raise ValueError('Invalid previous performance scale')
    values[:, :2] /= previous_scale
    mapper = (map_score_beats_structural_earliest if anchor_method == "structural_earliest"
              else map_score_beats)
    if anchor_method not in ("onset_group_mean", "structural_earliest"):
        raise ValueError(f"Unknown beat anchor method: {anchor_method}")
    anchors, supported = mapper(beat_edges, score_ticks, np.cumsum(values[:, 0]))
    valid, durations = valid_beat_intervals(anchors, supported)
    score_ms = np.asarray(score_beat_ms, dtype=np.float64)
    valid &= np.isfinite(score_ms) & (score_ms > 0)
    if not valid.any():
        raise ValueError('No supported positive beat intervals for normalization')
    median = float(np.median(np.log(durations[valid] / score_ms[valid])))
    scale = float(np.exp(-median))
    values[:, :2] *= scale
    result['label_shared_raw'] = values.tolist()
    if 'grouped_ioi_raw' in result:
        result['grouped_ioi_raw'] = (np.asarray(result['grouped_ioi_raw'], dtype=float)
                                     * scale / previous_scale).tolist()
    normalized_anchors, normalized_support = mapper(beat_edges, score_ticks, np.cumsum(values[:, 0]))
    normalized_valid, normalized_duration = valid_beat_intervals(normalized_anchors, normalized_support)
    if not np.array_equal(valid, normalized_valid):
        raise ValueError('Timing scale changed supported positive beat topology')
    residual = float(np.median(np.log(normalized_duration[valid] / score_ms[valid])))
    if abs(residual) > 1e-9:
        raise ValueError(f'Beat normalization residual {residual}')
    result['global_timing_scale'] = dict(
        method=method, anchor_method=anchor_method, scale=scale, log_scale=float(-median),
        median_log_deviation=median, normalized_median_log_deviation=residual,
        valid_beat_intervals=int(valid.sum()),
        previous_scale_undone=previous_scale, previous_normalization=prior,
    )
    return result
