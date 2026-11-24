"""Simple score-beat to performance-time mapping.

The score beat grid is known.  We estimate its performance-time anchors from
paired notes by collapsing each score onset group to the mean performance
onset and linearly interpolating only between observed groups.  There is no
tempo threshold, monotonic filtering, or endpoint extrapolation here.
"""

from __future__ import annotations

import numpy as np


def onset_group_pairs(score_ticks, performance_onsets_ms):
    score_ticks = np.asarray(score_ticks, dtype=np.float64)
    performance_onsets_ms = np.asarray(performance_onsets_ms, dtype=np.float64)
    if score_ticks.ndim != 1 or performance_onsets_ms.shape != score_ticks.shape:
        raise ValueError("score ticks and performance onsets must be aligned 1-D arrays")
    if not len(score_ticks):
        return np.empty((0, 2), dtype=np.float64)
    unique_ticks, inverse = np.unique(score_ticks, return_inverse=True)
    sums = np.bincount(inverse, weights=performance_onsets_ms)
    counts = np.bincount(inverse)
    return np.column_stack((unique_ticks, sums / counts))


def map_score_beats(score_beats, score_ticks, performance_onsets_ms):
    """Map score beat ticks to performance milliseconds.

    Returns ``(anchors_ms, supported)``.  A beat boundary is supported only
    when it lies between the first and last observed score onset groups.  A
    beat interval is usable only when both boundaries are supported and its
    mapped duration is strictly positive.
    """
    score_beats = np.asarray(score_beats, dtype=np.float64)
    pairs = onset_group_pairs(score_ticks, performance_onsets_ms)
    anchors = np.full(len(score_beats), np.nan, dtype=np.float64)
    if not len(score_beats) or not len(pairs):
        return anchors, np.zeros(len(score_beats), dtype=bool)
    group_ticks, group_times = pairs[:, 0], pairs[:, 1]
    supported = (score_beats >= group_ticks[0]) & (score_beats <= group_ticks[-1])
    inside = np.flatnonzero(supported)
    if len(inside):
        anchors[inside] = np.interp(score_beats[inside], group_ticks, group_times)
    exact = np.searchsorted(group_ticks, score_beats, side="left")
    exact_mask = ((exact < len(group_ticks)) & supported
                  & (group_ticks[np.clip(exact, 0, len(group_ticks) - 1)] == score_beats))
    anchors[exact_mask] = group_times[exact[exact_mask]]
    return anchors, supported


def map_score_beats_structural_earliest(score_beats, score_ticks, performance_onsets_ms):
    """Map beats from earliest performance onsets of score onset groups.

    Each score onset group is collapsed to its earliest aligned performance
    onset, avoiding delay from rolled chords. Beat edges are interpolated
    between observed groups; no endpoint extrapolation is performed. The MIDI
    format has no reliable grace-note flag, so standalone ornaments cannot be
    removed automatically and remain an explicit limitation of this rule.
    """
    score_beats = np.asarray(score_beats, dtype=np.float64)
    score_ticks = np.asarray(score_ticks, dtype=np.float64)
    performance_onsets_ms = np.asarray(performance_onsets_ms, dtype=np.float64)
    if score_ticks.ndim != 1 or performance_onsets_ms.shape != score_ticks.shape:
        raise ValueError("score ticks and performance onsets must be aligned 1-D arrays")
    anchors = np.full(len(score_beats), np.nan, dtype=np.float64)
    if not len(score_beats) or not len(score_ticks):
        return anchors, np.zeros(len(score_beats), dtype=bool)

    # Each score onset group is a structural candidate; the earliest performed
    # member supplies its anchor and prevents a rolled chord from delaying it.
    structural_ticks = np.unique(score_ticks)
    if not len(structural_ticks):
        return anchors, np.zeros(len(score_beats), dtype=bool)
    structural_times = np.asarray([
        np.min(performance_onsets_ms[score_ticks == tick]) for tick in structural_ticks
    ], dtype=np.float64)
    supported = (score_beats >= structural_ticks[0]) & (score_beats <= structural_ticks[-1])
    inside = np.flatnonzero(supported)
    if len(inside):
        anchors[inside] = np.interp(score_beats[inside], structural_ticks, structural_times)
    exact = np.searchsorted(structural_ticks, score_beats, side="left")
    exact_mask = ((exact < len(structural_ticks)) & supported
                  & (structural_ticks[np.clip(exact, 0, len(structural_ticks) - 1)] == score_beats))
    anchors[exact_mask] = structural_times[exact[exact_mask]]
    return anchors, supported


def valid_beat_intervals(anchors_ms, supported=None):
    anchors_ms = np.asarray(anchors_ms, dtype=np.float64)
    if len(anchors_ms) < 2:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    valid = np.isfinite(anchors_ms[:-1]) & np.isfinite(anchors_ms[1:])
    if supported is not None:
        supported = np.asarray(supported, dtype=bool)
        valid &= supported[:-1] & supported[1:]
    durations = np.diff(anchors_ms)
    valid &= durations > 0.0
    return valid, durations
