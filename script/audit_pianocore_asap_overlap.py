#!/usr/bin/env python3
"""Find PianoCoRe training pairs matching the official ASAP test recordings."""

import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PIANOCORE = PROJECT / 'backup/PianoCoRe'
ASAP = PROJECT / 'results/pt_rendering_public_recipe_20261124'
REFERENCES = ASAP / 'official_split_measure/reference_manifest.json'
OUTPUT = ROOT / 'results/pianocore_chord_window_audit/asap_overlap_split.json'


def event_array(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    events = sorted((note.start, note.pitch, note.velocity, note.end)
                    for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes)
    values = np.asarray(events, dtype=np.float64)
    if not len(values):
        return values
    values[:, (0, 3)] -= values[0, 0]
    return values


def same_recording(test_events, candidate_events):
    return (test_events.shape == candidate_events.shape and
            np.array_equal(test_events[:, 1:3], candidate_events[:, 1:3]) and
            np.all(np.abs(test_events[:, (0, 3)] - candidate_events[:, (0, 3)]) <= .002))


def main():
    references = json.loads(REFERENCES.read_text())
    rows_by_name = defaultdict(list)
    with (PIANOCORE / 'metadata.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['performance_dataset'] != 'ASAP' or row['tier_a'] != 'True':
                continue
            source_name = Path(row['performance_midi_path']).name.removeprefix('ASAP_')
            rows_by_name[source_name.casefold()].append(row)

    candidates = []
    matches = []
    for reference in references:
        filename = Path(reference['performance']).name.casefold()
        rows = rows_by_name[filename]
        if not rows:
            continue
        test_path = ASAP / 'asap-dataset-master' / reference['performance']
        test_events = event_array(test_path)
        for row in rows:
            source_path = PIANOCORE / 'raw' / row['performance_midi_path']
            if not source_path.is_file():
                continue
            candidate = dict(score=reference['score'], performance=reference['performance'],
                             pianocore_id=row['id'], pianocore_score=row['refined_score_midi_path'],
                             pianocore_performance=row['refined_performance_midi_path'],
                             original_performance=row['performance_midi_path'], source_split=row['split'],
                             tier_a_star=row['tier_a_star'], note_count=len(test_events))
            candidates.append(candidate)
            if row['split'] == 'train' and same_recording(test_events, event_array(source_path)):
                matches.append(dict(candidate, move_to='test'))

    report = dict(reference_performances=len(references), test_scores=len({r['score'] for r in references}),
                  same_name_candidates=len(candidates), exact_event_matches=len(matches),
                  affected_test_performances=len({r['performance'] for r in matches}),
                  affected_test_scores=len({r['score'] for r in matches}),
                  match_rule='same note count, exact pitch and velocity sequence, onset-relative onset/end differences <= 2 ms',
                  split_overlay='Keep PianoCoRe metadata unchanged. Treat matched training rows as test through this overlay.',
                  matches=matches)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'matches'}, indent=2))


if __name__ == '__main__':
    main()
