#!/usr/bin/env python3
"""Run the corrected beat/onset-dev six-setting queue."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.run_dinr_beat_queue import Supervisor, atomic, now, spec


SETTINGS = [
    spec('A1', seeds=[42], onset_target='dev'),
    spec('A2', target='log_dev', seeds=[42], onset_target='dev'),
    spec('A3', measure=False, seeds=[42], onset_target='dev'),
    spec('B1', tables='separate', seeds=[42], onset_target='dev'),
    spec('B2', target='log_dev', tables='separate', seeds=[42], onset_target='dev'),
    spec('D4', tables='separate', head='shared_conditioned', seeds=[42], onset_target='dev'),
]


class Supervisor1117(Supervisor):
    def report(self):
        if time.monotonic() - self.last_report < 60 or not self.report_lock.acquire(blocking=False):
            return
        try:
            self.last_report = time.monotonic()
            with (self.run_dir / 'report.log').open('a') as log:
                result = subprocess.run(
                    [sys.executable, str(ROOT / 'script/report_exp1116_beat_dev.py'),
                     '--campaign', str(self.run_dir)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                print('Report refresh failed; see report.log', flush=True)
        finally:
            self.report_lock.release()

    def __init__(self, run_dir, gpus):
        super().__init__(
            run_dir, gpus,
            data_dir=ROOT.parent / 'data/ASAP_processed_dinr_beat_dev_e512_v2',
            sidecar_tag='e512_v2', batch_size=24, accumulation=2,
            fixed_event_length=512, normalize_partial=True,
            beat_type_embedding=False, experiment='exp1117',
        )
        self.manifest.update(
            experiment='exp1117', planned_settings=6, planned_trainings=6,
            report_document=str(ROOT / 'docs/exp1116_BEAT.md'),
            protocol={
                'window_events': 512, 'window_policy': 'fixed_event_count',
                'stride_events': 512, 'fixed_event_length': 512,
                'batch_size': 24, 'gradient_accumulation_steps': 2,
                'trainings_per_setting': 1, 'sampling_draws': 3,
                'onset_target': 'dev', 'onset_bins': 200,
                'onset_range': [-1.0, 1.0], 'onset_bin_width': 0.01,
                'duration_bins': 500, 'duration_bin_width': 0.01,
                'duration_absolute_log_range': [3.0, 8.0],
                'duration_log_dev_range': [-3.0, 2.0],
            },
        )
        self.save()

    def campaign(self):
        self.stage(SETTINGS)
        self.manifest.update(status='completed', finished_at=now())
        self.save()
        self.last_report = 0
        self.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT.parent / 'results/exp1117_beat_dev_bs48')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / 'analysis/exp1117_beat_dev/fixed_event_data_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and audit['splits']['test']['works'] == 19
    for split, row in audit['splits'].items():
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'], split
    with (args.run_dir / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Supervisor1117(args.run_dir.resolve(), args.gpus)
        atomic(args.run_dir / 'data_audit.json', audit)
        if args.prepare_only:
            manager.add(SETTINGS)
            manager.manifest['status'] = 'prepared'
            manager.save()
            manager.report()
            return
        try:
            manager.campaign()
        except Exception:
            manager.manifest.update(status='failed', error=traceback.format_exc())
            manager.save()
            manager.last_report = 0
            manager.report()
            raise


if __name__ == '__main__':
    main()
