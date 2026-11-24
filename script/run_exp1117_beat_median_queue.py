#!/usr/bin/env python3
"""Persistent A1-A3/C1-C3 campaign using whole-performance beat-median scales."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.run_dinr_beat_queue import Supervisor, atomic, now, spec

SETTINGS = [
    spec('A1', seeds=[42], onset_target='dev'),
    spec('A2', target='log_dev', seeds=[42], onset_target='dev'),
    spec('A3', measure=False, seeds=[42], onset_target='dev'),
    spec('C1', seeds=[42], onset_target='abs'),
    spec('C2', target='log_dev', seeds=[42], onset_target='abs'),
    spec('C3', head='shared_conditioned', seeds=[42], onset_target='dev'),
]


class BeatMedianSupervisor(Supervisor):
    def __init__(self, run_dir, gpus):
        super().__init__(run_dir, gpus,
            data_dir=ROOT.parent / 'data/ASAP_processed_dinr_beat_median_e512_v1',
            sidecar_tag='e512_v1', batch_size=24, accumulation=2,
            fixed_event_length=512, normalize_partial=True,
            beat_type_embedding=False, experiment='exp1117_beat_median')
        self.manifest.update(experiment='exp1117_beat_median', planned_settings=6, planned_trainings=6,
            report_document=str(ROOT / 'docs/exp1116_BEAT.md'),
            protocol=dict(normalization='beat_median_log_ratio', scope='complete performance',
                window_events=512, stride_events=512, batch_size=24, accumulation=2,
                effective_batch_size=48, duration_input_tables='shared',
                onset_dev=dict(bins=200, bounds=[-1, 1]),
                onset_abs=dict(bins=500, bounds=[-.25, 1]), sampling_draws=3,
                normalization_statistics=str(ROOT / 'analysis/exp1117_beat_median/statistics.json')))
        self.manifest['code_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('src/data_process/beat_normalization.py', 'src/data_process/dinr_beat_dataset.py',
                         'src/train/train_dinr_beat.py', 'src/model/dinr_beat.py',
                         'src/evaluate/evaluate_dinr_beat.py', 'script/evaluate_exp1117_beat_dev.py',
                         'script/prepare_exp1117_beat_median.py', 'script/run_exp1117_beat_median_queue.py')}
        self.save()

    def report(self):
        if time.monotonic() - self.last_report < 60 or not self.report_lock.acquire(blocking=False):
            return
        try:
            self.last_report = time.monotonic()
            with (self.run_dir / 'report.log').open('a') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'script/report_exp1117_beat_median.py'),
                    '--campaign', str(self.run_dir)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print('Report refresh failed; see report.log', flush=True)
        finally:
            self.report_lock.release()

    def campaign(self):
        self.stage(SETTINGS)
        self.manifest.update(status='completed', finished_at=now())
        self.save()
        self.last_report = 0
        self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT.parent / 'results/exp1117_beat_median_bs48')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / 'analysis/exp1117_beat_median/fixed_event_data_audit_checked.json').read_text())
    assert audit['status'] == 'passed' and audit['splits']['test']['works'] == 19
    for split, row in audit['splits'].items():
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'], split
    with (args.run_dir / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = BeatMedianSupervisor(args.run_dir.resolve(), args.gpus)
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
