#!/usr/bin/env python3
"""exp1116: A/B/D, one training per setting, 512 arbitrary events, 16x2."""
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from script.run_dinr_beat_queue import Supervisor, atomic, now, spec

BASE=[spec('A1',seeds=[42]),spec('A2',target='log_dev',seeds=[42]),
      spec('A3',measure=False,seeds=[42]),spec('B1',tables='separate',seeds=[42]),
      spec('B2',target='log_dev',tables='separate',seeds=[42])]
MIXED=[spec('D1',beat='absolute_log',note='log_dev',seeds=[42]),
       spec('D2',beat='log_dev',note='absolute_log',seeds=[42])]


class Supervisor1116(Supervisor):
    def __init__(self,run_dir,gpus):
        super().__init__(run_dir,gpus,data_dir=ROOT.parent/'data/ASAP_processed_dinr_beat_e512_v1',
            sidecar_tag='e512',batch_size=16,accumulation=2,fixed_event_length=512,
            normalize_partial=True,beat_type_embedding=False,experiment='exp1116')
        self.manifest.update(experiment='exp1116',planned_settings=10,planned_trainings=10,
            protocol=dict(window_events=512,window_policy='fixed_event_count',stride_events=512,
                          fixed_event_length=512,batch_size=16,gradient_accumulation_steps=2,
                          trainings_per_setting=1,sampling_draws=3,beat_type_embedding=False,
                          normalize_partial_accumulation=True))
        self.save()

    def best(self,identifiers,decision):
        result=super().best(identifiers,decision)
        self.manifest['decisions'][decision]['criterion']='valid greedy physical-error score from one training per setting; no test metrics'
        self.save()
        return result

    def campaign(self):
        self.stage(BASE+MIXED)
        pair=self.best(['A1','A2','D1','D2'],'D_duration_pair')
        d3=deepcopy(pair);d3.update(id='D3',seeds=[45],source=pair['id'])
        d4=deepcopy(pair);d4.update(id='D4',duration_input_tables='separate',
                                  duration_output_head='shared_conditioned',seeds=[42],source=pair['id'])
        self.stage([d3,d4])
        chosen=self.best(['A1','A2','A3','B1','B2','D1','D2','D3','D4'],'D_measure_start')
        d5=deepcopy(chosen);d5.update(id='D5',measure_start=not chosen['measure_start'],
                                    seeds=[45],source=chosen['id'])
        self.stage([d5])
        self.manifest.update(status='completed',finished_at=now())
        self.save();self.last_report=0;self.report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT.parent/'results/exp1116_beat')
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    with (args.run_dir/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit=json.loads((ROOT/'analysis/exp1115_beat/fixed_event_data_audit.json').read_text())
        assert audit['status']=='passed' and audit['splits']['test']['works']==19
        for split,row in audit['splits'].items():
            assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'], split
        manager=Supervisor1116(args.run_dir.resolve(),args.gpus)
        atomic(args.run_dir/'data_audit.json',audit)
        if args.prepare_only:
            manager.add(BASE+MIXED);manager.manifest['status']='prepared';manager.save();manager.report();return
        try:
            manager.campaign()
        except Exception:
            manager.manifest.update(status='failed',error=traceback.format_exc())
            manager.save();manager.last_report=0;manager.report();raise


if __name__=='__main__':main()
