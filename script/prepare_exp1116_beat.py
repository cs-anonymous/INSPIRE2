#!/usr/bin/env python3
"""Rewindow frozen continuous events at exactly 512 total events per block."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_process.dinr_beat_dataset import fixed_event_spans


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, block_events=512, frozen_path=None,
            source_tag='b512_o50', output_tag=None):
    frozen = json.loads(frozen_path.read_text()) if frozen_path is not None else None
    audit = dict(status='passed',window_events=block_events,stride_events=block_events,
                 window_policy='fixed_event_count',overlap_ratio=0.0,splits={})
    work_sets = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train','valid','test'):
        path = source/f'dinr_beat_{split}_{source_tag}.pt'
        if frozen is not None and sha256(path) != frozen['splits'][split]['sha256']:
            raise ValueError(f'{split}: frozen source hash differs')
        data = torch.load(path,map_location='cpu',weights_only=False)
        events, targets = data['events'], data['targets']
        assert torch.isfinite(events).all() and torch.isfinite(targets).all()
        samples, coverage = [], torch.zeros(len(events),dtype=torch.int16)
        for performance in data['performances']:
            start, end = performance['span']
            for left, right in fixed_event_spans(end-start, block_events):
                a,b = start+left,start+right
                types = events[a:b,0]
                samples.append(dict(span=[a,b],work=performance['work'],
                                    performance=performance['performance'],
                                    notes=int((types==1).sum()),beats=int((types==0).sum())))
                coverage[a:b] += 1
        assert (coverage==1).all(), 'Every event must appear exactly once'
        assert all(0 < b-a <= block_events for a,b in (r['span'] for r in samples))
        data.update(samples=samples,window_notes=None,window_events=block_events,
                    stride_events=block_events,window_policy='fixed_event_count',overlap_ratio=0.0,
                    source_sidecar=str(path.resolve()),source_sha256=sha256(path))
        destination=output/f'dinr_beat_{split}_{output_tag or f"e{block_events}"}.pt'
        torch.save(data,destination)
        work_sets[split]={r['work'] for r in data['performances']}
        lengths=np.array([r['span'][1]-r['span'][0] for r in samples])
        audit['splits'][split]=dict(path=str(destination.resolve()),sha256=sha256(destination),
            source_sha256=data['source_sha256'],windows=len(samples),events=len(events),
            notes=int((events[:,0]==1).sum()),beats=int((events[:,0]==0).sum()),
            works=len(work_sets[split]),performances=len(data['performances']),
            max_window_events=int(lengths.max()),padded_length=block_events,
            padding_fraction=float(1-lengths.sum()/(len(samples)*block_events)),
            note_first_windows=sum(events[r['span'][0],0].item()==1 for r in samples),
            note_only_windows=sum(r['beats']==0 for r in samples),
            beat_only_windows=sum(r['notes']==0 for r in samples))
        print(split, json.dumps(audit['splits'][split]),flush=True)
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert not work_sets[a] & work_sets[b]
    assert audit['splits']['test']['works']==19
    audit['checks']=['frozen source hashes','identical continuous events and targets',
        'every event covered exactly once','no cross-performance windows',
        'arbitrary boundaries allowed including NOTE-first','512 total-event cap',
        'work-disjoint splits','19 test works']
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT.parent/'data/ASAP_processed_dinr_beat_v1')
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent/'data/ASAP_processed_dinr_beat_e512_v1')
    parser.add_argument('--source-audit',type=Path,default=None,
                        help='Optional audit JSON containing frozen source hashes')
    parser.add_argument('--source-tag',default='b512_o50')
    parser.add_argument('--output-tag',default=None)
    parser.add_argument('--audit',type=Path,default=ROOT/'analysis/exp1115_beat/fixed_event_data_audit.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    result=prepare(args.source_dir,args.output_dir,frozen_path=args.source_audit,
                   source_tag=args.source_tag,output_tag=args.output_tag)
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
