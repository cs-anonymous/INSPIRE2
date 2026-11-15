#!/usr/bin/env python3
"""Audit frozen continuous sidecars before launching any beat experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_process.dinr_beat_dataset import BeatTokenSpec as S


def audit(root):
    reports,works = {},{}
    for split in ('train','valid','test'):
        path = root/f'dinr_beat_{split}_b512_o50.pt'
        data = torch.load(path,map_location='cpu',weights_only=False)
        e,t = data['events'],data['targets']
        assert torch.isfinite(e).all() and torch.isfinite(t).all()
        assert (t[:,0]>0).all() and (e[:,2]>0).all()
        covered=torch.zeros(len(e),dtype=torch.bool)
        for row in data['samples']:
            a,b=row['span']
            assert e[a,0]==0 and e[b-1,0]==1
            assert int((e[a:b,0]==1).sum())==row['notes']<=512
            assert int((e[a:b,0]==0).sum())==row['beats']
            assert b==len(e) or e[b,0]==0
            covered[a:b]=True
        note=e[:,0]==1
        assert covered[note].all(), 'At least one note is missing from every window'
        works[split]={r['work'] for r in data['performances']}
        assert len(works[split])>0
        def clip(x,bounds):
            lo,hi=bounds
            return dict(count=len(x),below=float((x<lo).float().mean()),
                        above=float((x>hi).float().mean()),minimum=float(x.min()),maximum=float(x.max()))
        clipping={}
        for name,mask in [('beat',~note),('note',note)]:
            clipping[name+'_score_absolute_log']=clip(e[mask,2].log(),S.duration_abs_range)
            clipping[name+'_performance_absolute_log']=clip(t[mask,0].log(),S.duration_abs_range)
            clipping[name+'_performance_log_dev']=clip((t[mask,0]/e[mask,2]).log(),S.duration_dev_range)
        clipping['note_onset']=clip(t[note,1],S.onset_range)
        # Quantized reconstruction retains event identity. Onset error is local
        # to beat; absolute timing also accumulates duration quantization drift.
        onset=S.dequantize(S.quantize(t[note,1],S.onset_range,500),S.onset_range,500)
        reports[split]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           path=str(path.resolve()),windows=len(data['samples']),events=len(e),
                           notes=int(note.sum()),beats=int((~note).sum()),
                           performances=len(data['performances']),works=len(works[split]),
                           max_window_notes=max(r['notes'] for r in data['samples']),
                           max_window_events=max(r['span'][1]-r['span'][0] for r in data['samples']),
                           quantized_onset_mae=float((onset-t[note,1]).abs().mean()),clipping=clipping)
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        assert not works[a]&works[b], f'{a}/{b} works overlap'
    return dict(status='passed',checks=['finite positive duration','whole-beat windows <=512 notes',
                'BEAT first / NOTE last','complete note coverage','work-disjoint train/valid/test'],splits=reports)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sidecar-dir',type=Path,default=Path('../data/ASAP_processed_dinr_beat_v1'))
    parser.add_argument('--output',type=Path,default=Path('docs/design1115_beat_data_audit.json'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    result=audit(args.sidecar_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in row.items() if k not in ('clipping','sha256','path')}
                      for s,row in result['splits'].items()},indent=2))


if __name__=='__main__':
    main()
