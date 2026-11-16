#!/usr/bin/env python3
"""Isolated worst-window memory check; does not change any campaign model."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.data_process.dinr_beat_dataset import DINRBeatDataset,DINRBeatCollator
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data=DINRBeatDataset('../data/ASAP_processed_dinr_beat_v1/dinr_beat_train_b512_o50.pt')
    index=max(range(len(data.samples)),key=lambda i:data.samples[i]['span'][1]-data.samples[i]['span'][0])
    example=data[index]
    result=dict(microbatch=args.batch_size,accumulation=32//args.batch_size,
                true_events=len(example['event_types']),device=torch.cuda.get_device_name(0),
                protocol='random disposable full-size A1 model; repeat longest training window; two optimizer updates; no campaign state altered')
    try:
        batch={k:v.cuda() for k,v in DINRBeatCollator()([example]*args.batch_size).items()}
        result['padded_events']=batch['event_types'].shape[-1]
        model=DINRBeatModel(PianoT5GemmaConfig(encoder_layers_num=10,decoder_layers_num=2)).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(result['accumulation']):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(**batch)['loss']/result['accumulation']
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        torch.cuda.synchronize()
        result.update(status='passed',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      seconds_per_optimizer_update=(time.monotonic()-started)/2)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='out_of_memory',error=str(error),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
