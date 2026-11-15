"""Beat DINR validation, typed AR/TF decoding, loss diagnostics and MIDI export.

All token metrics retain score event identity. Legacy PN/PP Wasserstein and HR
are separately computed from time-sorted MIDI, using the exp1112 human anchors.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from miditoolkit import Instrument, MidiFile, Note, TempoChange

from src.data_process.dinr_beat_dataset import BeatTokenSpec as S, DINRBeatDataset, DINRBeatCollator
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig

ANCHORS = dict(ioi=22.772456164440005, duration=75.08565114668764, velocity=10.580359191566208)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    run_path = checkpoint.parent / 'run_config.json'
    if not run_path.exists():
        run_path = checkpoint / 'run_config.json'
    run = json.loads(run_path.read_text())
    config = PianoT5GemmaConfig(
        encoder_layers_num=run['encoder_layers'], decoder_layers_num=run['decoder_layers'],
        hidden_size=run['hidden_size'], intermediate_size=run['intermediate_size'],
        num_attention_heads=run['attention_heads'], num_key_value_heads=run['key_value_heads'],
        head_dim=run['head_dim'])
    model = DINRBeatModel(config, run['duration_target'], run['duration_input_tables'],
                          run['duration_output_head'], run.get('decoder_mask_probability', .5))
    model.beat_duration_target = run['beat_duration_target']
    model.note_duration_target = run['note_duration_target']
    model.load_state_dict(torch.load(checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True))
    return model.to(device).eval(), run


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def dataset_for(sidecar, run):
    return DINRBeatDataset(sidecar, run['duration_target'], run['duration_input_tables'],
                           run['measure_start'], beat_duration_target=run['beat_duration_target'],
                           note_duration_target=run['note_duration_target'])


def decode_duration(tokens, events, model):
    tokens = torch.as_tensor(tokens)
    events = torch.as_tensor(events, device=tokens.device)
    is_beat = events[..., 0] == S.beat_type
    result = torch.zeros_like(tokens, dtype=torch.float32)
    for selected, mode in ((is_beat, model.beat_duration_target), (~is_beat, model.note_duration_target)):
        bounds = S.duration_abs_range if mode == 'absolute_log' else S.duration_dev_range
        value = S.dequantize(tokens, bounds, S.duration_bins)
        if mode == 'log_dev':
            value = value + events[..., 2].clamp_min(1).log()
        result = torch.where(selected, value, result)
    return result.exp()


def draw(logits, generator, temperature=1., top_p=.9):
    values, indices = (logits.float() / temperature).sort(descending=True, dim=-1)
    probability = values.softmax(dim=-1)
    values = values.masked_fill(probability.cumsum(-1) - probability > top_p, -torch.inf)
    selected = torch.multinomial(values.softmax(dim=-1), 1, generator=generator)
    return indices.gather(-1, selected).squeeze(-1)


def event_chunks(events, limit=512):
    """Nonoverlapping, whole-beat chunks; no more than limit notes."""
    types = events[:,0].long()
    last_note = torch.nonzero(types == S.note_type).flatten()
    if not len(last_note):
        return []
    end = int(last_note[-1]) + 1
    beats = torch.nonzero(types[:end] == S.beat_type).flatten().tolist() + [end]
    chunks, start, count = [], 0, 0
    for left,right in zip(beats[:-1],beats[1:]):
        notes = int((types[left:right] == S.note_type).sum())
        if notes > limit:
            raise ValueError('Single beat exceeds note cap')
        if count + notes > limit:
            chunks.append((start,left))
            start, count = left, 0
        count += notes
    chunks.append((start,end))
    return chunks


def quantized_example(events, targets, run):
    # Reuse exactly the training quantizer/shift logic, without writing a file.
    ds = object.__new__(DINRBeatDataset)
    ds.data = dict(events=events, targets=targets)
    ds.samples = [dict(span=[0,len(events)])]
    ds.duration_target = run['duration_target']
    ds.beat_duration_target = run['beat_duration_target']
    ds.note_duration_target = run['note_duration_target']
    ds.input_duration_tables = run['duration_input_tables']
    ds.input_measure_start = run['measure_start']
    return ds[0]


@torch.inference_mode()
def predict(model, events, targets, run, device, seed, teacher_forced=False, greedy=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks = event_chunks(events, run.get('window_notes',512))
    results = []
    for left,right in chunks:
        source = events[left:right]
        example = quantized_example(source, targets[left:right], run)
        encoder = example['encoder_tokens'][None].to(device)
        types = example['event_types'][None].to(device)
        attention = torch.ones_like(types)
        with amp(device):
            if teacher_forced:
                output = model(encoder_tokens=encoder, decoder_tokens=example['decoder_tokens'][None].to(device),
                               event_types=types, attention_mask=attention)
                predicted = []
                for key in ('duration','onset','velocity'):
                    logits = output['logits'][key][0]
                    predicted.append(logits.argmax(-1) if greedy else draw(logits,generator))
                results.append(torch.stack(predicted,dim=-1).cpu())
                continue
            encoded = model.backbone.encoder(inputs_embeds=model._sum_events(encoder), attention_mask=attention)
            previous = torch.tensor([S.bos_type,0,0,0,0,0], device=device).view(1,1,6)
            cache, predicted = None, []
            for i in range(len(source)):
                output = model(encoder_tokens=encoder, decoder_tokens=previous,
                               event_types=types[:,i:i+1], attention_mask=attention,
                               decoder_attention_mask=torch.ones((1,i+1),dtype=torch.long,device=device),
                               encoder_outputs=encoded, use_cache=True, past_key_values=cache,
                               cache_position=torch.tensor([i],device=device))
                cache = output['past_key_values']
                row = torch.stack([output['logits'][key][0,0].argmax(-1) if greedy
                                   else draw(output['logits'][key][0,0],generator)
                                   for key in ('duration','onset','velocity')])
                predicted.append(row)
                previous = encoder[:,i:i+1].clone()
                duration = decode_duration(row[0],source[i],model)
                previous[...,2] = S.quantize(duration.log(),S.duration_abs_range,S.duration_bins)
                previous[...,3] = row[1]
                previous[...,4] = row[2]
            results.append(torch.stack(predicted).cpu())
    return torch.cat(results)



@torch.inference_mode()
def predict_many(model, events, targets, run, device, seeds, batch_size=16):
    """Batch independent complete-beat windows and sampling draws for AR."""
    chunks = event_chunks(events,run.get('window_notes',512))
    tasks = [(sample,left,right) for sample in range(len(seeds)) for left,right in chunks]
    pieces = [[] for _ in seeds]
    # Fixed task order and batch size are part of the replay protocol.
    generator = torch.Generator(device=device).manual_seed(seeds[0])
    for offset in range(0,len(tasks),batch_size):
        group = tasks[offset:offset+batch_size]
        examples = [quantized_example(events[left:right],targets[left:right],run)
                    for _,left,right in group]
        batch = {k:v.to(device) for k,v in DINRBeatCollator()(examples).items()}
        encoder,types,attention = batch['encoder_tokens'],batch['event_types'],batch['attention_mask']
        size,length = types.shape
        raw = torch.ones((size,length,6),device=device)
        for index,(_,left,right) in enumerate(group):
            raw[index,:right-left] = events[left:right].to(device)
        previous = torch.zeros((size,1,6),dtype=torch.long,device=device)
        previous[...,0] = S.bos_type
        cache,predicted = None,[]
        with amp(device):
            encoded = model.backbone.encoder(inputs_embeds=model._sum_events(encoder),attention_mask=attention)
            for i in range(length):
                output=model(encoder_tokens=encoder,decoder_tokens=previous,event_types=types[:,i:i+1],
                             attention_mask=attention,decoder_attention_mask=attention[:,:i+1],
                             encoder_outputs=encoded,use_cache=True,past_key_values=cache,
                             cache_position=torch.tensor([i],device=device))
                cache=output['past_key_values']
                row=torch.stack([draw(output['logits'][k][:,0],generator)
                                 for k in ('duration','onset','velocity')],dim=-1)
                predicted.append(row)
                previous=encoder[:,i:i+1].clone()
                duration=decode_duration(row[:,0],raw[:,i],model)
                previous[...,2]=S.quantize(duration.log(),S.duration_abs_range,S.duration_bins)[:,None]
                previous[...,3]=row[:,1,None]
                previous[...,4]=row[:,2,None]
        predicted=torch.stack(predicted,dim=1).cpu()
        for index,(sample,left,right) in enumerate(group):
            pieces[sample].append(predicted[index,:right-left])
    return [torch.cat(part) for part in pieces]


def to_continuous(tokens, events, model):
    duration = decode_duration(tokens[:,0],events,model).cpu().numpy()
    onset = S.dequantize(tokens[:,1],S.onset_range,S.onset_bins).numpy()
    return np.column_stack((duration,onset,tokens[:,2].numpy()))


def absolute_notes(events, continuous):
    """Reconstruct beat curve and note times without clipping/reordering notes."""
    e, values = np.asarray(events), np.asarray(continuous)
    beat = e[:,0] == S.beat_type
    beat_durations = values[beat,0]
    anchors = np.r_[0.,np.cumsum(beat_durations)[:-1]]
    membership = np.cumsum(beat) - 1
    note = ~beat
    onsets = anchors[membership[note]] + values[note,1] * beat_durations[membership[note]]
    return np.column_stack((e[note,1],onsets,values[note,0],values[note,2])), anchors


def midi_arrays(notes):
    # MIDI is time sorted, just like the legacy HR evaluator. Retain score IDs
    # separately in npz; never confuse this sorting with token-level alignment.
    order = np.lexsort((notes[:,0],notes[:,1]))
    n = notes[order]
    onset = n[:,1] - n[:,1].min()
    return np.column_stack((np.diff(onset,prepend=0.),n[:,2],np.clip(n[:,3],1,127)))


def export_midi(notes, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    midi = MidiFile(ticks_per_beat=1000)
    midi.tempo_changes = [TempoChange(60,0)]
    instrument = Instrument(0)
    shift = -min(float(notes[:,1].min()),0.)
    for pitch,onset,duration,velocity in notes:
        start = max(0,int(round(onset+shift)))
        instrument.notes.append(Note(max(1,min(127,int(velocity))),int(pitch),start,
                                     start+max(1,int(round(duration)))))
    instrument.notes.sort(key=lambda n:(n.start,n.pitch,n.end))
    midi.instruments = [instrument]
    midi.dump(str(path))


def js_counts(p,q,bins):
    p = np.bincount(np.asarray(p).astype(int).ravel(),minlength=bins).astype(float)
    q = np.bincount(np.asarray(q).astype(int).ravel(),minlength=bins).astype(float)
    p,q = p/p.sum(),q/q.sum()
    m = (p+q)/2
    return float((rel_entr(p,m).sum()+rel_entr(q,m).sum())/2)


def token_metrics(tokens, reference_tokens, events, continuous, reference):
    beat = events[:,0].numpy() == S.beat_type
    result = {}
    for name,mask,col,bins in (('beat_duration',beat,0,500),('note_duration',~beat,0,500),
                               ('onset',~beat,1,500),('velocity',~beat,2,128)):
        pred, ref = tokens.numpy()[mask,col], reference_tokens.numpy()[mask,col]
        result[name+'_accuracy'] = float(np.mean(pred == ref))
        result[name+'_js'] = js_counts(pred,ref,bins)
        if col == 0:
            error = np.log(continuous[mask,0]/reference[mask,0])
            result[name+'_log_mae'] = float(np.abs(error).mean())
            result[name+'_ratio_mae'] = float(np.abs(continuous[mask,0]/reference[mask,0]-1).mean())
        else:
            error = continuous[mask,col]-reference[mask,col]
            result[name+'_mae'] = float(np.abs(error).mean())
            result[name+'_signed_error'] = float(error.mean())
    _,pred_anchors = absolute_notes(events,continuous)
    _,ref_anchors = absolute_notes(events,reference)
    result['beat_anchor_mae_ms'] = float(np.abs(pred_anchors-ref_anchors).mean())
    result['selection_score'] = (result['beat_duration_log_mae'] + result['note_duration_log_mae']
                                 + result['onset_mae'] + result['velocity_mae']/127)/4
    return result


@torch.inference_mode()
def evaluate_valid(model,run,path,device,limit=None):
    dataset = dataset_for(path,run)
    rows,losses = [],[]
    for i in range(min(len(dataset),limit or len(dataset))):
        ex = dataset[i]
        start,end = dataset.samples[i]['span']
        events,reference = dataset.data['events'][start:end],dataset.data['targets'][start:end].numpy()
        batch = {k:v.to(device) for k,v in DINRBeatCollator()([ex]).items()}
        with amp(device):
            output = model(**batch)
        losses.append(float(output['loss']))
        tokens = torch.stack([output['logits'][k][0,:len(events)].argmax(-1).cpu()
                              for k in ('duration','onset','velocity')],dim=-1)
        labels = ex['labels']
        ref_tokens = torch.stack((torch.where(ex['event_types']==0,labels[:,0],labels[:,1]),
                                  labels[:,2].clamp_min(0),labels[:,3].clamp_min(0)),dim=-1)
        rows.append(token_metrics(tokens,ref_tokens,events,to_continuous(tokens,events,model),reference))
    return dict(windows=len(rows),loss=float(np.mean(losses)),
                metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0]})


def reference_tokens(events,targets,run):
    example = quantized_example(events,targets,run)
    labels = example['labels']
    return torch.stack((torch.where(example['event_types']==0,labels[:,0],labels[:,1]),
                        labels[:,2].clamp_min(0),labels[:,3].clamp_min(0)),dim=-1)


def evaluate_test(model,run,path,device,output,samples=3,limit=None):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    groups = {}
    for row in payload['performances']:
        groups.setdefault(row['work'],[]).append(row)
    work_rows = []
    for work_index,(work,performances) in enumerate(sorted(groups.items())):
        if limit and work_index >= limit:
            break
        work_dir = output / f'work_{work_index:02d}'
        work_dir.mkdir(parents=True,exist_ok=True)
        ready = work_dir / 'metrics.json'
        if ready.exists():
            work_rows.append(json.loads(ready.read_text()))
            continue
        first_start,first_end = performances[0]['span']
        events = payload['events'][first_start:first_end]
        chunks = event_chunks(events,run.get('window_notes',512))
        length = chunks[-1][1]
        events = events[:length]
        references, reference_token_arrays, reference_midis = [],[],[]
        for ref_index,row in enumerate(performances):
            start,end = row['span']
            if not torch.equal(payload['events'][start:start+length],events):
                raise ValueError('Reference performances have different score event topology')
            reference = payload['targets'][start:start+length]
            references.append(reference)
            reference_token_arrays.append(reference_tokens(events,reference,run))
            notes,_ = absolute_notes(events,reference.numpy())
            export_midi(notes,work_dir/f'reference_{ref_index:03d}.mid')
            reference_midis.append(midi_arrays(notes))
        refs = torch.stack(references)
        ref_tokens = torch.stack(reference_token_arrays)
        ar_seeds = [int.from_bytes(hashlib.sha256(f'1115:{work}:{s}:AR-batch16'.encode()).digest()[:4],'little') for s in range(samples)]
        ar_predictions = predict_many(model,events,refs[0],run,device,ar_seeds)
        mode_rows = []
        for mode in ('teacher_forced','autoregressive'):
            for sample in range(samples):
                histories = range(len(refs)) if mode == 'teacher_forced' else range(1)
                metrics, ar_tokens = [],[]
                for history in histories:
                    seed = int.from_bytes(hashlib.sha256(f'1115:{work}:{sample}:{history}:{mode}'.encode()).digest()[:4],'little')
                    tokens = (predict(model,events,refs[history],run,device,seed,True)
                              if mode == 'teacher_forced' else ar_predictions[sample])
                    ar_tokens.append(tokens.numpy())
                    continuous = to_continuous(tokens,events,model)
                    notes,_ = absolute_notes(events,continuous)
                    export_midi(notes,work_dir/f'{mode}_{sample:02d}_history_{history:03d}.mid')
                    array = midi_arrays(notes)
                    comparison = [token_metrics(tokens,rt,events,continuous,r.numpy())
                                  for rt,r in zip(ref_tokens,refs)]
                    row_metrics = {k:float(np.mean([r[k] for r in comparison])) for k in comparison[0]}
                    for col,name in enumerate(('ioi','duration','velocity')):
                        # W1(single generated value, human values) equals mean absolute error.
                        human = np.stack([r[:,col] for r in reference_midis])
                        pn = float(np.abs(array[None,:,col]-human).mean())
                        pp = float(np.mean([wasserstein_distance(array[:,col],r[:,col]) for r in reference_midis]))
                        row_metrics[name+'_pn_w1'] = pn
                        row_metrics[name+'_pp_w1'] = pp
                        row_metrics[name+'_hr'] = pn/ANCHORS[name]
                    row_metrics['pn_hr'] = float(np.mean([row_metrics[n+'_hr'] for n in ANCHORS]))
                    metrics.append(row_metrics)
                np.savez_compressed(work_dir/f'{mode}_{sample:02d}_tokens.npz',
                                    predictions=np.stack(ar_tokens),references=ref_tokens.numpy(),events=events.numpy())
                mode_rows.append(dict(mode=mode,sample=sample,metrics={k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}))
        result = dict(work=work,notes=int((events[:,0]==1).sum()),beats=int((events[:,0]==0).sum()),
                      references=len(refs),rows=mode_rows)
        atomic_json(ready,result)
        work_rows.append(result)
        print(f'completed test work {work_index+1}/{len(groups)}: {work}',flush=True)
    rounds = []
    for mode in ('teacher_forced','autoregressive'):
        for sample in range(samples):
            rows = [next(r['metrics'] for r in w['rows'] if r['mode']==mode and r['sample']==sample) for w in work_rows]
            rounds.append(dict(mode=mode,sample=sample,metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
    return dict(works=len(work_rows),samples=samples,rounds=rounds,work_rows=work_rows,
                hr_protocol='legacy MIDI time-sorted note-index PN W1 / exp1112 fixed anchors; average per work',
                token_protocol='exact raw output token, score event identity retained; JS in nats',
                generation_protocol='whole beat nonoverlapping windows <=512 notes; BOS/cache reset per window; beat timeline continuous; AR task batch size 16; deterministic work RNG shared across draws')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--split',choices=['valid','test'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    model,run=load_model(args.checkpoint,device)
    result=(evaluate_valid(model,run,args.sidecar,device,args.limit) if args.split=='valid' else
            evaluate_test(model,run,args.sidecar,device,args.output_dir,args.samples,args.limit))
    atomic_json(args.output_dir/'metrics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('work_rows','rounds')},indent=2),flush=True)


if __name__=='__main__':
    main()
