#!/usr/bin/env python3
"""Train, render and evaluate one PianoCoRe-to-ASAP experiment per GPU."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.evaluate_exp1125_chord import evaluate
from script.prepare_exp1126 import DATA, RUN, save, sha256, write_json
from script.run_exp1125_chord import read_json
from script.run_exp1125_chord_ar_inputs import DECODING_RUNS, Experiment, render_checkpoint, verify_report
from script.train_exp1125_chord import ACCUMULATION, LR, MICRO_BATCH, SEED
from src.data_process.dinr_chord_dataset import ChordBatches, encode_targets
from src.model.dinr_chord_ar_mlp128_output import DINRChordARMLP128DLMModel, MLP128_DLM_MASK50

MODES = ('paired', 'masked', 'denoise')
MODEL = 'all_dlm_k1_mlp128'
CONTINUATION_LR = 1e-4
PAIRED_PRETRAIN = RUN / 'runs/paired/pretrain'
CONTINUATION = RUN / 'paired_continuation'


class PianoCoreBatches:
    def __init__(self, manifest, split, device):
        self.device = device
        self.windows = []
        self.shards = []
        for work in manifest['works']:
            if split not in work['splits']:
                continue
            shard = torch.load(work['path'], mmap=True, map_location='cpu', weights_only=True)
            shard_id = len(self.shards)
            self.shards.append(shard)
            spans = shard['spans'].tolist()
            self.windows.extend((shard_id, index, start, end)
                                for index, value in enumerate(work['splits']) if value == split
                                for start, end in spans)
        if not self.windows:
            raise ValueError(f'No PianoCoRe windows for {split}')

    def __len__(self):
        return len(self.windows)

    def __call__(self, indices):
        windows = [self.windows[index] for index in indices.tolist()]
        length = max(end - start for _, _, start, end in windows)
        width = (length + 7) // 8 * 8
        shape = (len(windows), width)
        pitch = torch.zeros((*shape, 128), dtype=torch.uint8, pin_memory=True)
        score = torch.zeros((*shape, 3), pin_memory=True)
        performance = torch.zeros((*shape, 3), pin_memory=True)
        valid = torch.zeros((*shape, 3), dtype=torch.bool, pin_memory=True)
        attention = torch.zeros(shape, dtype=torch.bool, pin_memory=True)
        for i, (shard, perf, start, end) in enumerate(windows):
            data = self.shards[shard]
            length = end - start
            pitch[i, :length] = data['pitches'][start:end]
            score[i, :length] = data['score'][start:end]
            performance[i, :length] = data['targets'][perf, start:end]
            valid[i, :length] = data['valid'][perf, start:end]
            attention[i, :length] = True
        return tuple(x.to(self.device, non_blocking=True) for x in (pitch, score, performance, valid, attention))


class PretrainModel(DINRChordARMLP128DLMModel):
    def __init__(self, mode):
        super().__init__(MLP128_DLM_MASK50 if mode == 'paired' else MODEL)
        self.mode = mode
        if mode == 'masked':
            self.encoder_mask = nn.Parameter(torch.zeros(3, self.bos.numel()))

    def objectives(self, batch, generator, reference):
        pitches, score, target, valid, attention = batch
        advances = score[..., 0] > 0
        encoder_mask = None
        active = valid & attention.unsqueeze(-1)
        if self.mode != 'paired':
            score = torch.where(valid, target, reference)
            score[..., 0] = torch.where(advances, score[..., 0], 0.)
        if self.mode == 'masked':
            encoder_mask = (torch.rand(attention.shape, device=score.device, generator=generator) < .3) & attention
            active = active & encoder_mask.unsqueeze(-1)
            score = torch.where(encoder_mask.unsqueeze(-1), reference, score)
            score[..., 0] = torch.where(advances, score[..., 0], 0.)
        elif self.mode == 'denoise':
            noise = torch.randn(score.shape, device=score.device, generator=generator).clamp(-3, 3)
            score = torch.cat((score[..., :2] * (.1 * noise[..., :2]).exp(),
                               (score[..., 2:] + (4 * noise[..., 2:]).round()).clamp(1, 127)), dim=-1)
        labels = encode_targets(score.flatten(0, 1), target.flatten(0, 1), valid[..., 0].flatten(),
                                'log_dev', timing_bins=500).reshape_as(target)
        labels = labels.masked_fill(~valid, -100)
        return pitches, score, attention, advances, labels, active, encoder_mask

    def pretraining_losses(self, batch, generator, reference):
        pitches, score, attention, advances, labels, active, masked = self.objectives(batch, generator, reference)
        context = None
        if masked is not None:
            pitch, *fields = self.score_embeddings(pitches, score)
            inputs = pitch + sum(torch.where(masked.unsqueeze(-1), mask.to(field.dtype), field)
                                 for mask, field in zip(self.encoder_mask, fields, strict=True))
            context = self.encoder(inputs_embeds=inputs, attention_mask=attention.long()).last_hidden_state
        hidden = self.teacher_forced_hidden(pitches, score, attention, advances, labels, context=context)
        losses = [self.field_loss(slot, head(hidden[active[..., slot]]), labels[..., slot][active[..., slot]])
                  for slot, head in enumerate((self.ioi_head, self.duration_head, self.velocity_head))]
        return torch.stack(losses), active.sum(dim=(0, 1))

    def export(self):
        return {key: value for key, value in self.state_dict().items() if key != 'encoder_mask'}


def subset(data, split, rows):
    data.spans = data.spans[torch.tensor(split[rows], dtype=torch.long)]
    return data


def asap_data(device):
    split = read_json(RUN / 'asap_split.json')
    payload = torch.load(DATA / 'train.pt', map_location='cpu', weights_only=True, mmap=True)
    batches = [subset(ChordBatches(payload, 'log_dev', device, raw_score=True, timing_bins=500), split, name)
               for name in ('train', 'valid')]
    return batches


@torch.no_grad()
def calibration(model, data, device):
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED))[:512]
    def samples():
        for block in indices.split(MICRO_BATCH):
            yield data(block)
    values = [[], [], []]
    for _, _, target, valid, _ in samples():
        for slot in range(3):
            values[slot].append(target[..., slot][valid[..., slot]].cpu())
    reference = torch.stack([torch.cat(field).median() for field in values]).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    arrays = {'score': [], 'targets': [], 'ioi_valid': []}
    for batch in samples():
        _, score, attention, advances, _, active, _ = model.objectives(batch, generator, reference)
        use = active[..., 1] & attention
        arrays['score'].append(score[use].cpu())
        arrays['targets'].append(batch[2][use].cpu())
        arrays['ioi_valid'].append((active[..., 0] & advances)[use].cpu())
    report = model.initialize_from_training({key: torch.cat(value) for key, value in arrays.items()})
    return reference, report


@torch.inference_mode()
def validate(model, data, loss_fn, indices):
    model.eval()
    sums = torch.zeros(3, device='cuda')
    counts = torch.zeros_like(sums)
    for block in indices.split(MICRO_BATCH):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, count = loss_fn(data(block))
        sums += loss
        counts += count
    model.train()
    values = (sums / counts.clamp_min(1)).tolist()
    return sum(values), counts.int().tolist()


def train(mode, stage, output, smoke, initial=None, continuation=False):
    output.mkdir(parents=True, exist_ok=True)
    completion_file = output / 'training_complete.json'
    if completion_file.is_file():
        completion = read_json(completion_file)
        if sha256(Path(completion['checkpoint'])) != completion['sha256']:
            raise ValueError('Checkpoint digest differs')
        return completion
    set_seed(SEED)
    device = torch.device('cuda:0')
    rng = torch.Generator(device=device).manual_seed(SEED + 1)
    best_path = output / 'best.pt'
    if stage == 'pretrain':
        manifest = read_json(RUN / ('smoke_data' if smoke and not continuation else 'data') / 'manifest.json')
        data, valid = (PianoCoreBatches(manifest, split, device) for split in ('train', 'valid'))
        model = PretrainModel(mode)
        reference, initialization = calibration(model, data, device)
        loss_fn = lambda batch: model.pretraining_losses(batch, rng, reference)
        validation_fn = lambda batch: model.pretraining_losses(batch, validation_rng, reference)
        epochs, eval_steps = 1, 500
    else:
        data, valid = asap_data(device)
        model = DINRChordARMLP128DLMModel(MLP128_DLM_MASK50)
        model.load_state_dict(torch.load(initial['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        initialization = initial['sha256']
        reference = None
        epochs, eval_steps = 4, 50
        loss_fn = validation_fn = model.losses
    model.to(device)
    model.history_mask_rng = torch.Generator(device=device).manual_seed(SEED)
    validation_rng = torch.Generator(device=device)
    per_epoch = math.ceil(len(data) / (MICRO_BATCH * ACCUMULATION))
    steps = 2 if smoke else epochs * per_epoch
    learning_rate = LR
    schedule_steps = steps
    milestones = {}
    if continuation:
        original = read_json(PAIRED_PRETRAIN / 'training_complete.json')
        if original['optimizer_steps'] != per_epoch or original['best_step'] != per_epoch:
            raise ValueError('Expected the original paired checkpoint at the end of its first epoch')
        epochs = 2
        schedule_steps = 4 if smoke else per_epoch
        steps = per_epoch + schedule_steps
        milestones = {per_epoch + math.ceil(schedule_steps / 2): 'epoch1p5', steps: 'epoch2'}
        learning_rate = CONTINUATION_LR
        initialization = original['sha256']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0., fused=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, schedule_steps)
    valid_ids = torch.randperm(len(valid), generator=torch.Generator().manual_seed(SEED))[
        :16 if smoke else (512 if stage == 'pretrain' else len(valid))]
    write_json(output / 'run_config.json', dict(mode=mode, stage=stage, training_windows=len(data),
               validation_windows=len(valid_ids), optimizer_steps=steps, epochs=epochs, learning_rate=learning_rate,
               seed=SEED, initialization=initialization, history_mask_probability=model.history_mask_probability,
               encoder_mask_probability=.3 if mode == 'masked' and stage == 'pretrain' else 0.,
               timing_reference=None if reference is None else reference.tolist(), micro_batch=MICRO_BATCH,
               accumulation=ACCUMULATION, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               continuation_steps=schedule_steps if continuation else None,
               continuation_milestones=milestones))
    resume = output / 'resume.pt'
    start_step, best_step, best_loss, best_digest, best_file = 0, None, math.inf, None, best_path
    supervised = torch.zeros(3, device=device, dtype=torch.int64)
    restoring = resume.is_file()
    if restoring or continuation:
        source = resume if restoring else PAIRED_PRETRAIN / 'resume.pt'
        state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if restoring:
            scheduler.load_state_dict(state['scheduler'])
            best_step, best_loss, best_digest = (state[key] for key in ('best_step', 'best_loss', 'best_digest'))
            best_file = Path(state['best_file'])
        else:
            if state['step'] != per_epoch or state['best_digest'] != original['sha256']:
                raise ValueError('Original continuation state differs from the scored checkpoint')
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] = learning_rate
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, schedule_steps)
        rng.set_state(state['rng'])
        model.history_mask_rng.set_state(state['history_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        start_step = state['step']
        supervised.copy_(state['supervised'])
        del state
    order_rng = torch.Generator().manual_seed(SEED)
    training_started = time.monotonic()
    if continuation:
        print(f'Resuming paired pretraining at {start_step:,} through {steps:,} with lr {scheduler.get_last_lr()[0]:g}',
              flush=True)
    with (output / 'loss_history.jsonl').open('a') as log:
        def record(row):
            row['elapsed_s'] = time.monotonic() - training_started
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            print(row, flush=True)

        for epoch in range(epochs):
            order = torch.randperm(len(data), generator=order_rng)
            for i, block in enumerate(order.split(MICRO_BATCH * ACCUMULATION)):
                step = epoch * per_epoch + i + 1
                if step <= start_step:
                    continue
                if step > steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                minibatches = block.split(MICRO_BATCH)
                total = torch.zeros(3, device=device)
                for indices in minibatches:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        sums, counts = loss_fn(data(indices))
                        means = sums / counts.clamp_min(1)
                    (means.sum() / len(minibatches)).backward()
                    total += means.detach()
                    supervised += counts
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                if step <= 5 or step == start_step + 1 or step % 25 == 0 or step in milestones or step == steps:
                    record(dict(step=step, epoch=epoch + 1, train_loss=float(total.sum() / len(minibatches)),
                                learning_rate=scheduler.get_last_lr()[0]))
                if step % eval_steps == 0 or step in milestones or step == steps:
                    validation_rng.manual_seed(SEED + 2)
                    loss, counts = validate(model, valid, validation_fn, valid_ids)
                    if loss < best_loss:
                        best_step, best_loss = step, loss
                        best_file = output / f'best_{step}.pt'
                        save(best_file, model.export() if stage == 'pretrain' else model.state_dict())
                        best_digest = sha256(best_file)
                    if step in milestones:
                        snapshot = output.parent / milestones[step] / 'pretrain'
                        snapshot.mkdir(parents=True, exist_ok=True)
                        checkpoint = snapshot / 'best.pt'
                        if step == best_step:
                            checkpoint.unlink(missing_ok=True)
                            os.link(best_file, checkpoint)
                            digest = best_digest
                        else:
                            save(checkpoint, model.export())
                            digest = sha256(checkpoint)
                        write_json(snapshot / 'training_complete.json', dict(checkpoint=str(checkpoint), sha256=digest,
                                   best_step=step, best_valid_loss=loss, optimizer_steps=step,
                                   supervised_fields=supervised.tolist()))
                    save(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), rng=rng.get_state(), history_rng=model.history_mask_rng.get_state(),
                         cuda_rng=torch.cuda.get_rng_state(), step=step, best_step=best_step, best_loss=best_loss,
                         best_digest=best_digest, best_file=str(best_file), supervised=supervised))
                    for old in output.glob('best_*.pt'):
                        if old != best_file:
                            old.unlink()
                    record(dict(step=step, epoch=epoch + 1, valid_loss=loss, valid_fields=counts, best_step=best_step))
    best_path.unlink(missing_ok=True)
    os.link(best_file, best_path)
    completion = dict(checkpoint=str(best_path), sha256=best_digest, best_step=best_step,
                      best_valid_loss=best_loss, optimizer_steps=steps, supervised_fields=supervised.tolist())
    write_json(completion_file, completion)
    return completion


def predict(mode, output, checkpoint, gpu):
    folder = output / 'test_official'
    if not all((folder / f'{kind}.json').is_file() for kind in ('median', 'sampled')):
        experiment = Experiment(output, (MODEL,), 'mlp128_output', 'ar_mlp128_output', 900000,
                                first_gpu=gpu, shared_sampling_seed_gpu=2)
        render_checkpoint(experiment, MODEL, output, checkpoint)
    predictions = {}
    for kind, draws in DECODING_RUNS.items():
        report = read_json(folder / f'{kind}.json')
        verify_report(report, checkpoint, draws)
        predictions[f'{mode} {output.name} {kind}'] = report
    return predictions


def evaluate_pipeline(mode, root, initial, gpu, smoke):
    pretrain, sft = root / 'pretrain', root / 'sft'
    torch.cuda.empty_cache()
    predictions = predict(mode, pretrain, initial, gpu)
    if not (pretrain / 'multilevel_metrics.json').exists():
        evaluate(levels='all', extra_chord_manifests=predictions, output=pretrain)
    best = train(mode, 'sft', sft, smoke, initial=initial)
    torch.cuda.empty_cache()
    predictions.update(predict(mode, sft, best, gpu))
    evaluate(levels='all', extra_chord_manifests=predictions, output=sft)
    write_json(root / 'complete.json', {'pretrain_metrics': str(pretrain / 'multilevel_metrics.csv'),
                                       'sft_metrics': str(sft / 'multilevel_metrics.csv')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--smoke', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--continue-pretrain', action='store_true')
    action.add_argument('--evaluate-continuation', choices=('epoch1p5', 'epoch2'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if args.continue_pretrain or args.evaluate_continuation:
        if args.mode != 'paired':
            parser.error('Continuation requires paired pretraining')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible.isdecimal():
            parser.error('Continuation requires one GPU in CUDA_VISIBLE_DEVICES')
        root = CONTINUATION / ('smoke' if args.smoke else 'runs')
        if args.continue_pretrain:
            train('paired', 'pretrain', root / 'pretrain', args.smoke, continuation=True)
        else:
            root /= args.evaluate_continuation
            initial = read_json(root / 'pretrain/training_complete.json')
            evaluate_pipeline(f'paired {args.evaluate_continuation}', root, initial, int(visible), args.smoke)
        return
    gpu = MODES.index(args.mode)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        parser.error(f'{args.mode} must run with CUDA_VISIBLE_DEVICES={gpu}')
    root = RUN / ('smoke' if args.smoke else 'runs') / args.mode
    best = train(args.mode, 'pretrain', root / 'pretrain', args.smoke)
    evaluate_pipeline(args.mode, root, best, gpu, args.smoke)


if __name__ == '__main__':
    main()
