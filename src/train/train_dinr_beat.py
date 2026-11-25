"""Train a beat-aware DINR experiment with periodic validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments, set_seed

from src.data_process.dinr_beat_dataset import DINRBeatCollator, DINRBeatDataset
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig
from src.train.halfepoch_protocol import BestOnlyTrainer, HalfEpochValidationCallback, LiveLossHistoryCallback


class BeatTrainer(BestOnlyTrainer):
    normalize_partial_accumulation = False

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        samples, items = super().get_batch_samples(epoch_iterator, num_batches, device)
        self.actual_accumulation_steps = max(1, len(samples))
        return samples, items

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if model.training and self.normalize_partial_accumulation:
            # Trainer 4.54 divides by configured G in training_step, including
            # the partial last update. Restore averaging over its actual r
            # microbatches for both backward and logged loss. Eval is unchanged.
            factor = self.args.gradient_accumulation_steps / self.actual_accumulation_steps
            if return_outputs:
                result = (result[0] * factor, result[1])
            else:
                result = result * factor
        return result

    def _save(self, output_dir=None, state_dict=None):
        self.model.save_pretrained(output_dir or self.args.output_dir)


class SaveAtValidationCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Transformers also requests a save on the final optimizer step.
        control.should_save &= control.should_evaluate
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sidecar", type=Path, default=Path("../data/ASAP_processed_dinr_beat_v1/dinr_beat_train_b512_o50.pt"))
    parser.add_argument("--valid-sidecar", type=Path, default=Path("../data/ASAP_processed_dinr_beat_v1/dinr_beat_valid_b512_o50.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-target", choices=["absolute_log", "log_dev"], default="absolute_log")
    parser.add_argument("--beat-duration-target", choices=["absolute_log", "log_dev"], default=None)
    parser.add_argument("--note-duration-target", choices=["absolute_log", "log_dev"], default=None)
    parser.add_argument("--duration-input-tables", choices=["shared", "separate"], default="shared")
    parser.add_argument("--duration-output-head", choices=["separate", "shared_conditioned", "shared"], default="separate")
    parser.add_argument("--measure-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder-mask-probability", type=float, default=0.5)
    parser.add_argument("--onset-target", choices=["abs", "ratio", "dev"], default="ratio")
    parser.add_argument("--epochs", type=float, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--fixed-event-length", type=int)
    parser.add_argument("--normalize-partial-accumulation", action="store_true")
    parser.add_argument("--beat-type-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, help="Validate every N optimizer steps instead of every half epoch")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--load-best-model-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--encoder-layers", type=int, default=10)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    args = parser.parse_args()
    if args.eval_steps is not None and args.eval_steps <= 0:
        parser.error('--eval-steps must be positive')
    if args.early_stopping_patience < 0:
        parser.error('--early-stopping-patience must be nonnegative; zero disables early stopping')
    if not torch.cuda.is_available():
        raise RuntimeError("Beat experiments require CUDA")
    set_seed(args.seed)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    beat_target = args.beat_duration_target or args.duration_target
    note_target = args.note_duration_target or args.duration_target
    train = DINRBeatDataset(args.train_sidecar, args.duration_target,
                            args.duration_input_tables, args.measure_start, args.limit_train,
                            beat_target, note_target, onset_target=args.onset_target)
    valid = DINRBeatDataset(args.valid_sidecar, args.duration_target,
                            args.duration_input_tables, args.measure_start, args.limit_valid,
                            beat_target, note_target, onset_target=args.onset_target)
    config = PianoT5GemmaConfig(
        encoder_layers_num=args.encoder_layers,
        decoder_layers_num=args.decoder_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.key_value_heads,
        head_dim=args.head_dim,
        torch_dtype=torch.bfloat16,
    )
    model = DINRBeatModel(config, args.duration_target,
                          args.duration_input_tables, args.duration_output_head,
                          args.decoder_mask_probability,
                          use_beat_type_embedding=args.beat_type_embedding,
                          measure_start_vocab_size=2 if args.measure_start or args.beat_type_embedding else 1,
                          onset_target=args.onset_target)
    model.beat_duration_target = beat_target
    model.note_duration_target = note_target
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update({
        "data_seed": data_seed,
        "train_windows": len(train),
        "valid_windows": len(valid),
        "duration_bins": 500,
        "duration_bin_width": 0.01,
        "beat_duration_target": beat_target,
        "note_duration_target": note_target,
        "duration_abs_log_range": [3.0, 8.0],
        "duration_log_dev_range": [-3.0, 2.0],
        "onset_input_bins": 500,
        "onset_input_range": [-0.25, 1.0],
        "onset_bins": (200 if args.onset_target == "dev" else 500),
        "onset_range": ([-1.0, 1.0] if args.onset_target == "dev" else [-0.25, 1.0]),
        "onset_target": args.onset_target,
        "sidecar_format": train.data['format'],
        "initialization": "random",
        "window_events": train.data.get("window_events"),
        "window_policy": train.data.get("window_policy", "whole_beat_note_cap"),
        "measure_start_vocab_size": model.measure_start_vocab_size,
    })
    if int(os.environ.get('RANK', '0')) == 0:
        (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
    cadence = 'steps' if args.eval_steps is not None else 'epoch'
    step_intervals = {} if args.eval_steps is None else dict(eval_steps=args.eval_steps, save_steps=args.eval_steps)
    callbacks = [SaveAtValidationCallback()] if args.eval_steps is not None else [HalfEpochValidationCallback()]
    if args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    callbacks.append(LiveLossHistoryCallback())
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=data_seed,
        eval_strategy=cadence,
        save_strategy=cadence,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        bf16=True,
        report_to="none",
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        **step_intervals,
    )
    trainer = BeatTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=DINRBeatCollator(args.fixed_event_length),
        callbacks=callbacks,
    )
    trainer.normalize_partial_accumulation = args.normalize_partial_accumulation
    trainer.train()
    if not args.load_best_model_at_end and trainer.is_world_process_zero():
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint:
            state_path = Path(best_checkpoint) / "pytorch_model.bin"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            trainer.model.load_state_dict(state)
    # Explicitly guard the final export. In multi-GPU Trainer runs, calling
    # the custom save hook from every rank can make all ranks write the same
    # 474 MB checkpoint concurrently and deadlock during teardown.
    if trainer.is_world_process_zero():
        trainer.save_model(str(args.output_dir / "best"))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(args.output_dir / "trainer_state.json"))
        (args.output_dir / "trainer_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
        )
        result = {
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_valid_loss": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "stopped_epoch": trainer.state.epoch,
        }
        (args.output_dir / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
