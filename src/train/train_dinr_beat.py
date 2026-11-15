"""Train one beat-aware DINR experiment with half-epoch validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from src.data_process.dinr_beat_dataset import DINRBeatCollator, DINRBeatDataset
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig
from src.train.halfepoch_protocol import BestOnlyTrainer, HalfEpochValidationCallback, LiveLossHistoryCallback


class BeatTrainer(BestOnlyTrainer):
    def _save(self, output_dir=None, state_dict=None):
        self.model.save_pretrained(output_dir or self.args.output_dir)


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
    parser.add_argument("--epochs", type=float, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
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
    if not torch.cuda.is_available():
        raise RuntimeError("Beat experiments require CUDA")
    set_seed(args.seed)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    beat_target = args.beat_duration_target or args.duration_target
    note_target = args.note_duration_target or args.duration_target
    train = DINRBeatDataset(args.train_sidecar, args.duration_target,
                            args.duration_input_tables, args.measure_start, args.limit_train,
                            beat_target, note_target)
    valid = DINRBeatDataset(args.valid_sidecar, args.duration_target,
                            args.duration_input_tables, args.measure_start, args.limit_valid,
                            beat_target, note_target)
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
                          args.decoder_mask_probability)
    model.beat_duration_target = beat_target
    model.note_duration_target = note_target
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update({
        "data_seed": data_seed,
        "train_windows": len(train),
        "valid_windows": len(valid),
        "duration_bins": 500,
        "beat_duration_target": beat_target,
        "note_duration_target": note_target,
        "duration_abs_log_range": [3.0, 8.0],
        "duration_log_dev_range": [-2.0, 3.0],
        "onset_bins": 500,
        "onset_range": [-0.25, 1.0],
        "initialization": "random",
    })
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
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
    )
    trainer = BeatTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=DINRBeatCollator(),
        callbacks=[HalfEpochValidationCallback(),
                   EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
                   LiveLossHistoryCallback()],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
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
