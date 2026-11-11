"""Train the clean note-level DINR absolute-timing baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments
from transformers import set_seed

from src.data_process.dinr_dataset import DINRCollator, DINRAbsoluteDataset
from src.model.dinr import DINRAbsoluteModel
from src.model.pianoformer import PianoT5GemmaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, default=Path("../data/ASAP_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/dinr_absolute"))
    parser.add_argument("--block-notes", type=int, default=512)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pretrained-pt", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--encoder-layers", type=int, default=10)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--max-train-performances", type=int, default=None)
    parser.add_argument("--max-valid-performances", type=int, default=None)
    parser.add_argument("--precomputed-train", type=Path, default=None)
    parser.add_argument("--precomputed-valid", type=Path, default=None)
    parser.add_argument("--eval-split", choices=["none", "train", "valid", "test"], default="none")
    parser.add_argument("--pedal-mode", choices=["ce", "bce", "none"], default="ce")
    parser.add_argument("--numerical-coordinates", action="store_true",
                        help="Add Fourier numerical coordinates to timing/velocity token embeddings.")
    parser.add_argument("--coordinate-frequencies", type=int, default=16)
    parser.add_argument("--decoder-mask-probability", type=float, default=0.0,
                        help="Probability of masking each selected decoder note or attribute.")
    parser.add_argument("--decoder-mask-mode",
                        choices=["note", "attribute", "timestep", "timestep_single"],
                        default="note",
                        help="Mask full notes, individual attributes, or timestep attributes while optionally preserving pitch.")
    parser.add_argument("--separate-ioi-duration-token-tables", action="store_true",
                        help="Use independent categorical embedding tables for IOI and duration.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-scheduler-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--timing-target-mode", choices=["absolute", "log_absolute", "score_relative_log"], default="absolute")
    parser.add_argument("--time-input-mode", choices=["linear", "log_absolute"], default="linear",
                        help="Encoding for timing values supplied to the encoder and shifted decoder inputs.")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="Disable checkpoint and final model writes for diagnostic runs.")
    parser.add_argument("--load-best-model-at-end", action="store_true",
                        help="Restore the lowest-eval-loss checkpoint after training.")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="Maximum number of epoch checkpoints to retain.")
    parser.add_argument("--skip-final-model-save", action="store_true",
                        help="Keep epoch checkpoints without writing a duplicate model at output root.")
    args = parser.parse_args()
    set_seed(args.seed)

    train = DINRAbsoluteDataset(
        args.sidecar_root, "train", args.block_notes, args.overlap_ratio,
        max_performances=args.max_train_performances,
        include_pedal=args.pedal_mode != "none",
        precomputed_path=args.precomputed_train,
        timing_target_mode=args.timing_target_mode,
        time_input_mode=args.time_input_mode,
    )
    valid = None
    if args.eval_split != "none":
        valid = DINRAbsoluteDataset(
            args.sidecar_root, args.eval_split, args.block_notes, args.overlap_ratio,
            max_performances=args.max_valid_performances,
            include_pedal=args.pedal_mode != "none",
            precomputed_path=args.precomputed_valid,
            timing_target_mode=args.timing_target_mode,
            time_input_mode=args.time_input_mode,
        )
    if args.load_best_model_at_end and valid is None:
        raise ValueError("--load-best-model-at-end requires --eval-split valid|train|test")
    if args.limit_train:
        train.samples = train.samples[: args.limit_train]
    if valid is not None and args.limit_valid:
        valid.samples = valid.samples[: args.limit_valid]

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
    model = DINRAbsoluteModel(
        config,
        pedal_mode=args.pedal_mode,
        numerical_coordinates=args.numerical_coordinates,
        coordinate_frequencies=args.coordinate_frequencies,
        decoder_mask_probability=args.decoder_mask_probability,
        decoder_mask_mode=args.decoder_mask_mode,
        separate_ioi_duration_tables=args.separate_ioi_duration_token_tables,
        timing_target_mode=args.timing_target_mode,
        time_input_mode=args.time_input_mode,
    )
    if args.pretrained_pt:
        print("PT initialization:", model.initialize_from_pt(args.pretrained_pt))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="epoch" if valid is not None else "no",
        save_strategy="no" if args.no_checkpointing else "epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        logging_strategy="steps",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to="none",
        prediction_loss_only=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=DINRCollator(),
    )
    trainer.train()
    if not args.no_checkpointing and not args.skip_final_model_save:
        trainer.save_model(str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    config.save_pretrained(args.output_dir)
    (args.output_dir / "dinr_config.json").write_text(
        json.dumps({
            "model_type": "dinr_absolute",
            "timing_target_mode": args.timing_target_mode,
            "time_input_mode": args.time_input_mode,
            "pedal_mode": args.pedal_mode,
            "slot_conditioned_value_embeddings": True,
            "separate_ioi_duration_token_tables": args.separate_ioi_duration_token_tables,
            "numerical_coordinates": args.numerical_coordinates,
            "coordinate_frequencies": args.coordinate_frequencies,
            "decoder_mask_probability": args.decoder_mask_probability,
            "decoder_mask_mode": args.decoder_mask_mode,
            "decoder_timestep_mask_vector": (
                "single learned vector added to retained pitch embedding"
                if args.decoder_mask_mode == "timestep_single" else None
            ),
            "decoder_timestep_mask_vector_init_std": (
                config.initializer_range if args.decoder_mask_mode == "timestep_single" else None
            ),
            "decoder_attribute_mask_embeddings": "three slot-specific learned vectors",
            "decoder_attribute_mask_init_std": config.initializer_range,
            "num_input_slots": model.num_input_slots,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "skip_final_model_save": args.skip_final_model_save,
            "vocab_size": 763,
            "token_ranges": {
                "special": {"pad": 0, "bos": 2},
                "pitch": [5, 132],
                "velocity": [133, 260],
                "ioi_and_duration_timing": [261, 760],
                "pedal_binary": [761, 762],
            },
            "timing_bins": 500,
            "timing_bin_ms": 10,
            "max_timing_ms": 5000,
            "log_time_transform": "ln(1+ms), scaled to [0,9], then quantized to 500 bins",
            "log_deviation_ranges": {"score_ioi_positive": [-3.0, 2.0], "score_ioi_zero": [0.0, 5.0], "duration": [-3.0, 2.0]},
            "sidecar": "score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt",
            "use_pretrained_pt": bool(args.pretrained_pt),
            "pretrained_pt": args.pretrained_pt,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
