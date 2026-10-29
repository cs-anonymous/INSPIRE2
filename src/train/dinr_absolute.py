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
    args = parser.parse_args()
    set_seed(args.seed)

    train = DINRAbsoluteDataset(
        args.sidecar_root, "train", args.block_notes, args.overlap_ratio,
        max_performances=args.max_train_performances,
        include_pedal=args.pedal_mode != "none",
        precomputed_path=args.precomputed_train,
    )
    valid = None
    if args.eval_split != "none":
        valid = DINRAbsoluteDataset(
            args.sidecar_root, args.eval_split, args.block_notes, args.overlap_ratio,
            max_performances=args.max_valid_performances,
            include_pedal=args.pedal_mode != "none",
            precomputed_path=args.precomputed_valid,
        )
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
    model = DINRAbsoluteModel(config, pedal_mode=args.pedal_mode)
    if args.pretrained_pt:
        print("PT initialization:", model.initialize_from_pt(args.pretrained_pt))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="epoch" if valid is not None else "no",
        save_strategy="epoch",
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
    trainer.save_model(str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(args.output_dir)
    (args.output_dir / "dinr_config.json").write_text(
        json.dumps({
            "model_type": "dinr_absolute",
            "pedal_mode": args.pedal_mode,
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
            "sidecar": "score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt",
            "use_pretrained_pt": bool(args.pretrained_pt),
            "pretrained_pt": args.pretrained_pt,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
