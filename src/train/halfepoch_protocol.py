"""Shared validation, checkpoint, and loss-recording protocol for PT and DINR."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from transformers import Trainer, TrainerCallback


class HalfEpochValidationCallback(TrainerCallback):
    """Add one midpoint evaluation to Trainer's normal epoch-end evaluation."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = math.floor(float(state.epoch or 0.0) + 1e-8)
        self.midpoint_evaluated = False

    def on_step_end(self, args, state, control, **kwargs):
        progress = float(state.epoch or 0.0) - self.epoch_start
        if not self.midpoint_evaluated and 0.5 <= progress < 1.0 - 1e-8:
            self.midpoint_evaluated = True
            control.should_evaluate = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = float(state.epoch or 0.0)
        if control.should_training_stop and abs(epoch - round(epoch)) > 1e-8:
            # Trainer also calls on_epoch_end when early stopping interrupts
            # an epoch at its midpoint. Do not evaluate that same point twice.
            control.should_evaluate = False
            control.should_save = False
        return control


class BestOnlyCheckpointMixin:
    best_only_checkpoints = False

    def _determine_best_metric(self, metrics, trial):
        if self.best_only_checkpoints and not math.isfinite(float(metrics["eval_loss"])):
            raise ValueError("Validation loss is not finite")
        improved = super()._determine_best_metric(metrics, trial)
        if self.best_only_checkpoints:
            # DefaultFlowCallback requests a save at every epoch end. Override
            # that request after evaluation so only strict improvements save.
            self.control.should_save = improved
        return improved


class BestOnlyTrainer(BestOnlyCheckpointMixin, Trainer):
    best_only_checkpoints = True


class EpochDiagnosticTrainer(BestOnlyTrainer):
    """Retain every validation checkpoint and track unrounded online train loss."""

    best_only_checkpoints = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if model.training:
            loss = result[0] if return_outputs else result
            epoch = math.floor(float(self.state.epoch or 0.0) + 1e-8) + 1
            if not hasattr(self, "epoch_train_losses"):
                self.epoch_train_losses = {}
            stats = self.epoch_train_losses.setdefault(epoch, dict(weighted_loss=0.0, examples=0, batches=0))
            size = int(inputs["score_tokens"].shape[0])
            stats["weighted_loss"] += float(loss.detach().float()) * size
            stats["examples"] += size
            stats["batches"] += 1
        return result

    def _determine_best_metric(self, metrics, trial):
        if not math.isfinite(float(metrics["eval_loss"])):
            raise ValueError("Validation loss is not finite")
        improved = super()._determine_best_metric(metrics, trial)
        self.control.should_save = True
        return improved


class EpochDiagnosticCallback(TrainerCallback):
    """Publish checkpoint readiness only after Trainer has finished saving it."""

    trainer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = float(state.epoch)
        stats = self.trainer.epoch_train_losses[math.ceil(epoch - 1e-8)]
        self.latest = dict(epoch=epoch, step=state.global_step,
                           train_loss_epoch_mean=stats["weighted_loss"] / stats["examples"],
                           train_examples=stats["examples"], train_batches=stats["batches"],
                           train_loss_scope="current epoch only; sample-weighted online masked loss",
                           valid_loss=float(metrics["eval_loss"]),
                           full_epoch=abs(epoch - round(epoch)) < 1e-8)

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        shutil.copyfile(Path(args.output_dir) / "dinr_config.json", checkpoint / "dinr_config.json")
        record = {**self.latest, "checkpoint": str(checkpoint.resolve())}
        ready = checkpoint / "epoch_diagnostic.json"
        temporary = ready.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(ready)


class LiveLossHistoryCallback(TrainerCallback):
    """Persist every logged interval, including history after the best epoch."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not args.should_save or not logs:
            return
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        row = {**logs, "step": state.global_step, "epoch": float(state.epoch or 0.0)}
        with (output / "loss_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        if args.should_save:
            output = Path(args.output_dir)
            state.save_to_json(str(output / "latest_trainer_state.json"))
            (output / "trainer_log_history.json").write_text(
                json.dumps(state.log_history, indent=2) + "\n", encoding="utf-8"
            )
