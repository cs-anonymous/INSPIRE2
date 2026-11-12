"""Shared validation, checkpoint, and loss-recording protocol for PT and DINR."""

from __future__ import annotations

import json
import math
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
