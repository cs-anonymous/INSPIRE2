#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash script/run_dinr_mask_revised_6.sh <gpu-attribute> <gpu-absolute> <gpu-log>" >&2
  exit 2
fi

gpu_attribute="$1"
gpu_absolute="$2"
gpu_log="$3"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
root="models/dinr_mask_revised_heldout12_8ep"

run_one() {
  local gpu="$1" name="$2" table="$3" mask_mode="$4" input_mode="$5" target_mode="$6"
  local output_dir="$root/$name"
  local table_args=()
  mkdir -p "$output_dir"
  if [[ "$table" == "separate" ]]; then
    table_args+=(--separate-ioi-duration-token-tables)
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
    --sidecar-root ../data/ASAP_processed \
    --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
    --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
    --output-dir "$output_dir" \
    --block-notes 512 --overlap-ratio 0.5 \
    --epochs 8 --batch-size 32 --learning-rate 3e-4 --seed 42 \
    --eval-split valid --pretrained-pt "$pretrained_pt" --pedal-mode none \
    --decoder-mask-mode "$mask_mode" --decoder-mask-probability 0.5 \
    "${table_args[@]}" \
    --time-input-mode "$input_mode" --timing-target-mode "$target_mode" \
    --save-total-limit 2 --load-best-model-at-end --skip-final-model-save \
    > "$output_dir/train.log" 2>&1

  local best_checkpoint
  best_checkpoint=$(python - "$output_dir" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
states = sorted(root.glob("checkpoint-*/trainer_state.json"), key=lambda path: path.stat().st_mtime)
if not states:
    raise SystemExit(f"no checkpoint trainer_state.json found under {root}")
state = json.loads(states[-1].read_text())
best = state.get("best_model_checkpoint")
if not best:
    raise SystemExit(f"trainer state has no best_model_checkpoint under {root}")
print(Path(best).resolve())
PY
)

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
    --checkpoint "$best_checkpoint" --sidecar-root ../data/ASAP_processed \
    --output-dir "$output_dir/test_eval_sampling_w8_s2" \
    --pedal-mode none --device cuda:0 --num-workers 8 --num-samples 2 \
    --temperature 1.0 --top-p 0.9 --seed 42 \
    > "$output_dir/test_inference.log" 2>&1
  echo "Completed $name on GPU $gpu"
}

run_attribute_card() {
  local gpu="$1"
  run_one "$gpu" separate_attribute_abs500 separate attribute linear absolute
  run_one "$gpu" shared_attribute_abs500 shared attribute linear absolute
}

run_absolute_card() {
  local gpu="$1"
  run_one "$gpu" separate_retain_pitch_abs500 separate timestep linear absolute
  run_one "$gpu" shared_retain_pitch_abs500 shared timestep linear absolute
}

run_log_card() {
  local gpu="$1"
  run_one "$gpu" separate_retain_pitch_log_abs separate timestep log_absolute log_absolute
  run_one "$gpu" separate_retain_pitch_log_dev separate timestep log_absolute score_relative_log
}

run_attribute_card "$gpu_attribute" & pid_attribute=$!
run_absolute_card "$gpu_absolute" & pid_absolute=$!
run_log_card "$gpu_log" & pid_log=$!

status=0
wait "$pid_attribute" || status=1
wait "$pid_absolute" || status=1
wait "$pid_log" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one revised DINR experiment failed; inspect $root/*/{train.log,test_inference.log}" >&2
  exit "$status"
fi

echo "All six revised training and 19-work x 2-sample inference runs completed under $root"
