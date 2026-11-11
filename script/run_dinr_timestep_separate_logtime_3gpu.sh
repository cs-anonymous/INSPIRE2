#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash script/run_dinr_timestep_separate_logtime_3gpu.sh <gpu-absolute-linear> <gpu-log-absolute> <gpu-log-deviation>" >&2
  exit 2
fi

cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
root="models/dinr_timestep_separate_logtime_heldout12_8ep"

run_one() {
  local gpu="$1" name="$2" input_mode="$3" target_mode="$4"
  local output_dir="$root/$name"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
    --sidecar-root ../data/ASAP_processed \
    --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
    --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
    --output-dir "$output_dir" \
    --block-notes 512 --overlap-ratio 0.5 \
    --epochs 8 --batch-size 32 --learning-rate 3e-4 --seed 42 \
    --eval-split valid --pretrained-pt "$pretrained_pt" --pedal-mode none \
    --decoder-mask-mode timestep --decoder-mask-probability 0.5 \
    --separate-ioi-duration-token-tables \
    --time-input-mode "$input_mode" --timing-target-mode "$target_mode" \
    --save-total-limit 2 --load-best-model-at-end --skip-final-model-save \
    > "$output_dir/train.log" 2>&1
}

run_one "$1" absolute_linear linear absolute & p1=$!
run_one "$2" log_absolute log_absolute log_absolute & p2=$!
run_one "$3" log_deviation log_absolute score_relative_log & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one training run failed; inspect $root/*/train.log" >&2
  exit "$status"
fi

run_inference() {
  local name="$1" gpu="$2"
  output_dir="$root/$name"
  best_checkpoint=$(python - "$output_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
states = sorted(root.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.stat().st_mtime)
if not states:
    raise SystemExit("no checkpoint trainer_state.json found")
state = json.loads(states[-1].read_text())
best = state.get("best_model_checkpoint")
if not best:
    raise SystemExit("trainer state has no best_model_checkpoint")
print(Path(best).resolve())
PY
)
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
    --checkpoint "$best_checkpoint" --sidecar-root ../data/ASAP_processed \
    --output-dir "$output_dir/test_eval_sampling_w8_s2" \
    --pedal-mode none --device cuda:0 --num-workers 8 --num-samples 2 \
    --temperature 1.0 --top-p 0.9 --seed 42 \
    > "$output_dir/test_inference.log" 2>&1
}

run_inference absolute_linear "$1" & pi1=$!
run_inference log_absolute "$2" & pi2=$!
run_inference log_deviation "$3" & pi3=$!
status=0
wait "$pi1" || status=1
wait "$pi2" || status=1
wait "$pi3" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one test inference run failed; inspect $root/*/test_inference.log" >&2
  exit "$status"
fi

echo "All three 8-epoch trainings and 19-work x 2-sample inference runs completed under $root"
