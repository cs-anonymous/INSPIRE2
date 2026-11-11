#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
variant="${2:-separate_abs500}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
table_args=()
case "$variant" in
  separate_abs500)
    run_name="separate_retain_pitch_single_vector_abs500"
    table_args+=(--separate-ioi-duration-token-tables)
    input_mode=linear; target_mode=absolute ;;
  shared_abs500)
    run_name="shared_retain_pitch_single_vector_abs500"
    input_mode=linear; target_mode=absolute ;;
  separate_log_abs)
    run_name="separate_retain_pitch_single_vector_log_abs"
    table_args+=(--separate-ioi-duration-token-tables)
    input_mode=log_absolute; target_mode=log_absolute ;;
  separate_log_dev)
    run_name="separate_retain_pitch_single_vector_log_dev"
    table_args+=(--separate-ioi-duration-token-tables)
    input_mode=log_absolute; target_mode=score_relative_log ;;
  *)
    echo "Unknown variant: $variant (expected separate_abs500, shared_abs500, separate_log_abs, separate_log_dev)" >&2
    exit 2 ;;
esac
root="models/dinr_mask_revised_heldout12_8ep/$run_name"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
if [[ -e "$root/train.log" ]]; then
  echo "Training already exists at $root; refusing to overwrite it." >&2
  exit 1
fi
mkdir -p "$root"
trap 'result=$?; if [[ $result -ne 0 ]]; then echo "failed (exit $result)" > "$root/status.txt"; fi' EXIT
echo "training on GPU $gpu" > "$root/status.txt"

CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
  --sidecar-root ../data/ASAP_processed \
  --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
  --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
  --output-dir "$root" \
  --block-notes 512 --overlap-ratio 0.5 \
  --epochs 8 --batch-size 32 --learning-rate 3e-4 --seed 42 \
  --eval-split valid --pretrained-pt "$pretrained_pt" --pedal-mode none \
  --decoder-mask-mode timestep_single --decoder-mask-probability 0.5 \
  "${table_args[@]}" \
  --time-input-mode "$input_mode" --timing-target-mode "$target_mode" \
  --save-total-limit 2 --load-best-model-at-end --skip-final-model-save \
  > "$root/train.log" 2>&1

best_checkpoint=$(python - "$root" <<'PY'
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

echo "test inference on GPU $gpu: $best_checkpoint" > "$root/status.txt"
CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
  --checkpoint "$best_checkpoint" --sidecar-root ../data/ASAP_processed \
  --output-dir "$root/test_eval_sampling_w8_s2" \
  --pedal-mode none --device cuda:0 --num-workers 8 --num-samples 2 \
  --time-input-mode "$input_mode" --timing-target-mode "$target_mode" \
  --temperature 1.0 --top-p 0.9 --seed 42 \
  > "$root/test_inference.log" 2>&1

echo "completed" > "$root/status.txt"
echo "Completed timestep 1+pitch experiment on GPU $gpu: $root"
