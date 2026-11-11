#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash script/run_dinr_stage1_mask_ablation_3gpu.sh <gpu-attr> <gpu-timestep> <gpu-none>" >&2
  exit 2
fi

gpu_attr="$1"
gpu_timestep="$2"
gpu_none="$3"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
root="models/dinr_stage1_mask_ablation_heldout12"

run_one() {
  local gpu="$1"
  local name="$2"
  local mask_mode="$3"
  local mask_probability="$4"
  local output_dir="$root/$name"
  mkdir -p "$output_dir"

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
    --sidecar-root ../data/ASAP_processed \
    --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
    --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
    --output-dir "$output_dir" \
    --block-notes 512 --overlap-ratio 0.5 \
    --epochs 8 --batch-size 32 --learning-rate 3e-4 --seed 42 \
    --eval-split valid \
    --pretrained-pt "$pretrained_pt" \
    --pedal-mode none \
    --decoder-mask-mode "$mask_mode" \
    --decoder-mask-probability "$mask_probability" \
    --save-total-limit 2 \
    --load-best-model-at-end \
    --skip-final-model-save > "$output_dir/train.log" 2>&1
}

run_one "$gpu_attr" attribute_random attribute 0.5 &
pid_attr=$!
run_one "$gpu_timestep" timestep_random note 0.5 &
pid_timestep=$!
run_one "$gpu_none" no_mask attribute 0.0 &
pid_none=$!

status=0
wait "$pid_attr" || status=1
wait "$pid_timestep" || status=1
wait "$pid_none" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one DINR mask ablation failed; inspect $root/*/train.log" >&2
  exit "$status"
fi

echo "All runs completed. Per-epoch logs: $root/*/trainer_log_history.json"
