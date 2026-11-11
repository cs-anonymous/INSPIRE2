#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash script/run_dinr_absolute_asap_heldout12.sh <ce|bce|none> <gpu-index>" >&2
  exit 2
fi

mode="$1"
gpu="$2"
case "$mode" in
  ce|bce|none) ;;
  *) echo "invalid pedal mode: $mode" >&2; exit 2 ;;
esac

output_dir="models/dinr_absolute_${mode}_heldout12"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
  --sidecar-root ../data/ASAP_processed \
  --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
  --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
  --output-dir "$output_dir" \
  --block-notes 512 --overlap-ratio 0.5 \
  --epochs 16 --batch-size 32 \
  --eval-split valid \
  --pretrained-pt ../models/official_redownload_20260711/pianist-transformer-rendering \
  --pedal-mode "$mode"

CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
  --checkpoint "$output_dir" \
  --sidecar-root ../data/ASAP_processed \
  --output-dir "$output_dir/test_eval_sampling_w8_s3" \
  --pedal-mode "$mode" --device cuda:0 \
  --num-workers 8 --num-samples 3 --temperature 1.0 --top-p 0.9 --seed 42
