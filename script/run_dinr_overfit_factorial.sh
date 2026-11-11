#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash script/run_dinr_overfit_factorial.sh <decoder_mask|outer_optimizer|random_init> <gpu-index>" >&2
  exit 2
fi

variant="$1"
gpu="$2"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
output_dir="models/overfit_factorial_20261030/${variant}"

args=(
  --precomputed-train "${cache_dir}/dinr_absolute_train_train_b512_o50.pt"
  --precomputed-valid "${cache_dir}/dinr_absolute_train_valid_b512_o50.pt"
  --output-dir "${output_dir}"
  --block-notes 512 --overlap-ratio 0.5
  --epochs 6 --batch-size 32
  --eval-split valid --pedal-mode ce --seed 42
  --learning-rate 0.0003
)

case "${variant}" in
  decoder_mask)
    args+=(--decoder-mask-probability 0.5)
    args+=(--pretrained-pt ../models/official_redownload_20260711/pianist-transformer-rendering)
    ;;
  outer_optimizer)
    args+=(--gradient-accumulation-steps 2 --lr-scheduler-type cosine --warmup-steps 50)
    args+=(--pretrained-pt ../models/official_redownload_20260711/pianist-transformer-rendering)
    ;;
  random_init)
    ;;
  *)
    echo "invalid variant: ${variant}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}"
log_path="${output_dir}/train.log"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
  python -u -m src.train.dinr_absolute "${args[@]}" 2>&1 | tee "${log_path}"
