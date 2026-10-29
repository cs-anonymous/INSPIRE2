#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash script/run_dinr_absolute_asap.sh <ce|bce|none> <gpu-index>" >&2
  exit 2
fi

mode="$1"
gpu="$2"
case "$mode" in
  ce|bce|none) ;;
  *) echo "invalid pedal mode: $mode" >&2; exit 2 ;;
esac

output_dir="models/dinr_absolute_${mode}_validlast"
CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
  --sidecar-root ../data/ASAP_processed \
  --precomputed-train ../data/ASAP_processed_dinr_tokens/dinr_absolute_train_train_b512_o50.pt \
  --precomputed-valid ../data/ASAP_processed_dinr_tokens/dinr_absolute_train_valid_b512_o50.pt \
  --output-dir "$output_dir" \
  --block-notes 512 --overlap-ratio 0.5 \
  --epochs 16 --batch-size 32 \
  --eval-split valid \
  --pretrained-pt ../models/official_redownload_20260711/pianist-transformer-rendering \
  --pedal-mode "$mode"

CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
  --checkpoint "$output_dir" \
  --sidecar-root ../data/ASAP_processed \
  --output-dir "$output_dir/test_eval" \
  --pedal-mode "$mode" --device cuda:0
