#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash script/evaluate_dinr_absolute_sampling.sh <ce|bce|none> <gpu-index>" >&2
  exit 2
fi

mode="$1"
gpu="$2"
case "$mode" in
  ce|bce|none) ;;
  *) echo "invalid pedal mode: $mode" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.evaluate.evaluate_dinr_absolute \
  --checkpoint "models/dinr_absolute_${mode}_validlast" \
  --sidecar-root ../data/ASAP_processed \
  --output-dir "models/dinr_absolute_${mode}_validlast/test_eval_sampling_w8_s3" \
  --pedal-mode "$mode" --device cuda:0 \
  --num-workers 8 --num-samples 3 --temperature 1.0 --top-p 0.9 --seed 42
