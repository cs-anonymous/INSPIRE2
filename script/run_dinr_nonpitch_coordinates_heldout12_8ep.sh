#!/usr/bin/env bash
set -euo pipefail

# Match the 8-epoch held-out-12 mask ablation recipe; add Fourier coordinates
# to every numerical performance attribute (IOI, velocity, duration), while
# pitch remains categorical and receives no coordinate embedding.
gpu="${1:-2}"
output_dir="models/dinr_stage1_nonpitch_coordinates_heldout12_8ep"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
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
  --numerical-coordinates --coordinate-frequencies 16 \
  --decoder-mask-mode attribute --decoder-mask-probability 0.0 \
  --save-total-limit 2 \
  --load-best-model-at-end \
  --skip-final-model-save > "$output_dir/train.log" 2>&1
