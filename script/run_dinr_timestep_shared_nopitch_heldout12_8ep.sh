#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
root="models/dinr_timestep_shared_nopitch_heldout12_8ep"
cache_dir="../data/ASAP_processed_dinr_tokens_heldout12_v1"
pretrained_pt="../models/official_redownload_20260711/pianist-transformer-rendering"
mkdir -p "$root"

CUDA_VISIBLE_DEVICES="$gpu" python -u -m src.train.dinr_absolute \
  --sidecar-root ../data/ASAP_processed \
  --precomputed-train "$cache_dir/dinr_absolute_train_train_b512_o50.pt" \
  --precomputed-valid "$cache_dir/dinr_absolute_train_valid_b512_o50.pt" \
  --output-dir "$root" \
  --block-notes 512 --overlap-ratio 0.5 \
  --epochs 8 --batch-size 32 --learning-rate 3e-4 --seed 42 \
  --eval-split valid --pretrained-pt "$pretrained_pt" --pedal-mode none \
  --decoder-mask-mode timestep --decoder-mask-probability 0.5 \
  --time-input-mode linear --timing-target-mode absolute \
  --save-total-limit 2 --load-best-model-at-end --skip-final-model-save \
  > "$root/train.log" 2>&1

best_checkpoint=$(python - "$root" <<'PY'
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
  --output-dir "$root/test_eval_sampling_w8_s2" \
  --pedal-mode none --device cuda:0 --num-workers 8 --num-samples 2 \
  --temperature 1.0 --top-p 0.9 --seed 42 \
  > "$root/test_inference.log" 2>&1

echo "Completed shared-table pitch-preserving timestep experiment: $root"
