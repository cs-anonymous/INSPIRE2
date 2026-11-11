#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || "$1" == "$2" || "$1" == "$3" || "$2" == "$3" ]]; then
  echo "usage: bash script/run_dinr_single_pitch_missing_3gpu.sh <gpu-shared-abs> <gpu-log-abs> <gpu-log-dev> (three distinct GPUs)" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

variants=(shared_abs500 separate_log_abs separate_log_dev)
run_names=(shared_retain_pitch_single_vector_abs500 separate_retain_pitch_single_vector_log_abs separate_retain_pitch_single_vector_log_dev)
gpus=("$1" "$2" "$3")
root="models/dinr_mask_revised_heldout12_8ep"
for run_name in "${run_names[@]}"; do
  if [[ -e "$root/$run_name/train.log" ]]; then
    echo "Existing training found at $root/$run_name; refusing to overwrite it." >&2
    exit 1
  fi
done

pids=()
for index in 0 1 2; do
  bash script/run_dinr_retain_pitch_single_mask_heldout12_8ep.sh "${gpus[$index]}" "${variants[$index]}" &
  pids+=("$!")
  echo "Started ${variants[$index]} on GPU ${gpus[$index]}: pipeline PID ${pids[$index]}"
done

status=0
for index in 0 1 2; do
  if wait "${pids[$index]}"; then
    echo "Completed ${variants[$index]}"
  else
    echo "FAILED ${variants[$index]}; inspect $root/${run_names[$index]}/" >&2
    status=1
  fi
  python script/update_revised_dinr_table.py || status=1
done
exit "$status"
