#!/usr/bin/env bash
# Benchmark de la zoo de modeles : params, MAdds (train + reparam), latence GPU/CPU, acc si ckpt.
# Usage :
#   phase0/run_bench.sh                                   # bench sans checkpoint
#   CKPT_DIR=outputs DATA_SET=IMNETTE DATA_PATH=./data/imagenette phase0/run_bench.sh
#   (cherche alors outputs/<model>/checkpoint-best.pth pour l'accuracy)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_FILE=${MODELS_FILE:-phase0/models_zoo.txt}
CKPT_DIR=${CKPT_DIR:-}
CSV=${CSV:-phase0/results/bench_$(date +%Y%m%d_%H%M%S).csv}
DEVICE=${DEVICE:-cuda}
INPUT_SIZE=${INPUT_SIZE:-256}
DATA_SET=${DATA_SET:-IMNETTE}
DATA_PATH=${DATA_PATH:-./data/imagenette}

mkdir -p "$(dirname "$CSV")"
echo "model,input_size,ckpt" > /dev/null

while read -r model; do
  [[ -z "$model" || "$model" == \#* ]] && continue
  ckpt=""
  if [[ -n "$CKPT_DIR" && -f "$CKPT_DIR/$model/checkpoint-best.pth" ]]; then
    ckpt="$CKPT_DIR/$model/checkpoint-best.pth"
  fi
  echo "== $model (ckpt=${ckpt:-aucun}) =="
  if [[ -n "$ckpt" ]]; then
    python phase0/bench_model.py --model "$model" --input_size "$INPUT_SIZE" --device "$DEVICE" \
      --data_set "$DATA_SET" --data_path "$DATA_PATH" --ckpt "$ckpt" --csv "$CSV"
  else
    python phase0/bench_model.py --model "$model" --input_size "$INPUT_SIZE" --device "$DEVICE" \
      --csv "$CSV"
  fi
done < "$MODELS_FILE"

echo "Resultats ecrits dans : $CSV"
