#!/usr/bin/env bash
# Evaluation d'un checkpoint en mode REPARAMETRISE (cout reel de deploiement) :
# affiche [REAL] params / MAdds et l'accuracy top-1 sur le jeu de validation.
# Exemple :
#   MODEL=MobileViTv4_XXS OUT=outputs/imagenette_mobilevitv4_xxs_s0 DATA_SET=IMNETTE DATA_PATH=./data/imagenette phase0/eval.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=${MODEL:-MobileViTv4_XXS}
OUT=${OUT:?"definir OUT=outputs/<run>"}
CKPT=${CKPT:-$OUT/checkpoint-best.pth}
DATA_SET=${DATA_SET:-IMNETTE}
DATA_PATH=${DATA_PATH:-./data/imagenette}
INPUT_SIZE=${INPUT_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}

[[ -f "$CKPT" ]] || { echo "Checkpoint introuvable: $CKPT"; exit 1; }

python main.py \
  --model "$MODEL" \
  --data_set "$DATA_SET" --data_path "$DATA_PATH" \
  --input_size "$INPUT_SIZE" \
  --eval true --resume "$CKPT" \
  --dist_eval false --use_amp false \
  --num_workers "$NUM_WORKERS" --batch_size 128 \
  --device cuda 2>&1 | tee "$OUT/eval_reparam.log"
