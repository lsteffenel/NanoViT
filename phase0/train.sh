#!/usr/bin/env bash
# Entraînement d'un baseline NanoViT (recette timm du depot).
# DGX Spark = 1 GPU (GB10) -> NPROC=1 lance en "python" simple ; NPROC>1 -> torchrun.
# Exemples :
#   MODEL=MobileViTv4_XXS DATA_SET=IMNETTE DATA_PATH=./data/imagenette SEED=0 phase0/train.sh
#   MODEL=MobileViTv4_XS  DATA_SET=CIFAR  DATA_PATH=./data/cifar SEED=1 OUT=outputs/cifar_xs_s1 phase0/train.sh
# Note : --input_size 256 est requis (le pool final est AvgPool2d(input_size//32)).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=${MODEL:-MobileViTv4_XXS}
DATA_SET=${DATA_SET:-IMNETTE}
DATA_PATH=${DATA_PATH:-./data/imagenette}
SEED=${SEED:-0}
OUT=${OUT:-outputs/${DATA_SET,,}_${MODEL,,}_s${SEED}}

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-256}
UPDATE_FREQ=${UPDATE_FREQ:-2}
LR=${LR:-6e-3}
MIN_LR=${MIN_LR:-1e-6}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-20}
INPUT_SIZE=${INPUT_SIZE:-256}
MIXUP=${MIXUP:-0.8}
CUTMIX=${CUTMIX:-1.0}
SMOOTHING=${SMOOTHING:-0.1}
REPROB=${REPROB:-0.25}
DROP_PATH=${DROP_PATH:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
NPROC=${NPROC:-1}
NUM_WORKERS=${NUM_WORKERS:-16}
USE_AMP=${USE_AMP:-true}

mkdir -p "$OUT"

COMMON_ARGS=(
  --model "$MODEL"
  --data_set "$DATA_SET"
  --data_path "$DATA_PATH"
  --input_size "$INPUT_SIZE"
  --batch_size "$BATCH"
  --update_freq "$UPDATE_FREQ"
  --epochs "$EPOCHS"
  --lr "$LR" --min_lr "$MIN_LR" --warmup_epochs "$WARMUP_EPOCHS"
  --opt adamw --weight_decay "$WEIGHT_DECAY"
  --mixup "$MIXUP" --cutmix "$CUTMIX" --mixup_prob 1.0
  --smoothing "$SMOOTHING"
  --aa rand-m9-mstd0.5-inc1 --color_jitter 0.4 --reprob "$REPROB"
  --drop_path "$DROP_PATH"
  --output_dir "$OUT"
  --seed "$SEED"
  --device cuda
  --use_amp "$USE_AMP"
  --num_workers "$NUM_WORKERS"
  --save_ckpt true --save_ckpt_freq 10
)

if [[ "$NPROC" -gt 1 ]]; then
  echo "Lancement torchrun (nproc=$NPROC) -> $OUT"
  torchrun --nproc_per_node="$NPROC" main.py "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT/train.log"
else
  echo "Lancement single-process -> $OUT"
  python main.py "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT/train.log"
fi
