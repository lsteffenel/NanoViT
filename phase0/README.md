# Phase 0 — Harnais de benchmark NanoViT sur DGX Spark

Objectif : disposer d'un banc de mesure reproductible (accuracy, paramètres, MAdds, latence GPU/CPU)
avant de tester les stratégies d'allègement (A : convolutions atrous, B : Model Carving).

## 0. Environnement (DGX Spark, 1 GPU GB10)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # build CUDA
pip install -r phase0/requirements.txt
nvidia-smi        # verifier le GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Données

- **Imagenette** (recommandé pour itérer vite, 10 classes) : torchvision télécharge tout seul
  (`--data_set IMNETTE --data_path ./data/imagenette`).
- **CIFAR-100** : `--data_set CIFAR --data_path ./data/cifar` (téléchargement auto).
  Attention : le modèle exige `--input_size 256` (le pool final est `AvgPool2d(input_size//32)`,
  seule la résolution 256 tombe juste sur une carte 8x8 -> 1x1). Les images 32x32 sont upsamplees
  par le transform.
- **ImageNet-1k** : `--data_set IMNET --data_path /chemin/imagenet` avec `train/` et `val/`.

## 2. Entraîner les baselines (3 seeds chacune)

DGX Spark = 1 GPU : laisser `NPROC=1` (lancement `python main.py` direct).

```bash
for SEED in 0 1 2; do
  MODEL=MobileViTv4_XXS        DATA_SET=IMNETTE DATA_PATH=./data/imagenette SEED=$SEED phase0/train.sh
  MODEL=MobileViTv4_XS         DATA_SET=IMNETTE DATA_PATH=./data/imagenette SEED=$SEED phase0/train.sh
  MODEL=MobileViTv4_linear_XXS DATA_SET=IMNETTE DATA_PATH=./data/imagenette SEED=$SEED phase0/train.sh
  MODEL=MobileViTv4_flat_XXS   DATA_SET=IMNETTE DATA_PATH=./data/imagenette SEED=$SEED phase0/train.sh
done
```

Chaque run produit `outputs/<data>_<model>_s<seed>/` avec `checkpoint-best.pth`,
`checkpoint-<epoch>.pth` et `log.txt` (une ligne JSON par époque, clés `test_acc1`, `test_acc5`).
Recette : AdamW lr 6e-3 (batch effectif 256x2), cosine + 20 warmup, mixup 0.8 / cutmix 1.0,
label smoothing 0.1, RandAugment, random erasing 0.25, drop_path 0.1, AMP. Tout est surchargeable
par variables d'environnement (voir l'en-tête de `phase0/train.sh`).

## 3. Évaluer un run en mode reparamétrisé (coût réel de déploiement)

```bash
MODEL=MobileViTv4_XXS OUT=outputs/imagenette_mobilevitv4_xxs_s0 \
  DATA_SET=IMNETTE DATA_PATH=./data/imagenette phase0/eval.sh
```

`main.py --eval` reparametrise automatiquement les blocs `conv_rep` (fusion RepVGG/MobileOne)
puis affiche `[REAL] number of params after reparameterization`, `[REAL] MAdds after
reparameterization` et l'accuracy top-1. Log : `$OUT/eval_reparam.log`.

## 4. Benchmark params / MAdds / latence (GPU + CPU)

```bash
# Sans checkpoint (poids, FLOPs, latence brute)
phase0/run_bench.sh                                   # -> phase0/results/bench_<date>.csv

# Avec accuracy des runs entraînés (cherche outputs/<model>/checkpoint-best.pth)
CKPT_DIR=outputs DATA_SET=IMNETTE DATA_PATH=./data/imagenette phase0/run_bench.sh
```

La latence est mesurée sur le modèle **reparamétrisé** : bs=1 et bs=256 sur GPU, bs=1 sur CPU
(warmup + itérations moyennées, `torch.cuda.synchronize`). MAdds via fvcore, en mode train et
reparamétrisé.

## 5. Agréger les résultats

```bash
python phase0/aggregate.py --outputs_dir outputs --out phase0/results/summary.md
```

Produit un tableau markdown trié par MAdds reparamétrisés : run, modèle, acc1/acc5 (meilleure époque
du `log.txt`), params, MAdds, latence GPU/CPU.

## 6. Arborescence produite

```
outputs/<data>_<model>_s<seed>/
  checkpoint-best.pth  checkpoint-<epoch>.pth  log.txt  train.log  eval_reparam.log
phase0/results/
  bench_<date>.csv     summary.md
```

## 7. Passage aux phases suivantes

- **Phase A (atrous)** : variantes de `models/nanovit.py` (dilation dans les dw 7x7, remplacement
  des blocs MVIT par convs dilatées) -> mêmes scripts, modèles ajoutés à `phase0/models_zoo.txt`.
- **Phase B (Model Carving)** : NanoViT « gated » (portes + pression) -> entraînement via
  `phase0/train.sh`, comparaison à iso-budget contre (i) dense, (ii) shallower from scratch,
  (iii) pruning post-hoc + finetune, (iv) distillation.
