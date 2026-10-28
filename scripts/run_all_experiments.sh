#!/usr/bin/env bash
# 依次跑 configs/ 下所有模型实验（template.yaml 除外）
#
# 用法:
#   bash scripts/run_all_experiments.sh
#   TRAIN_CMD="torchrun --nproc_per_node=4 -m classifier.train" bash scripts/run_all_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_CMD="${TRAIN_CMD:-python -m classifier.train}"

for cfg in \
  configs/mobilenet_v3_large.yaml \
  configs/efficientnet_b0.yaml \
  configs/regnet_x_400mf.yaml \
  configs/mnasnet0_75.yaml \
  configs/resnet18.yaml
do
  echo "========== $cfg =========="
  $TRAIN_CMD --config "$cfg"
done

echo "all experiments done, see outputs/"
