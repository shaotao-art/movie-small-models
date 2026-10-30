#!/usr/bin/env bash
# 跑 pair（两帧台词是否相同）任务的全部实验：模型 × 翻转模式。
# 每次训练用一份临时 config（写在 outputs/pair_configs/ 下），不改动原配置。
#
# 用法:
#   bash scripts/run_pair_experiments.sh
#   MODELS="mobilenet_v3_small" FLIPS="h" bash scripts/run_pair_experiments.sh
#   EPOCHS=30 bash scripts/run_pair_experiments.sh
#   TRAIN_CMD="torchrun --nproc_per_node=4 -m classifier.train" bash scripts/run_pair_experiments.sh
#   FORCE=1 bash scripts/run_pair_experiments.sh     # 已有 best.pth 也重跑
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="classifier/configs/pair_mobilenet_v3_small.yaml"
MODELS="${MODELS:-mobilenet_v3_small mobilenet_v3_large efficientnet_b0 resnet18}"
FLIPS="${FLIPS:-none h hv}"
TRAIN_CMD="${TRAIN_CMD:-python -m classifier.train}"
EPOCHS="${EPOCHS:-}"
CONF_DIR="outputs/pair_configs"
mkdir -p "$CONF_DIR"

total=0 skipped=0 failed=0
for model in $MODELS; do
  for flip in $FLIPS; do
    total=$((total + 1))
    exp="pair_${model}_flip-${flip}"
    cfg="$CONF_DIR/${exp}.yaml"

    if [ -f "outputs/${exp}/best.pth" ] && [ -z "${FORCE:-}" ]; then
      echo "---- 跳过（已有结果，FORCE=1 重跑）: $exp"
      skipped=$((skipped + 1))
      continue
    fi

    python - "$BASE" "$cfg" "$model" "$flip" "$exp" "$EPOCHS" <<'PY'
import sys, yaml
base, out, model, flip, exp, epochs = sys.argv[1:7]
cfg = yaml.safe_load(open(base))
cfg["model"]["name"] = model
cfg["data"]["flip"] = flip
cfg["swanlab"]["experiment_name"] = exp
if epochs:
    cfg["train"]["epochs"] = int(epochs)
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

    echo "========== [$total] $exp =========="
    if $TRAIN_CMD --config "$cfg"; then
      echo "---- 完成: $exp"
    else
      echo "---- 失败: $exp" >&2
      failed=$((failed + 1))
    fi
  done
done

echo "全部结束：共 ${total} 组，跳过 ${skipped} 组，失败 ${failed} 组。日志与权重在 outputs/<实验名>/"
[ "$failed" -eq 0 ]
