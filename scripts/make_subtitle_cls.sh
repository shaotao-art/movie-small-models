#!/usr/bin/env bash
# 用已有的 all_images.jsonl 标注结果，从 data/frames/ 生成二分类数据集 data/subtitle_cls
# 不重新请求 VLM；缺少标注的图片会被跳过。
#
# 用法:
#   bash scripts/make_subtitle_cls.sh
#   RESULTS=other.jsonl bash scripts/make_subtitle_cls.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="${RESULTS:-data/all_images.jsonl}"

python data_making/build_cls_dataset.py "$RESULTS" -o data/subtitle_cls --overwrite

echo "done: data/subtitle_cls/{train,val}/{has_subtitle,no_subtitle}"
