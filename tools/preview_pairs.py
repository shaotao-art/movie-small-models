#!/usr/bin/env python
"""把 pair 数据集的采样结果渲染成 HTML，人工检查正/难负/易负样本是否构造正确。

用法:
    python -m tools.preview_pairs --cues data/pairs/cues -o data/pairs/preview.html
    python -m tools.preview_pairs --cues data/pairs/cues --per-type 12 --seed 7
"""

import argparse
import base64
import html
import io
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from classifier.pair_data import PairDataset, load_cues
from config import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger("preview_pairs")
TYPES = {1: ("正样本", "#3f9e63"), 2: ("难负样本", "#d4a72c"), 3: ("易负样本", "#b5483a")}
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def tensor_to_uri(t):
    """归一化后的 [3,H,W] -> 可直接嵌入 HTML 的 JPEG data URI。"""
    img = ((t.cpu() * STD + MEAN).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_pairs(ds, kind, per_type):
    """按类型取若干条 (i, j) 下标；kind: 1 正 / 2 难负 / 3 易负。"""
    rng = random.Random(0)
    picked = []
    for i in range(len(ds.cues)):
        if len(picked) >= per_type:
            break
        if kind == 2:
            j = ds._hard_negative(i)
        elif kind == 3:
            j = ds._easy_negative(i, rng)
        else:
            j = i
        if j is None:
            continue
        picked.append((i, j))
    return picked


def render(ds, per_type):
    sections = []
    for kind, (title, color) in TYPES.items():
        cards = []
        for i, j in build_pairs(ds, kind, per_type):
            ci, cj = ds.cues[i], ds.cues[j]
            if i == j:
                fa = ci["frames"][0]
                fb = ci["frames"][-1]
            else:
                fa, fb = ci["frames"][0], cj["frames"][0]
            ia, ib = ds._load(ci["root"], fa), ds._load(cj["root"], fb)
            cards.append(f"""
      <div class="pair">
        <div class="imgs">
          <img src="{tensor_to_uri(ia)}">
          <img src="{tensor_to_uri(ib)}">
        </div>
        <div class="meta">
          <div class="row"><span class="tag" style="background:{color}">{title}</span>
            <span class="cue">cue{i:02d} · {html.escape(fa.split('/')[-1])}</span></div>
          <div class="text">{html.escape(ci['text'])}</div>
          <div class="cue">cue{j:02d} · {html.escape(fb.split('/')[-1])}</div>
          <div class="text">{html.escape(cj['text'])}</div>
        </div>
      </div>""")
        sections.append(f"<h2><span class=\"dot\" style=\"background:{color}\"></span>{title}"
                        f"<span class=\"count\">{len(cards)} 对</span></h2>{''.join(cards)}")

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>pair 样本预览</title>
<style>
  body {{ margin: 0 auto; padding: 24px; max-width: 980px; background: #fcfcfb; color: #1a1a1a;
         font-family: system-ui, -apple-system, "PingFang SC", sans-serif; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  .summary {{ color: #52514e; font-size: 13px; margin-bottom: 18px; }}
  h2 {{ font-size: 15px; display: flex; align-items: center; gap: 8px; margin: 28px 0 10px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .count {{ color: #898781; font-weight: 400; font-size: 12px; }}
  .pair {{ border: 1px solid #e1e0d9; border-radius: 10px; background: #fff; margin-bottom: 14px;
           overflow: hidden; }}
  .imgs {{ display: grid; grid-template-columns: 1fr; gap: 2px; background: #000; }}
  .imgs img {{ width: 100%; display: block; }}
  .meta {{ padding: 10px 12px; font-size: 12px; color: #52514e; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .tag {{ color: #fff; border-radius: 999px; padding: 1px 8px; font-size: 11px; }}
  .cue {{ color: #898781; font-variant-numeric: tabular-nums; }}
  .text {{ color: #1a1a1a; font-size: 13px; margin-bottom: 4px; }}
</style></head>
<body>
<h1>pair 样本预览</h1>
<div class="summary">
  共 {len(ds.cues)} 条 cue · 图像为模型实际输入（裁剪 {ds.crop}，{ds.size[0]}×{ds.size[1]}）·
  上图为第一帧、下图为第二帧
</div>
{''.join(sections)}
</body></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cues", required=True, help="cues 目录或 jsonl")
    parser.add_argument("-o", "--output", default=None, help="默认 <cues 上级目录>/preview.html")
    parser.add_argument("--per-type", type=int, default=8, help="每类展示几对（默认 8）")
    parser.add_argument("--size", default="448x128", help="模型输入尺寸 WxH（默认 448x128）")
    parser.add_argument("--crop", nargs=4, type=float, default=[0.125, 0.8, 0.875, 1.0],
                        metavar=("X0", "Y0", "X1", "Y1"), help="按比例裁剪字幕区（默认底部 1/5、左右各 1/8）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    w, h = (int(v) for v in args.size.lower().split("x"))
    ds = PairDataset(load_cues(args.cues), size=(w, h), crop=args.crop, train=False)

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.cues)), "preview.html")
    with open(out, "w") as f:
        f.write(render(ds, args.per_type))
    logger.info("已写出 %s（%.1f MB）", out, os.path.getsize(out) / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
