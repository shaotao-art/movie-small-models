#!/usr/bin/env python
"""从训练 checkpoint 里只导出模型权重（去掉 optimizer / epoch / config）。

用法:
    python tools/export_weights.py outputs/efficientnet_b0/best.pth
    python tools/export_weights.py outputs/efficientnet_b0/best.pth -o weights.pth   # 默认 <ckpt>_weights.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WEIGHTS_SUFFIX


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", help="train.py 保存的 best.pth / last.pth")
    parser.add_argument("-o", "--output", default=None, help="默认 <ckpt 同目录>/<名字>_weights.pth")
    args = parser.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    output = args.output or os.path.splitext(args.ckpt)[0] + WEIGHTS_SUFFIX
    torch.save(state, output)
    print(f"saved {len(state)} tensors -> {output}")


if __name__ == "__main__":
    main()
