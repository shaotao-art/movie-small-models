#!/usr/bin/env python
"""把 all_images.jsonl 转成 ms-swift SFT 格式（每张图一条：<image> + 提示词 -> 结果 JSON）。

用法:
    python data_making/srt_model/make_sft_jsonl.py
    python data_making/srt_model/make_sft_jsonl.py -i data/all_images.jsonl -o outputs/subtitle_sft.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import ALL_IMAGES_JSONL, SFT_JSONL, SFT_PROMPT


def image_path(path):
    """jsonl 里记录的是旧路径 frames/；frames 挪到 data/ 后自动兼容。"""
    if path.startswith("frames" + os.sep):
        path = os.path.join("data", path)
    return os.path.abspath(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default=ALL_IMAGES_JSONL)
    parser.add_argument("-o", "--output", default=SFT_JSONL)
    parser.add_argument("--prompt", default=SFT_PROMPT)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n = skipped = n_sub = 0
    with open(args.input) as f, open(args.output, "w") as out:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            result, image = rec.get("result"), rec.get("image")
            if not result or not image:
                skipped += 1
                continue
            messages = [
                {"role": "user", "content": "<image>" + args.prompt},
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
            ]
            out.write(json.dumps({"messages": messages, "images": [image_path(image)]}, ensure_ascii=False) + "\n")
            n += 1
            n_sub += bool(result.get("has_subtitle"))

    print(f"wrote {n} samples ({n_sub} has_subtitle, {n - n_sub} no_subtitle, skipped {skipped}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
