#!/usr/bin/env python
"""把 all_images.jsonl 转成 ms-swift SFT 格式（每张图一条：<image> + 提示词 -> 结果 JSON）。

用法:
    python data_making/make_sft_jsonl.py
    python data_making/make_sft_jsonl.py -i data/all_images.jsonl -o outputs/subtitle_sft.jsonl
"""

import argparse
import json
import os

PROMPT = (
    "这是一帧电影画面。请判断画面中是否有字幕：有字幕则提取中文字幕和英文字幕，"
    "没有则 has_subtitle 为 false。只输出 JSON，格式为 "
    '{"has_subtitle": true, "subtitle_cn": "...", "subtitle_en": "..."}'
)


def image_path(path):
    """jsonl 里记录的是旧路径 frames/；frames 挪到 data/ 后自动兼容。"""
    if not os.path.isabs(path) and not path.startswith("data" + os.sep):
        path = os.path.join("data", path)
    return os.path.abspath(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default="data/all_images.jsonl")
    parser.add_argument("-o", "--output", default="outputs/subtitle_sft.jsonl")
    parser.add_argument("--prompt", default=PROMPT)
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
