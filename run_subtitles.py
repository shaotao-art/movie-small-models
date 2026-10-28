#!/usr/bin/env python
"""一键字幕流程：逐帧分类 -> VLM 提取字幕 -> 合并成 SRT。

参数全部来自 config.py：
    ANALYZE_CKPT / ANALYZE_CONFIG     分类器
    ANALYZE_INTERVAL / BATCH_SIZE     抽帧与批大小
    VLM_BASE_URL / VLM_WORKERS ...    已部署的字幕提取模型
    ANALYZE_OUTPUT_DIR                输出目录
    SRT_MERGE_*                       字幕抖动合并参数

用法:
    python run_subtitles.py                          # 默认 config.VIDEO_PATH
    python run_subtitles.py data/xinluT/376_raw.MP4
"""

import argparse
import os

from config import ANALYZE_OUTPUT_DIR, VIDEO_PATH
from analyze_video import main as analyze_main
from make_srt import main as srt_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.video):
        raise SystemExit(f"找不到视频: {args.video}")

    rc = analyze_main(["--video", args.video])
    stem = os.path.splitext(os.path.basename(args.video))[0]
    sub_jsonl = os.path.join(ANALYZE_OUTPUT_DIR, stem + "_subtitles.jsonl")
    srt = os.path.join(ANALYZE_OUTPUT_DIR, stem + ".srt")

    if os.path.isfile(sub_jsonl):
        srt_main([sub_jsonl, "-o", srt])
    else:
        print(f"没有字幕提取结果，跳过 SRT 生成: {sub_jsonl}")

    print(f"\n分类结果: {os.path.join(ANALYZE_OUTPUT_DIR, stem + '_cls.jsonl')}")
    print(f"字幕 JSONL: {sub_jsonl}")
    print(f"SRT: {srt}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
