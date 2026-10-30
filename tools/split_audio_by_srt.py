#!/usr/bin/env python
"""按 SRT 把音频切成一句一条，输出 wav/m4a 片段 + segments.jsonl 清单。

用法:
    python -m tools.split_audio_by_srt --audio data/audio/阿甘正传.m4a --srt outputs/analyze/阿甘正传.srt
    python -m tools.split_audio_by_srt --audio a.m4a --srt a.srt -o data/segments/a --format m4a
    python -m tools.split_audio_by_srt --audio a.m4a --srt a.srt --overwrite
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render.patch_video_subtitles import find_ffmpeg
from subtitles.srt_preprocessor import SrtProcesser

AUDIO_ARGS = {
    "wav": ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"],
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
}
BAD_CHARS = re.compile(r'''[\\/:*?"<>|'!\s]+''')

logger = logging.getLogger("split_audio")


def safe_name(index: int, text: str, max_len: int = 30) -> str:
    """0007_我们准备撤退，非法字符替换成下划线。"""
    clean = BAD_CHARS.sub("_", text).strip("_")[:max_len]
    return f"{index:04d}_{clean}" if clean else f"{index:04d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, help="音频文件（mp3/m4a/wav/mp4 都行，ffmpeg 能解即可）")
    parser.add_argument("--srt", required=True, help="同一条时间线的 SRT")
    parser.add_argument("-o", "--output-dir", default=None, help="默认 data/segments/<音频名>")
    parser.add_argument("--format", choices=sorted(AUDIO_ARGS), default="wav", help="片段格式（默认 wav 16k 单声道）")
    parser.add_argument("--overwrite", action="store_true", help="已存在也重切")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for path in (args.audio, args.srt):
        if not os.path.isfile(path):
            logger.error("找不到文件: %s", path)
            return 1

    ffmpeg, _ = find_ffmpeg()
    logger.info("ffmpeg: %s", ffmpeg)
    out_dir = args.output_dir or os.path.join("data", "segments", os.path.splitext(os.path.basename(args.audio))[0])
    os.makedirs(out_dir, exist_ok=True)

    lines = [l for l in SrtProcesser(args.srt).lines if l.text.strip()]
    manifest_path = os.path.join(out_dir, "segments.jsonl")
    logger.info("audio=%s srt=%s 台词 %d 条 -> %s（%s）", args.audio, args.srt, len(lines), out_dir, args.format)
    made = skipped = 0
    with open(manifest_path, "w") as manifest:
        for i, line in enumerate(lines, 1):
            path = os.path.join(out_dir, f"{safe_name(i, line.text)}.{args.format}")
            manifest.write(json.dumps({
                "index": i, "file": os.path.basename(path),
                "start": round(line.start_time, 3), "end": round(line.end_time, 3),
                "duration": round(line.duration, 3), "text": line.text,
            }, ensure_ascii=False) + "\n")
            if os.path.exists(path) and os.path.getsize(path) > 0 and not args.overwrite:
                skipped += 1
                logger.debug("[%d/%d] 已存在，跳过: %s", i, len(lines), path)
                continue
            logger.info("[%d/%d] %.2f-%.2fs (%.2fs) %s", i, len(lines),
                        line.start_time, line.end_time, line.duration, line.text)
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{line.start_time:.3f}", "-t", f"{line.duration:.3f}",
                   "-i", args.audio, "-vn", "-map", "0:a:0", *AUDIO_ARGS[args.format], path]
            logger.debug("cmd: %s", " ".join(cmd))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("[%d/%d] 切分失败 %s: %s", i, len(lines), path, r.stderr.strip()[-300:])
                continue
            made += 1

    logger.info("完成：切出 %d，跳过 %d，失败 %d，共 %d 条 -> %s",
                made, skipped, len(lines) - made - skipped, len(lines), out_dir)
    logger.info("清单: %s", manifest_path)
    return 1 if made + skipped < len(lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())
