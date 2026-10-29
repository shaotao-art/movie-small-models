#!/usr/bin/env python
"""把 data/movie 下所有视频的音频抽到 data/audio，已抽过的自动跳过。

用法:
    python -m tools.extract_audio                 # 默认 m4a（AAC 192k，体积小）
    python -m tools.extract_audio --format wav    # 16k 单声道 wav（喂 FunASR 用）
    python -m tools.extract_audio --overwrite     # 忽略已有文件重抽
    python -m tools.extract_audio --src data/movie --dst data/audio
"""

import argparse
import os
import subprocess
import sys

from render.patch_video_subtitles import find_ffmpeg

VIDEO_EXT = {".mp4", ".mkv", ".rmvb", ".avi", ".mov", ".ts", ".m4v", ".flv", ".wmv"}
AUDIO_ARGS = {
    "m4a": ["-c:a", "aac", "-b:a", "192k"],
    "wav": ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/movie", help="视频目录（默认 data/movie）")
    parser.add_argument("--dst", default="data/audio", help="音频输出目录（默认 data/audio）")
    parser.add_argument("--format", choices=sorted(AUDIO_ARGS), default="m4a", help="输出格式（默认 m4a）")
    parser.add_argument("--overwrite", action="store_true", help="已存在也重抽")
    args = parser.parse_args(argv)

    ffmpeg, _ = find_ffmpeg()
    os.makedirs(args.dst, exist_ok=True)

    videos = sorted(
        p for p in os.listdir(args.src)
        if not p.startswith(".") and os.path.splitext(p)[1].lower() in VIDEO_EXT
    )
    done, skipped, failed = 0, 0, []
    for i, name in enumerate(videos, 1):
        src = os.path.join(args.src, name)
        out = os.path.join(args.dst, os.path.splitext(name)[0] + "." + args.format)
        if os.path.exists(out) and os.path.getsize(out) > 0 and not args.overwrite:
            skipped += 1
            continue
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
               "-vn", "-map", "0:a:0", *AUDIO_ARGS[args.format], out]
        print(f"[{i}/{len(videos)}] {name}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
            failed.append((name, (r.stderr or "无音频流").strip().splitlines()[-1:]))
            print(f"    失败: {r.stderr.strip()[-300:]}", flush=True)
            continue
        done += 1

    print(f"\n完成：抽出 {done}，跳过 {skipped}，失败 {len(failed)} -> {args.dst}")
    for name, err in failed:
        print(f"  {name}: {' '.join(err)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
