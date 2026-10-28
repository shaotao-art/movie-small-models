#!/usr/bin/env python
"""按固定时间间隔从视频中抽帧。

用法:
    python get_frames.py movie.mp4                 # 每 60s 一帧, 输出到 data/frames/<md5(文件名)>/
    python get_frames.py movie.mp4 --interval 30   # 每 30s 一帧
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys

logger = logging.getLogger("get_frames")


def default_out_dir(video):
    name = os.path.basename(video)
    return os.path.join("data", "frames", hashlib.md5(name.encode()).hexdigest())


def _fps_expr(interval):
    # 整秒用有理数 1/60 保持精确，避免浮点误差在长视频里累积漂移
    return f"1/{interval:g}" if float(interval).is_integer() else f"{1 / interval:.10g}"


def extract_frames(video, out_dir, interval=60.0, quality=2):
    """用 ffmpeg 的 fps 滤镜抽帧，长视频也只需一遍解码。返回写出的帧数。"""
    if not os.path.isfile(video):
        raise FileNotFoundError(video)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video,
        "-vf", f"fps={_fps_expr(interval)}",
        "-q:v", str(quality),
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = [f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    logger.info("%d frames written to %s", len(frames), out_dir)
    return len(frames)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--out-dir", default=None, help="输出目录，默认 data/frames/<md5(文件名)>")
    parser.add_argument("-i", "--interval", type=float, default=60.0, help="抽帧间隔（秒），默认 60")
    parser.add_argument("-q", "--quality", type=int, default=2, help="JPEG 质量，2(最好)~31(最差)，默认 2")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir or default_out_dir(args.video)

    try:
        extract_frames(args.video, out_dir, args.interval, args.quality)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed with exit code %d", e.returncode)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
