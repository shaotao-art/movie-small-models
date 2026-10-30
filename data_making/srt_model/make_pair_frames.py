#!/usr/bin/env python
"""为「两帧台词是否相同」任务抽帧：每个 SRT cue 抽 K 帧存 jpg，并写出 cues.jsonl。

用法:
    python -m data_making.srt_model.make_pair_frames \
        --video data/test/阿甘正传_1min.mp4 \
        --srt outputs/analyze/阿甘正传_1min.srt \
        --out data/pairs

输出结构（PairDataset 直接吃这个目录）：
    data/pairs/frames/<片名>/<cue_id>_<k>.jpg
    data/pairs/cues/<片名>.jsonl   {movie, cue_id, start, end, text, frames:[...]}
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2

from subtitles.srt_preprocessor import SrtProcesser

logger = logging.getLogger("make_pair_frames")


def cue_times(start, end, k):
    """cue 内均匀取 k 个时间点，两端各留 15%（最多 0.2s）避开字幕进出场。"""
    margin = min(0.2, (end - start) * 0.15)
    lo, hi = start + margin, end - margin
    if hi <= lo:
        lo, hi = start + (end - start) * 0.1, end - (end - start) * 0.1
    return [lo + (hi - lo) * (i + 0.5) / k for i in range(k)]


def extract(video, srt, out_dir, k, short_side, overwrite, limit=None):
    movie = os.path.splitext(os.path.basename(video))[0]
    cues_path = os.path.join(out_dir, "cues", movie + ".jsonl")
    if os.path.exists(cues_path) and not overwrite:
        logger.info("已存在，跳过（--overwrite 重抽）: %s", cues_path)
        return 0

    lines = [l for l in SrtProcesser(srt).lines if l.text.strip()]
    if limit:
        lines = lines[:limit]
    if not lines:
        logger.error("SRT 里没有台词: %s", srt)
        return 1

    frames_dir = os.path.join(out_dir, "frames", movie)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cues_path), exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        logger.error("打不开视频: %s", video)
        return 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = short_side / min(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    logger.info("%s %dx%d -> %dx%d, %d 条 cue x %d 帧", movie, w, h, size[0], size[1], len(lines), k)

    written, records = 0, []
    for ci, line in enumerate(lines):
        frames = []
        for ki, t in enumerate(cue_times(line.start_time, line.end_time, k)):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            path = os.path.join(frames_dir, f"{ci:04d}_{ki}.jpg")
            cv2.imwrite(path, cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(os.path.relpath(path, out_dir))
            written += 1
        if len(frames) < 2:
            logger.warning("cue %d 只抽到 %d 帧，丢弃: %s", ci, len(frames), line.text[:30])
            continue
        records.append({
            "movie": movie, "cue_id": ci,
            "start": round(line.start_time, 3), "end": round(line.end_time, 3),
            "text": line.text, "frames": frames,
        })
    cap.release()

    with open(cues_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("完成：%d 条 cue，%d 张帧 -> %s", len(records), written, cues_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--out", default="data/pairs", help="输出根目录（默认 data/pairs）")
    parser.add_argument("-k", "--frames-per-cue", type=int, default=4)
    parser.add_argument("--short-side", type=int, default=480, help="帧短边像素（默认 480，太小字幕读不出来）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条 cue（默认全部）")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for p in (args.video, args.srt):
        if not os.path.isfile(p):
            logger.error("找不到文件: %s", p)
            return 1
    return extract(args.video, args.srt, args.out, args.frames_per_cue, args.short_side,
                   args.overwrite, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
