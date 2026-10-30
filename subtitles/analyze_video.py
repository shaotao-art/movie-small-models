#!/usr/bin/env python
"""整片字幕分析：分类器判断每帧有无字幕，再用 SFT VLM 提取字幕文本。

先跑测试（随机切 2min 片段）：
    python -m subtitles.analyze_video --test
全片分析：
    python -m subtitles.analyze_video

输出目录（测试为 outputs/analyze_test，全片为 outputs/analyze）：
    <stem>_cls.jsonl         每帧分类结果
    <stem>_subtitles.jsonl   VLM 逐帧提取结果
    <stem>_subtitles.txt     按时间排序、合并连续重复后的字幕
"""

import argparse
import json
import logging
import os
import random

import cv2

from config import (
    ANALYZE_CKPT, ANALYZE_CONFIG, ANALYZE_INTERVAL, ANALYZE_OUTPUT_DIR,
    ANALYZE_TEST_DIR, ANALYZE_TEST_SECONDS, CLASSIFIER_BATCH_SIZE, VIDEO_PATH,
    VLM_API_KEY, VLM_BASE_URL, VLM_BATCH_SIZE, VLM_WORKERS,
)
from subtitles.eval_video import main as run_classifier
from subtitles.infer_video import main as run_vlm

logger = logging.getLogger("analyze")


def cut_clip(video: str, out_path: str, start: float, seconds: float) -> float:
    """用 OpenCV 截取 [start, start+seconds)，不依赖 ffmpeg；返回实际时长。"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    written = 0
    try:
        while written < int(seconds * fps):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    return written / fps


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_transcript(sub_path: str, txt_path: str) -> tuple[int, int]:
    """把逐帧结果按时间排序、合并连续重复，写成可读文本；返回 (有效帧数, 行数)。"""
    records = []
    if os.path.exists(sub_path):
        with open(sub_path) as f:
            for line in f:
                rec = json.loads(line)
                result = rec.get("result") or {}
                if result.get("has_subtitle"):
                    records.append(rec)
    records.sort(key=lambda r: r.get("timestamp") or 0)

    lines, prev = [], None
    for rec in records:
        result = rec["result"]
        text = ((result.get("subtitle_cn") or "").strip(), (result.get("subtitle_en") or "").strip())
        if not any(text) or text == prev:
            continue
        prev = text
        body = f"{text[0]} | {text[1]}" if text[1] else text[0]
        lines.append(f"[{fmt_time(rec.get('timestamp') or 0)}] {body}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(records), len(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--test", action="store_true", help="随机截取一段短片段测试")
    parser.add_argument("--seconds", type=float, default=ANALYZE_TEST_SECONDS, help="测试片段时长（秒）")
    parser.add_argument("--start", type=float, default=None, help="测试片段起始秒，默认随机")
    parser.add_argument("--ckpt", default=ANALYZE_CKPT, help="分类器 checkpoint 或导出的纯权重")
    parser.add_argument("--config", default=ANALYZE_CONFIG, help="纯权重文件配套的训练配置")
    parser.add_argument("--interval", type=float, default=ANALYZE_INTERVAL, help="抽帧间隔（秒）")
    parser.add_argument("--batch-size", type=int, default=CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--vlm-url", default=VLM_BASE_URL)
    parser.add_argument("--api-key", default=VLM_API_KEY)
    parser.add_argument("--workers", type=int, default=VLM_WORKERS, help="VLM 并发数")
    parser.add_argument("--vlm-batch", type=int, default=VLM_BATCH_SIZE, help="VLM 解码批大小")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="已有分类结果也重新分类")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.path.isfile(args.video):
        parser.error(f"找不到视频: {args.video}")

    out_dir = args.output_dir or (ANALYZE_TEST_DIR if args.test else ANALYZE_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    video = args.video
    stem = os.path.splitext(os.path.basename(video))[0]

    if args.test:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        start = args.start if args.start is not None else random.uniform(0, max(0.0, total - args.seconds))
        stem = f"{stem}_clip_{int(start)}s_{int(args.seconds)}s"
        clip = os.path.join(out_dir, stem + ".mp4")
        logger.info("截取测试片段: %.0fs ~ %.0fs -> %s", start, start + args.seconds, clip)
        duration = cut_clip(video, clip, start, args.seconds)
        logger.info("实际时长 %.1fs", duration)
        video = clip

    cls_out = os.path.join(out_dir, stem + "_cls.jsonl")
    sub_out = os.path.join(out_dir, stem + "_subtitles.jsonl")
    txt_out = os.path.join(out_dir, stem + "_subtitles.txt")

    if os.path.exists(cls_out) and os.path.getsize(cls_out) > 0 and not args.force:
        logger.info("分类结果已存在，跳过分类（--force 强制重跑）: %s", cls_out)
    else:
        logger.info("分类 %s（每 %.1fs 一帧）-> %s", video, args.interval, cls_out)
        run_classifier([
            "--ckpt", args.ckpt, "--config", args.config, "--video", video,
            "--interval", str(args.interval), "--batch-size", str(args.batch_size),
            "-o", cls_out,
        ])

    logger.info("VLM 提取字幕 %s -> %s", args.vlm_url, sub_out)
    rc = run_vlm([
        cls_out, "--base-url", args.vlm_url, "--api-key", args.api_key,
        "--workers", str(args.workers), "--batch-size", str(args.vlm_batch),
        "-o", sub_out,
    ])

    with open(cls_out) as f:
        cls_records = [json.loads(line) for line in f if line.strip()]
    n_pos = sum(bool((r.get("result") or {}).get("has_subtitle")) for r in cls_records)
    n_extract, n_lines = write_transcript(sub_out, txt_out)

    logger.info(
        "分类帧数 %d（有字幕 %d），VLM 提取 %d 帧，去重后 %d 行",
        len(cls_records), n_pos, n_extract, n_lines,
    )
    logger.info("分类结果 %s", cls_out)
    logger.info("提取结果 %s", sub_out)
    logger.info("字幕文本 %s", txt_out)
    with open(txt_out) as f:
        for line in list(f)[:10]:
            print(line.rstrip())
    if rc:
        logger.error("VLM 有失败请求，重跑同一命令会自动续跑")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
