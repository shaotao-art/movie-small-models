#!/usr/bin/env python
"""用字幕提取结果构建二分类数据集（有字幕 / 无字幕），按电影划分 train/val，多线程拷贝图片。

用法:
    python build_cls_dataset.py all_images.jsonl -o data/subtitle_cls
    python build_cls_dataset.py all_images.jsonl -o data/subtitle_cls --overwrite
"""

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("build_dataset")


def _copy(job):
    shutil.copy2(*job)


def resolve_image(path):
    """jsonl 里记录的是旧路径 frames/；frames 挪到 data/ 后自动兼容。"""
    if os.path.exists(path):
        return path
    moved = os.path.join("data", path)
    return moved if os.path.exists(moved) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="dist_use_qwen38.py 的输出")
    parser.add_argument("-o", "--output", default="data/subtitle_cls", help="数据集输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集图片占比，默认 0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16, help="拷贝线程数，默认 16")
    parser.add_argument("--overwrite", action="store_true", help="输出目录已存在时先删除")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if os.path.exists(args.output):
        if not args.overwrite:
            logger.error("%s already exists, pass --overwrite to rebuild", args.output)
            return 1
        shutil.rmtree(args.output)

    # 按电影目录分组，保证同一部电影的帧只出现在 train 或 val 一边
    by_movie = defaultdict(list)
    skipped = 0
    with open(args.jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            result = rec.get("result")
            image = resolve_image(rec["image"])
            if not result or image is None:
                skipped += 1
                continue
            movie = os.path.basename(os.path.dirname(image))
            label = "has_subtitle" if result.get("has_subtitle") else "no_subtitle"
            by_movie[movie].append((image, label))

    movies = sorted(by_movie)
    random.Random(args.seed).shuffle(movies)
    # 按电影整组划入 val，凑够约 val_ratio 的图片量，避免同片帧同时出现在两边
    target = sum(len(v) for v in by_movie.values()) * args.val_ratio
    val_movies, val_count = [], 0
    for movie in movies:
        if val_count >= target:
            break
        val_movies.append(movie)
        val_count += len(by_movie[movie])
    splits = {"val": set(val_movies), "train": set(movies) - set(val_movies)}

    counts = {}
    jobs = []
    for split, movie_set in splits.items():
        for movie in sorted(movie_set):
            for image, label in by_movie[movie]:
                dst_dir = os.path.join(args.output, split, label)
                os.makedirs(dst_dir, exist_ok=True)
                # 前缀电影哈希，避免不同电影的同名帧冲突
                dst = os.path.join(dst_dir, f"{movie}_{os.path.basename(image)}")
                jobs.append((image, dst))
                counts[(split, label)] = counts.get((split, label), 0) + 1

    total = len(jobs)
    step = max(1, total // 20)
    logger.info("copying %d images with %d threads", total, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, _ in enumerate(pool.map(_copy, jobs), 1):
            if done % step == 0 or done == total:
                logger.info("copy progress %d/%d (%d%%)", done, total, 100 * done // total)

    logger.info("skipped %d bad records, %d movies", skipped, len(movies))
    for split in ("train", "val"):
        total = sum(counts.get((split, c), 0) for c in ("has_subtitle", "no_subtitle"))
        logger.info(
            "%s: %d movies, %d images (has_subtitle=%d no_subtitle=%d)",
            split, len(splits[split]), total,
            counts.get((split, "has_subtitle"), 0), counts.get((split, "no_subtitle"), 0),
        )
    logger.info("dataset written to %s", os.path.abspath(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
