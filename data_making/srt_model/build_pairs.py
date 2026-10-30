#!/usr/bin/env python
"""从 cues.jsonl 生成固定数量的 pair 清单：train.jsonl / val.jsonl。

用法:
    python -m data_making.srt_model.build_pairs --cues data/pairs/cues \
        --train-movies 阿甘正传 八恶人 --val-movies 爱乐之城 \
        --train-size 5000 --val-size 1000

每行一条样本：{a, b, label, type, text_a, text_b, movie_a, movie_b, cue_a, cue_b}
type 为 pos / hard / easy，比例固定 2:1:1（正 : 相邻难负 : 跨场景易负）。
"""

import argparse
import collections
import glob
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classifier.pair_data import load_cues, normalize_text

logger = logging.getLogger("build_pairs")


def pick_frames(cue, rng):
    """同一 cue 里取两帧，优先取时间跨度大的组合。"""
    frames = cue["frames"]
    if len(frames) < 2:
        return None
    a, b = rng.sample(range(len(frames)), 2)
    if abs(a - b) < 2:  # 尽量别挑相邻帧，避免模型靠“像素几乎相同”走捷径
        a, b = 0, len(frames) - 1
    return frames[a], frames[b]


def make_samples(cues, groups, n_pos, n_hard, n_easy, rng):
    samples = []
    pos_in_group = {i: k for group in groups.values() for k, i in enumerate(group)}

    def add(kind, label, i, j, fa, fb):
        ci, cj = cues[i], cues[j]
        samples.append({
            "a": fa, "b": fb, "label": label, "type": kind,
            "text_a": ci["text"], "text_b": cj["text"],
            "movie_a": ci["movie"], "movie_b": cj["movie"],
            "cue_a": ci["cue_id"], "cue_b": cj["cue_id"],
        })

    for _ in range(n_pos):
        while True:
            i = rng.randrange(len(cues))
            pair = pick_frames(cues[i], rng)
            if pair:
                add("pos", 1, i, i, *pair)
                break

    for _ in range(n_hard):
        done = False
        for _ in range(30):
            i = rng.randrange(len(cues))
            group = groups[cues[i]["movie"]]
            text = normalize_text(cues[i]["text"])
            for k in group[pos_in_group[i] + 1: pos_in_group[i] + 4]:  # 拆句的相邻 cue 跳过
                if normalize_text(cues[k]["text"]) == text:
                    continue
                pair = pick_frames(cues[i], rng)
                if pair and cues[k]["frames"]:
                    add("hard", 0, i, k, pair[0], rng.choice(cues[k]["frames"]))
                    done = True
                    break
            if done:
                break
        if not done:  # 整部片都是同一句（几乎不会）就退化成易负
            i, j = rng.randrange(len(cues)), rng.randrange(len(cues))
            add("easy", 0, i, j, rng.choice(cues[i]["frames"]), rng.choice(cues[j]["frames"]))

    for _ in range(n_easy):
        while True:
            i, j = rng.randrange(len(cues)), rng.randrange(len(cues))
            if i == j or normalize_text(cues[i]["text"]) == normalize_text(cues[j]["text"]):
                continue
            if cues[j]["movie"] == cues[i]["movie"] and abs(cues[j]["cue_id"] - cues[i]["cue_id"]) < 20:
                continue  # 同片子的近邻留给难负
            add("easy", 0, i, j, rng.choice(cues[i]["frames"]), rng.choice(cues[j]["frames"]))
            break
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cues", default="data/pairs/cues", help="cues 目录")
    parser.add_argument("-o", "--output", default="data/pairs", help="配对清单输出目录")
    parser.add_argument("--train-movies", nargs="+", required=True)
    parser.add_argument("--val-movies", nargs="+", required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cues = load_cues(args.cues)
    by_movie = collections.defaultdict(list)
    for c in cues:
        by_movie[c["movie"]].append(c)
    missing = [m for m in args.train_movies + args.val_movies if m not in by_movie]
    if missing:
        logger.error("缺少 cue 文件: %s（现有: %s）", missing, sorted(by_movie))
        return 1

    def split(movies):
        sel = [c for m in movies for c in by_movie[m]]
        groups = collections.defaultdict(list)
        for i, c in enumerate(sel):
            groups[c["movie"]].append(i)
        return sel, groups

    rng = random.Random(args.seed)
    train_cues, train_groups = split(args.train_movies)
    val_cues, val_groups = split(args.val_movies)
    logger.info("train 电影 %s(%d cue) / val 电影 %s(%d cue)",
                args.train_movies, len(train_cues), args.val_movies, len(val_cues))

    def ratios(n):
        return n // 2, n // 4, n - n // 2 - n // 4

    os.makedirs(args.output, exist_ok=True)
    for name, sel, groups, size in (
        ("train", train_cues, train_groups, args.train_size),
        ("val", val_cues, val_groups, args.val_size),
    ):
        n_pos, n_hard, n_easy = ratios(size)
        samples = make_samples(sel, groups, n_pos, n_hard, n_easy, rng)
        path = os.path.join(args.output, name + ".jsonl")
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        stats = collections.Counter(s["type"] for s in samples)
        logger.info("%s: %d 条 %s -> %s", name, len(samples), dict(stats), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
