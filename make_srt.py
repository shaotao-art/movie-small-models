#!/usr/bin/env python
"""把 analyze_video.py 的逐帧字幕结果合并成 SRT，容忍 VLM 的轻微抖动。

相邻帧文本（去掉标点/空格、NFKC 归一化后）相似度超过阈值、且时间间隔不大时
视为同一条字幕；夹在两条相同字幕之间的 1~2 帧异常结果会被吸收，不会产生
闪一下就消失的字幕。每条字幕取该组里出现次数最多的文本。

用法:
    python make_srt.py
    python make_srt.py outputs/analyze/xxx_subtitles.jsonl -o xxx.srt --similarity 0.8
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from statistics import median

from config import (
    ANALYZE_OUTPUT_DIR, SRT_MERGE_MAX_GAP, SRT_MERGE_MIN_DURATION,
    SRT_MERGE_SAME_GAP, SRT_MERGE_SIMILARITY, VIDEO_PATH,
)


def norm(cn: str, en: str) -> str:
    text = unicodedata.normalize("NFKC", f"{cn} {en}").lower()
    return re.sub(r"[\W_]+", "", text)


def similar(a: str, b: str, threshold: float) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= threshold


def ref_key(cue: dict) -> str:
    return cue["keys"].most_common(1)[0][0]


def add(cue: dict, o: dict) -> None:
    cue["last"] = o["ts"]
    cue["n"] += 1
    cue["keys"][o["key"]] += 1
    cue["texts"][(o["cn"], o["en"])] += 1


def new_cue(o: dict) -> dict:
    cue = {"start": o["ts"], "last": o["ts"], "n": 0, "keys": Counter(), "texts": Counter()}
    add(cue, o)
    return cue


def merge_text(dst: dict, src: dict) -> None:
    dst["last"] = max(dst["last"], src["last"])
    dst["n"] += src["n"]
    dst["keys"].update(src["keys"])
    dst["texts"].update(src["texts"])


def extend(dst: dict, src: dict) -> None:
    """只延长时间，不把抖动文本计入投票。"""
    dst["last"] = max(dst["last"], src["last"])


def load_obs(path: str) -> list[dict]:
    obs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            result = rec.get("result") or {}
            if not result.get("has_subtitle"):
                continue
            cn = (result.get("subtitle_cn") or "").strip()
            en = (result.get("subtitle_en") or "").strip()
            if cn or en:
                obs.append({"ts": rec["timestamp"], "cn": cn, "en": en, "key": norm(cn, en)})
    obs.sort(key=lambda o: o["ts"])
    return obs


def group(obs: list[dict], similarity: float, max_gap: float) -> list[dict]:
    cues = []
    for o in obs:
        if cues and o["ts"] - cues[-1]["last"] <= max_gap and \
                similar(ref_key(cues[-1]), o["key"], similarity):
            add(cues[-1], o)
        else:
            cues.append(new_cue(o))
    return cues


def absorb_outliers(cues: list[dict], similarity: float, max_gap: float,
                    max_outlier: int = 2) -> list[dict]:
    """吸收夹在两条相同字幕之间的短抖动。"""
    out = []
    for i, cue in enumerate(cues):
        if out:
            prev = out[-1]
            nxt = cues[i + 1] if i + 1 < len(cues) else None
            if cue["start"] - prev["last"] <= max_gap and \
                    similar(ref_key(prev), ref_key(cue), similarity):
                merge_text(prev, cue)
                continue
            if nxt and cue["n"] <= max_outlier and \
                    cue["start"] - prev["last"] <= 2 * max_gap and \
                    nxt["start"] - cue["last"] <= 2 * max_gap and \
                    similar(ref_key(prev), ref_key(nxt), similarity):
                extend(prev, cue)  # 抖动帧，并进上一条；下一轮再和 nxt 合并
                continue
        out.append(cue)
    return out


def merge_same(cues: list[dict], merge_gap: float) -> list[dict]:
    """相邻两条文字完全相同且间隔不大时合并（漏采导致的重复）。"""
    out = []
    for cue in cues:
        if out and ref_key(out[-1]) == ref_key(cue) and \
                cue["start"] - out[-1]["last"] <= merge_gap:
            merge_text(out[-1], cue)
        else:
            out.append(cue)
    return out


def assign_times(cues: list[dict], obs: list[dict], min_duration: float) -> None:
    diffs = [b["ts"] - a["ts"] for a, b in zip(obs, obs[1:]) if b["ts"] > a["ts"]]
    step = median(diffs) if diffs else 0.1
    for i, cue in enumerate(cues):
        end = max(cue["last"] + step, cue["start"] + min_duration)
        if i + 1 < len(cues):
            end = min(end, cues[i + 1]["start"] - 0.01)
        cue["end"] = max(end, cue["start"] + 0.05)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for i, cue in enumerate(cues, 1):
            cn, en = cue["texts"].most_common(1)[0][0]
            f.write(f"{i}\n{srt_time(cue['start'])} --> {srt_time(cue['end'])}\n")
            f.write("\n".join(t for t in (cn, en) if t) + "\n\n")


def main(argv=None):
    stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=os.path.join(ANALYZE_OUTPUT_DIR, stem + "_subtitles.jsonl"))
    parser.add_argument("-o", "--output", default=None, help="默认与输入同名的 .srt")
    parser.add_argument("--similarity", type=float, default=SRT_MERGE_SIMILARITY,
                        help="相邻帧文本相似度阈值")
    parser.add_argument("--max-gap", type=float, default=SRT_MERGE_MAX_GAP, help="相邻字幕帧最大间隔（秒）")
    parser.add_argument("--same-gap", type=float, default=SRT_MERGE_SAME_GAP,
                        help="同一句字幕漏采时的重连间隔（秒）")
    parser.add_argument("--min-duration", type=float, default=SRT_MERGE_MIN_DURATION,
                        help="单条字幕最短时长（秒）")
    args = parser.parse_args(argv)

    output = args.output or os.path.splitext(args.input)[0] + ".srt"
    obs = load_obs(args.input)
    cues = group(obs, args.similarity, args.max_gap)
    cues = absorb_outliers(cues, args.similarity, args.max_gap)
    cues = merge_same(cues, args.same_gap)
    assign_times(cues, obs, args.min_duration)
    write_srt(cues, output)

    print(f"{len(obs)} 帧 -> {len(cues)} 条字幕 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
