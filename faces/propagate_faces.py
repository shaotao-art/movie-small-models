#!/usr/bin/env python
"""离线对 cluster_faces 的 faces.jsonl 做场景内双向人名扩展，不重新解码视频。

应用已经识别出的人名（如「比尔」）：同一 scene 内，相邻采样帧的人脸框 IoU > 0.6
时，把名字扩展到未识别的框上；先正向、再反向各一遍。

用法:
    python -m faces.propagate_faces
    python -m faces.propagate_faces outputs/face_clusters/tlou_s1e3/faces.jsonl \
        -o outputs/face_clusters/tlou_s1e3/faces_propagated.jsonl --iou 0.6
"""

import argparse
import bisect
import json
import os
from collections import Counter, defaultdict

from config import CHARACTER_REFS, FACE_TRACK_IOU, SCENES_PKL
from faces.face_recognizer import propagate_labels_bidirectional
from scenes.scene_det import load_scene_det_res


def propagate_run(frames: dict[int, list[dict]], frame_ids: list[int], iou_thresh: float) -> int:
    """对一个连续采样段做双向扩展；返回被扩展的检测数。"""
    seq, refs = [], []
    for frame_idx in frame_ids:
        dets = []
        for rec in frames[frame_idx]:
            person = rec.get("person")
            known = person in CHARACTER_REFS
            det = {
                "bbox": rec["bbox"],
                "name": person if known else None,
                "score": float(rec.get("match") or 0.0) if known else 0.0,
            }
            dets.append(det)
            refs.append((rec, det))
        seq.append(dets)
    propagate_labels_bidirectional(seq, iou_thresh)

    propagated = 0
    for rec, det in refs:
        if det["name"] and rec.get("person") != det["name"]:
            rec["person"] = det["name"]
            rec["propagated"] = True
            propagated += 1
    return propagated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?",
                        default="outputs/face_clusters/tlou_s1e3/faces.jsonl")
    parser.add_argument("-o", "--output", default=None, help="默认在输入同目录生成 faces_propagated.jsonl")
    parser.add_argument("--scenes-pkl", default=SCENES_PKL, help="镜头列表，用于划分 scene；不存在时整片当一场")
    parser.add_argument("--no-scenes", action="store_true", help="不按 scene 划分，整段视频按帧间隔分组")
    parser.add_argument("--iou", type=float, default=FACE_TRACK_IOU)
    parser.add_argument("--max-gap", type=float, default=1.5, help="相邻采样帧最大时间间隔（秒）")
    args = parser.parse_args(argv)

    output = args.output or os.path.join(os.path.dirname(args.input), "faces_propagated.jsonl")
    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    scenes = [] if args.no_scenes or not os.path.exists(args.scenes_pkl) else load_scene_det_res(args.scenes_pkl)
    starts = [s["start_time"] for s in scenes]

    # scene -> frame_index -> records；没有 scene 时整片一组（-1）
    groups = defaultdict(lambda: defaultdict(list))
    for rec in records:
        ts = rec["timestamp"]
        scene_idx = bisect.bisect_right(starts, ts) - 1 if scenes else -1
        groups[scene_idx][int(rec["frame_index"])].append(rec)

    propagated = 0
    for frames in groups.values():
        frame_ids = sorted(frames)
        run, prev_ts = [], None
        for frame_idx in frame_ids:
            ts = frames[frame_idx][0]["timestamp"]
            if run and prev_ts is not None and ts - prev_ts > args.max_gap:
                propagated += propagate_run(frames, run, args.iou)
                run = []
            run.append(frame_idx)
            prev_ts = ts
        if run:
            propagated += propagate_run(frames, run, args.iou)

    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    seeds = sum(1 for r in records if r.get("person") in CHARACTER_REFS and not r.get("propagated"))
    named = Counter(r["person"] for r in records if r.get("person") in CHARACTER_REFS)
    named_total = sum(named.values())
    print(f"{len(records)} 张脸 -> {output}")
    print(f"已识别人名：{seeds} 张种子 + {propagated} 张扩展 = {named_total} 张")
    print("各角色：" + "，".join(f"{k} {v}" for k, v in named.most_common()))
    print(f"未知/聚类：{len(records) - named_total} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
