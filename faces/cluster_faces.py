#!/usr/bin/env python
"""全片人脸检测 + 命名/聚类。

每 --every 秒抽一帧，YuNet 检测人脸、SFace 提特征：
- 给了 --refs-dir：先跟参考图比对，匹配上的人直接用人名；
- 匹配不上的继续按余弦相似度在线聚类（与簇中心相似度超过阈值就归入，否则新建），
  最后合并质心相近的簇，每个簇保存几张代表脸，方便人工确认是谁。

复用 faces/face_recognizer.py 的模型加载/检测/特征提取/参考图匹配。

用法:
    python -m faces.cluster_faces                              # 纯聚类
    python -m faces.cluster_faces --refs-dir data/tlou/ref_face # 参考图 + 聚类兜底
输出（默认 outputs/face_clusters/<视频名>/）:
    faces.jsonl    每张脸：时间、帧号、bbox、分数、person（人名或 person_XX）
    clusters.json  角色/簇：检测数、出现时间段、代表脸图片
    <person>_N.jpg 代表脸
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from common_utils.file_utils import save_image
from config import (
    CHARACTER_REFS, FACE_CLUSTER_CROPS, FACE_CLUSTER_DIR, FACE_CLUSTER_MIN_FACES,
    FACE_CLUSTER_SAMPLE_SECONDS, FACE_CLUSTER_THRESH, FACE_MATCH_THRESH, VIDEO_PATH,
)
from faces.face_recognizer import (
    build_gallery, detect_faces, embed_faces, ensure_models, load_models, match_feature,
)


def collect_refs(refs_dir: str) -> dict[str, str]:
    """读参考图目录，文件名作角色名；config.CHARACTER_REFS 里已有的路径沿用它的名字。"""
    configured = {
        os.path.abspath(p): name
        for name, value in CHARACTER_REFS.items()
        for p in ([value] if isinstance(value, str) else value)
    }
    refs = {}
    for name in sorted(os.listdir(refs_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(refs_dir, name)
            refs[configured.get(os.path.abspath(path), os.path.splitext(name)[0])] = path
    return refs


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def add_crop(entry: dict, score: float, crop: np.ndarray, keep: int) -> None:
    """每簇只保留分数最高的 keep 张，避免把几千张 112x112 人脸全存内存。"""
    entry["crops"].append((score, crop))
    entry["crops"].sort(key=lambda item: -item[0])
    del entry["crops"][keep:]


def merge_clusters(clusters: dict, alias: dict, threshold: float) -> None:
    """质心相似度超过阈值的簇合并；记录被并入的 uid，便于回填标签。"""
    uids = list(clusters)
    merged = True
    while merged:
        merged = False
        for i, a_uid in enumerate(uids):
            for b_uid in uids[i + 1:]:
                a, b = clusters[a_uid], clusters[b_uid]
                if float(unit(a["sum"]) @ unit(b["sum"])) >= threshold:
                    a["sum"] = a["sum"] + b["sum"]
                    a["n"] += b["n"]
                    a["first"] = min(a["first"], b["first"])
                    a["last"] = max(a["last"], b["last"])
                    a["crops"] += b["crops"]
                    alias[b_uid] = a_uid
                    del clusters[b_uid]
                    uids.remove(b_uid)
                    merged = True
                    break
            if merged:
                break


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--every", type=float, default=FACE_CLUSTER_SAMPLE_SECONDS, help="抽帧间隔（秒）")
    parser.add_argument("--thresh", type=float, default=FACE_CLUSTER_THRESH, help="归入簇的余弦相似度阈值")
    parser.add_argument("--merge-thresh", type=float, default=None, help="簇合并阈值，默认同 --thresh")
    parser.add_argument("--min-faces", type=int, default=FACE_CLUSTER_MIN_FACES, help="少于该检测数的簇标为 noise")
    parser.add_argument("--refs-dir", default=None, help="参考图目录；给了就按人脸识别结果命名")
    parser.add_argument("--refs", action="store_true", help="使用 config.CHARACTER_REFS 里的参考图")
    parser.add_argument("--match-thresh", type=float, default=FACE_MATCH_THRESH,
                        help="与参考图的匹配阈值")
    parser.add_argument("--crops", type=int, default=FACE_CLUSTER_CROPS, help="每簇保存几张代表脸")
    parser.add_argument("-o", "--output-dir", default=FACE_CLUSTER_DIR)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.video):
        parser.error(f"找不到视频: {args.video}")
    detector_path, recognizer_path = ensure_models()
    detector, recognizer = load_models(detector_path, recognizer_path)
    refs = CHARACTER_REFS if args.refs else (collect_refs(args.refs_dir) if args.refs_dir else {})
    gallery = build_gallery(refs, detector, recognizer) if refs else {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error(f"打不开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    step = max(1, round(fps * args.every))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clusters, alias, records, named = {}, {}, [], {}
    next_uid, idx, n_faces, n_sampled = 0, 0, 0, 0
    started = time.time()
    estimate = (total + step - 1) // step if total > 0 else None
    with tqdm(total=estimate, desc="faces", unit="frame") as bar:
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                n_sampled += 1
                ts = idx / fps
                faces = detect_faces(detector, frame)
                for face, (aligned, feat) in zip(faces, embed_faces(recognizer, frame, faces)):
                    raw = np.asarray(feat)
                    vec = unit(raw.reshape(-1))
                    quality = float(face[2] * face[3] * face[14])
                    record = {
                        "timestamp": round(ts, 3),
                        "frame_index": idx,
                        "bbox": [round(float(v), 1) for v in face[:4]],
                        "score": round(float(face[14]), 3),
                    }

                    name, match = match_feature(recognizer, raw, gallery) if gallery else (None, 0.0)
                    if name and match >= args.match_thresh:
                        ent = named.setdefault(name, {"n": 0, "first": ts, "last": ts, "crops": []})
                        ent["n"] += 1
                        ent["first"] = min(ent["first"], ts)
                        ent["last"] = max(ent["last"], ts)
                        add_crop(ent, quality, aligned, args.crops)
                        record.update(person=name, match=round(float(match), 3))
                    else:
                        uid, best = None, -1.0
                        for c_uid, c in clusters.items():
                            sim = float(vec @ unit(c["sum"]))
                            if sim > best:
                                uid, best = c_uid, sim
                        if uid is None or best < args.thresh:
                            uid = next_uid
                            next_uid += 1
                            clusters[uid] = {"sum": np.zeros_like(vec), "n": 0,
                                             "first": ts, "last": ts, "crops": []}
                        c = clusters[uid]
                        c["sum"] = c["sum"] + vec
                        c["n"] += 1
                        c["first"] = min(c["first"], ts)
                        c["last"] = max(c["last"], ts)
                        add_crop(c, quality, aligned, args.crops)
                        record.update(person=uid)
                    records.append(record)
                    n_faces += 1
                bar.update(1)
            idx += 1
    cap.release()

    merge_clusters(clusters, alias, args.merge_thresh if args.merge_thresh is not None else args.thresh)
    ordered = sorted(clusters.items(), key=lambda kv: -kv[1]["n"])
    kept = [uid for uid, c in ordered if c["n"] >= args.min_faces]
    labels = {uid: f"person_{rank:02d}" for rank, uid in enumerate(kept)}

    def resolve(uid: int) -> int:
        while uid in alias:
            uid = alias[uid]
        return uid

    run_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "faces.jsonl"), "w") as f:
        for rec in records:
            if isinstance(rec["person"], int):
                rec["person"] = labels.get(resolve(rec["person"]), "noise")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def save_crops(entry: dict, label: str) -> list[str]:
        crops = []
        for i, (_, crop) in enumerate(sorted(entry["crops"], key=lambda x: -x[0])[:args.crops], 1):
            path = os.path.join(run_dir, f"{label}_{i}.jpg")
            save_image(crop, path)
            crops.append(os.path.basename(path))
        return crops

    characters = {}
    for name, ent in sorted(named.items(), key=lambda kv: -kv[1]["n"]):
        characters[name] = {
            "faces": ent["n"],
            "first_seen": round(ent["first"], 3),
            "last_seen": round(ent["last"], 3),
            "crops": save_crops(ent, name),
        }
    unknown = {}
    for uid in kept:
        c = clusters[uid]
        label = labels[uid]
        unknown[label] = {
            "faces": c["n"],
            "first_seen": round(c["first"], 3),
            "last_seen": round(c["last"], 3),
            "crops": save_crops(c, label),
        }
    with open(os.path.join(run_dir, "clusters.json"), "w") as f:
        json.dump({
            "video": os.path.abspath(args.video),
            "sample_seconds": args.every,
            "thresh": args.thresh,
            "match_thresh": args.match_thresh,
            "faces_total": n_faces,
            "characters": characters,
            "unknown_clusters": unknown,
        }, f, ensure_ascii=False, indent=2)

    print(f"{n_sampled} 帧（每 {args.every}s）检测到 {n_faces} 张脸，"
          f"识别出 {len(characters)} 个角色，"
          f"未知 {len(ordered)} 簇（保留 {len(kept)}，noise {len(ordered) - len(kept)}），"
          f"耗时 {time.time() - started:.0f}s")
    for name, info in characters.items():
        print(f"  {name}: {info['faces']:>4} 张  "
              f"{fmt_time(info['first_seen'])}-{fmt_time(info['last_seen'])}  {info['crops']}")
    for label, info in unknown.items():
        print(f"  {label} (未知): {info['faces']:>4} 张  "
              f"{fmt_time(info['first_seen'])}-{fmt_time(info['last_seen'])}  {info['crops']}")
    print(f"输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
