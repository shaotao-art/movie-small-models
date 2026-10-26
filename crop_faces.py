#!/usr/bin/env python
"""把某个 person（人物名或 person_XX 簇）的全部人脸从视频里裁出来，供人工检查。

按时间顺序解码一遍视频，只对 faces.jsonl 里该 person 的帧做裁剪。

用法:
    python crop_faces.py outputs/face_clusters/tlou_s1e3 --person person_00
    python crop_faces.py outputs/face_clusters/tlou_s1e3 --person 乔尔
输出:
    <结果目录>/<person>_crops/0001_t0015.02_f000360.jpg ...
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from config import FACE_CLUSTER_DIR, VIDEO_PATH


def main(argv=None):
    stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=os.path.join(FACE_CLUSTER_DIR, stem),
                        help="cluster_faces.py 的输出目录")
    parser.add_argument("--person", required=True, help="人名或 person_XX")
    parser.add_argument("--video", default=None, help="默认用 clusters.json 里记录的视频")
    parser.add_argument("-o", "--output", default=None, help="默认 <input>/<person>_crops")
    parser.add_argument("--limit", type=int, default=0, help="只导出前 N 张，0 为全部")
    args = parser.parse_args(argv)

    with open(os.path.join(args.input, "clusters.json")) as f:
        summary = json.load(f)
    with open(os.path.join(args.input, "faces.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]

    video = args.video or summary.get("video")
    if not video or not os.path.isfile(video):
        parser.error(f"找不到视频: {video}")
    targets = sorted(
        (r for r in records if r.get("person") == args.person),
        key=lambda r: r["frame_index"],
    )
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        parser.error(f"没有 {args.person} 的检测结果")

    out_dir = args.output or os.path.join(args.input, f"{args.person}_crops")
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):  # 清掉上一次导出的图，避免编号错位留下旧文件
        if name.endswith(".jpg"):
            os.remove(os.path.join(out_dir, name))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        parser.error(f"打不开视频: {video}")
    by_frame = {}
    for rec in targets:  # 同一帧可能有多张脸
        by_frame.setdefault(rec["frame_index"], []).append(rec)
    last_frame = targets[-1]["frame_index"]
    saved = 0
    idx = 0
    with tqdm(total=len(targets), desc=args.person, unit="face") as bar:
        while idx <= last_frame and cap.grab():
            recs = by_frame.get(idx)
            if recs:
                ok, frame = cap.retrieve()
                if ok:
                    for j, rec in enumerate(recs, 1):
                        x, y, w, h = (int(round(v)) for v in rec["bbox"])
                        x0, y0 = max(0, x), max(0, y)
                        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        crop = frame[y0:y1, x0:x1]
                        if crop.size:
                            saved += 1
                            suffix = f"_{j}" if len(recs) > 1 else ""
                            name = f"{saved:04d}_t{rec['timestamp']:08.2f}_f{idx:06d}{suffix}.jpg"
                            cv2.imwrite(os.path.join(out_dir, name), crop)
                            bar.update(1)
            idx += 1
    cap.release()

    print(f"{args.person}: 导出 {saved}/{len(targets)} 张 -> {out_dir}")
    return 0 if saved == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
