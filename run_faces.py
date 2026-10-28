#!/usr/bin/env python
"""一键人脸流程：检测 + 参考图识别 + 聚类，再做场景内双向人名扩展。

参数全部来自 config.py：
    CHARACTER_REFS / FACE_MATCH_THRESH / FACE_TRACK_IOU
    FACE_CLUSTER_SAMPLE_SECONDS / FACE_CLUSTER_THRESH / FACE_CLUSTER_MIN_FACES
    FACE_CLUSTER_DIR / SCENES_PKL

用法:
    python run_faces.py                              # 默认 config.VIDEO_PATH
    python run_faces.py data/xinluT/376_raw.MP4
"""

import argparse
import os

import cv2

from cluster_faces import main as cluster_main
from config import FACE_CLUSTER_DIR, FACE_TRACK_IOU, SCENES_PKL, VIDEO_PATH
from propagate_faces import main as propagate_main
from scene_det import load_scene_det_res


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps else 0.0


def scenes_match_video(video: str) -> bool:
    """scene pkl 是否属于这个视频：用最后一镜的结束时间和视频时长粗略判断。"""
    if not os.path.exists(SCENES_PKL):
        return False
    scenes = load_scene_det_res(SCENES_PKL)
    if not scenes:
        return False
    duration = video_duration(video)
    last_end = max(s["end_time"] for s in scenes)
    return abs(last_end - duration) <= max(10.0, duration * 0.01)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.video):
        raise SystemExit(f"找不到视频: {args.video}")

    cluster_main(["--video", args.video, "--refs"])

    stem = os.path.splitext(os.path.basename(args.video))[0]
    run_dir = os.path.join(FACE_CLUSTER_DIR, stem)
    faces = os.path.join(run_dir, "faces.jsonl")
    propagated = os.path.join(run_dir, "faces_propagated.jsonl")
    if not os.path.isfile(faces):
        raise SystemExit(f"没有检测结果: {faces}")

    prop_args = [faces, "-o", propagated, "--iou", str(FACE_TRACK_IOU)]
    if scenes_match_video(args.video):
        prop_args += ["--scenes-pkl", SCENES_PKL]
    else:
        print(f"scene pkl 与视频不匹配，按整段视频 + 帧间隔做扩展")
        prop_args += ["--no-scenes"]
    propagate_main(prop_args)

    print(f"\n检测结果: {faces}")
    print(f"扩展后结果: {propagated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
