"""裁剪无台词场景的视频片段，供 make_video_caption.py 生成画面描述。

只裁剪 scene 片段（字幕之间的空白段落）：有台词的部分由字幕文本表达，不需要
VLM 描述，所有 scene 片段都会被切成「子镜头」和「完整场景」两份。

用法:
    python -m src.cut_videos \
        --video data/tlou/最后生还者第03集.mp4 \
        --srt data/tlou/ep3.srt \
        --scenes-pkl data/tlou/tlou_scenes.pkl \
        --out-dir output/tlou_s1e3_scenes
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.srt_preprocessor import SrtProcesser
from src.split_video import load_scene_det_res
import config
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)


def build_scene_clips(srt: SrtProcesser) -> list[dict]:
    """把 srt.lines 里的 scene 片段展开成待裁剪的片段列表

    每个 scene 产出两类 clip：
    - 子镜头：uid 为 {scene_uid}_sub_{idx:04d}，按 scene detect 的结果细分
    - 完整场景：uid 为 {scene_uid}，覆盖整个无台词区间
    """
    scene_clips = []
    for line in srt.lines:
        if line.clip_type != "scene":
            continue
        for sub_idx, (start_time, end_time) in enumerate(line.splited_scene):
            scene_clips.append(
                {
                    "uid": line.uid + f"_sub_{sub_idx:04d}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": "",
                }
            )
        scene_clips.append(
            {
                "uid": line.uid,
                "start_time": line.start_time,
                "end_time": line.end_time,
                "caption": "",
            }
        )
    return scene_clips


def cut_scene_clips(video_path: str, scene_clips: list[dict], out_dir: str):
    """按片段列表裁剪视频并保存到 out_dir，返回输出路径列表"""
    os.makedirs(out_dir, exist_ok=True)
    video_processor = VideoProcessor(video_path)
    out_paths = []
    try:
        for idx, scene in enumerate(scene_clips, start=1):
            clip_path = os.path.join(out_dir, f"{scene['uid']}.mp4")
            video_processor.cut_video(scene["start_time"], scene["end_time"], clip_path)
            out_paths.append(clip_path)
            logging.info(f"[cut_videos] {idx}/{len(scene_clips)} {clip_path}")
    finally:
        video_processor.close()
    return out_paths


def main():
    parser = argparse.ArgumentParser(description="裁剪无台词场景片段")
    parser.add_argument("--video", default=config.VIDEO_PATH)
    parser.add_argument("--srt", default=config.SRT_PATH)
    parser.add_argument(
        "--scenes-pkl", default=config.SCENES_PKL, help="split_video 产出的场景列表"
    )
    parser.add_argument("--out-dir", default=config.SCENES_DIR)
    parser.add_argument(
        "--no-line-time-thres", type=float, default=config.NO_LINE_TIME_THRES,
        help="无台词场景的最小时长（秒）"
    )
    parser.add_argument(
        "--scene-iof-thres", type=float, default=config.SCENE_IOF_THRES,
        help="镜头与场景的 iof 阈值"
    )
    args = parser.parse_args()

    # 整理 scene detect 结果
    scene_det_res = load_scene_det_res(args.scenes_pkl)

    # 读取字幕文件
    srt = SrtProcesser(
        args.srt,
        no_line_time_thres=args.no_line_time_thres,
        scene_iof_thres=args.scene_iof_thres,
    )
    # 获取无台词场景，无台词时间 > threshold，插入到原始lines
    srt.get_no_line_scenes()
    # 根据scene detect结果，获取每个无台词场景切分镜头
    srt.get_splited_scenes(scene_det_res)

    scene_clips = build_scene_clips(srt)
    logging.info(f"[cut_videos] 待裁剪 {len(scene_clips)} 个场景片段")
    cut_scene_clips(args.video, scene_clips, args.out_dir)
    logging.info(f"[cut_videos] 完成，片段在 {args.out_dir}")


if __name__ == "__main__":
    main()
