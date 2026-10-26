"""镜头切分：用 PySceneDetect 检测镜头边界，结果存成 pickle 供后续流程使用。

用法:
    python -m src.split_video --video data/tlou/最后生还者第03集.mp4
    # 顺带把每个镜头切出来（会在视频目录下产生大量 mp4 片段）
    python -m src.split_video --video data/tlou/最后生还者第03集.mp4 --split-clips
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

from common_utils.file_utils import load_from_pickle, save_to_pickle
import config

# reference: http://github.com/Breakthrough/PySceneDetect


def default_scene_pkl(video_path: str) -> str:
    """默认把场景结果存成视频同目录下的 <视频名>_scenes.pkl"""
    video_dir = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{stem}_scenes.pkl")


def split_video_into_scenes(
    video_path: str,
    out_pkl: str = None,
    threshold: float = config.CONTENT_DETECTOR_THRESHOLD,
    split_clips: bool = False,
    clips_dir: str = None,
):
    """检测镜头边界，保存场景列表

    Args:
        video_path: 输入视频
        out_pkl: 场景列表 pickle 输出路径，默认见 default_scene_pkl
        threshold: ContentDetector 的阈值，越小越敏感
        split_clips: 是否顺带调用 ffmpeg 把每个镜头切成独立片段
        clips_dir: 切片段输出目录（默认与视频同目录，会产生大量片段文件）
    Returns:
        scene_list: [(FrameTimecode(start), FrameTimecode(end)), ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"错误：找不到视频文件 '{video_path}'")
    out_pkl = out_pkl or default_scene_pkl(video_path)

    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    out_dir = os.path.dirname(os.path.abspath(out_pkl))
    os.makedirs(out_dir, exist_ok=True)
    save_to_pickle(scene_list, out_pkl)
    print(f"[split_video] 共 {len(scene_list)} 个镜头，已保存到 {out_pkl}")

    if split_clips:
        if clips_dir:
            os.makedirs(clips_dir, exist_ok=True)
        split_video_ffmpeg(video_path, scene_list, output_dir=clips_dir, show_progress=True)
    return scene_list


def load_scene_det_res(pkl_path: str) -> list[dict]:
    """读取场景 pkl，整理成统一的 dict 列表

    返回 [{"start_frame", "end_frame", "start_time", "end_time"}, ...]，
    start_time / end_time 单位为秒。
    """
    scene_det_res = []
    for start, end in load_from_pickle(pkl_path):
        scene_det_res.append(
            {
                "start_frame": start.frame_num,
                "end_frame": end.frame_num,
                "start_time": start.get_seconds(),
                "end_time": end.get_seconds(),
            }
        )
    return scene_det_res


def main():
    parser = argparse.ArgumentParser(description="PySceneDetect 镜头切分")
    parser.add_argument("--video", default=config.VIDEO_PATH)
    parser.add_argument("--out", default=None, help="场景列表 pkl 输出路径，默认与视频同目录")
    parser.add_argument(
        "--threshold", type=float, default=config.CONTENT_DETECTOR_THRESHOLD,
        help="ContentDetector 阈值"
    )
    parser.add_argument(
        "--split-clips", action="store_true", help="同时用 ffmpeg 切出每个镜头片段"
    )
    parser.add_argument("--clips-dir", default=None, help="镜头片段输出目录，默认与视频同目录")
    args = parser.parse_args()
    split_video_into_scenes(
        args.video,
        out_pkl=args.out,
        threshold=args.threshold,
        split_clips=args.split_clips,
        clips_dir=args.clips_dir,
    )


if __name__ == "__main__":
    main()
