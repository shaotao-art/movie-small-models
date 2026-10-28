#!/usr/bin/env python
"""按字幕文本使用不同区域替换视频里的硬字幕。

坐标使用笛卡尔坐标系（原点左下角，y 向上）：
    我们在西沙群岛观光了瑰丽无比的海水， -> (0, 204) - (2886, 378)
    这节课，栏目组继续邀请大家作为小记者， -> (0, 204) - (2942, 378)
其它字幕默认 (0, 204) - (2850, 378)；没有字幕的帧不做替换。
替换像素来自一张干净背景图（默认 data/xinluT/cleaned_bg_3840x2160.png）。

用法:
    python -m render.patch_video_subtitles
    python -m render.patch_video_subtitles --video data/xinluT/376_raw.MP4 \
        --subtitles outputs/analyze_376/376_raw_subtitles.jsonl \
        --output data/xinluT/gen_video_patched.MP4
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import cv2
from tqdm import tqdm

# 字幕文本 -> 替换区域 (x0, y0, x1, y1)，笛卡尔坐标
REPLACE_RECTS = {
    "我们在西沙群岛观光了瑰丽无比的海水，": (0, 204, 2886, 378),
    "这节课，栏目组继续邀请大家作为小记者，": (0, 204, 2942, 378),
}
DEFAULT_RECT = (0, 204, 2850, 378)


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text or ""))


def to_slices(rect, height: int, width: int):
    """笛卡尔坐标（左下原点）转图像切片。"""
    x0, y0, x1, y1 = rect
    top, bottom = max(0, height - y1), min(height, height - y0)
    left, right = max(0, x0), min(width, x1)
    return slice(top, bottom), slice(left, right)


def pick_rect(text: str, similarity: float):
    key = norm(text)
    best_rect, best_name, best_score = DEFAULT_RECT, "其它字幕", 0.0
    for name, rect in REPLACE_RECTS.items():
        score = SequenceMatcher(None, key, norm(name)).ratio()
        if score >= similarity and score > best_score:
            best_rect, best_name, best_score = rect, name, score
    return best_rect, best_name


def find_ffmpeg():
    """优先用带 libx264 的 ffmpeg，没有就退回系统 ffmpeg 的 mpeg4。"""
    candidates = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates = [exe for exe in candidates if exe]
    if not candidates:
        raise SystemExit("找不到 ffmpeg")
    for exe in candidates:
        try:
            out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        except Exception:
            continue
        if "libx264" in out:
            return exe, ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
    return candidates[0], ["-c:v", "mpeg4", "-q:v", "2"]


def load_frame_texts(path: str, fps: float) -> dict[int, str]:
    """frame_index -> 字幕文本（只保留判定有字幕的帧）。"""
    frame_text = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            result = rec.get("result") or {}
            text = (result.get("subtitle_cn") or "").strip()
            if not result.get("has_subtitle") or not text:
                continue
            idx = rec.get("frame_index")
            frame_text[int(idx if idx is not None else round(rec["timestamp"] * fps))] = text
    return frame_text


def patch_video(ffmpeg: str, venc: list[str], video: str, subtitles: str,
                clean_image: str, output: str, similarity: float = 0.8,
                audio_from: str = None) -> tuple[int, float, Counter]:
    """逐帧替换字幕区域并编码；audio_from 不为空时把它的音轨 copy 过来。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clean = cv2.imread(clean_image)
    if clean is None or clean.shape[:2] != (height, width):
        raise SystemExit(f"干净背景图必须是 {width}x{height}: {clean_image}")

    frame_text = load_frame_texts(subtitles, fps)
    patches = {}
    for rect in {DEFAULT_RECT, *REPLACE_RECTS.values()}:
        ys, xs = to_slices(rect, height, width)
        patches[rect] = clean[ys, xs].copy()

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    ]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [*venc, "-pix_fmt", "yuv420p", "-frames:v", str(total),
            "-movflags", "+faststart", output]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    counts = Counter()
    idx = 0
    try:
        with tqdm(total=total or None, desc="patch", unit="frame") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                text = frame_text.get(idx)
                if text:
                    rect, name = pick_rect(text, similarity)
                    ys, xs = to_slices(rect, height, width)
                    frame[ys, xs] = patches[rect]
                    counts[name] += 1
                else:
                    counts["无字幕（不处理）"] += 1
                proc.stdin.write(frame.tobytes())
                idx += 1
                bar.update(1)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg 退出码 {proc.returncode}")
    except BrokenPipeError:
        proc.wait()
        raise SystemExit(f"ffmpeg 失败（退出码 {proc.returncode}）")
    finally:
        cap.release()
    return idx, fps, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/xinluT/376_raw.MP4")
    parser.add_argument("--subtitles", default="outputs/analyze_376/376_raw_subtitles.jsonl")
    parser.add_argument("--clean-image", default="data/xinluT/cleaned_bg_3840x2160.png",
                        help="干净背景图（与视频同分辨率）")
    parser.add_argument("-o", "--output", default="data/xinluT/gen_video_patched.MP4")
    parser.add_argument("--similarity", type=float, default=0.8, help="字幕文本模糊匹配阈值")
    args = parser.parse_args(argv)

    if os.path.abspath(args.output) == os.path.abspath(args.video):
        parser.error("输出不能和输入视频相同")
    ffmpeg, venc = find_ffmpeg()
    frames, _, counts = patch_video(
        ffmpeg, venc, args.video, args.subtitles, args.clean_image, args.output,
        args.similarity, audio_from=args.video,
    )
    print(f"{frames} 帧 -> {args.output}  (ffmpeg: {ffmpeg})")
    for name, n in counts.most_common():
        print(f"  {n:4d} 帧  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
