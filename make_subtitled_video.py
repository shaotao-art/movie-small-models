#!/usr/bin/env python
"""根据音频生成 SRT（调用 align_srt.py），并可把字幕烧录进视频。

用法:
    # 只生成字幕
    python make_subtitled_video.py

    # 生成字幕 + 烧录
    python make_subtitled_video.py --burn

    # 指定字体/字号/离底部距离
    python make_subtitled_video.py --burn \
        --font assets/AaRiFuYiRiKeAiYuan-2.ttf --font-size 96 --margin-v 204
"""

import argparse
import os
import re
import shutil
import subprocess

import cv2
from PIL import ImageFont

from align_srt import main as align_main
from patch_video_subtitles import find_ffmpeg


def escape_filter_path(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def font_family(font: str):
    """返回 (family, fontsdir)。优先用 Windows/Unicode 的 family 名，libass 才能匹配到。"""
    if not os.path.isfile(font):
        if os.path.sep in font or font.lower().endswith((".ttf", ".otf", ".ttc")):
            raise SystemExit(f"字体文件不存在: {font}")
        return font, None
    family = None
    try:
        from fontTools.ttLib import TTFont
        name = TTFont(font)["name"]
        candidates = []
        for rec in name.names:
            if rec.nameID in (1, 16):
                try:
                    value = rec.toUnicode()
                except Exception:
                    continue
                score = (0 if rec.platformID in (0, 3) else 1, 0 if rec.nameID == 1 else 1)
                candidates.append((score, value))
        if candidates:
            family = min(candidates)[1]
    except Exception:
        pass
    if not family:
        family = ImageFont.truetype(font, 20).getname()[0]
    return family, os.path.dirname(os.path.abspath(font))


def read_srt(path: str) -> list[tuple[float, float, str]]:
    with open(path, encoding="utf-8-sig") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    cues = []
    for block in blocks:
        lines = block.splitlines()
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        m = re.match(
            r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[timing]
        )
        if not m:
            continue
        nums = [int(x) for x in m.groups()]
        start = nums[0] * 3600 + nums[1] * 60 + nums[2] + nums[3] / 1000
        end = nums[4] * 3600 + nums[5] * 60 + nums[6] + nums[7] / 1000
        text = "\n".join(lines[timing + 1:]).strip()
        if text and end > start:
            cues.append((start, end, text))
    return cues


def ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def srt_to_ass(srt: str, ass: str, family: str, font_size: int, margin_v: int,
               outline: int, width: int, height: int) -> None:
    """转成显式 PlayRes 的 ASS，字号/边距直接按像素算，避免 libass 默认 384x288 导致错位。"""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{family},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{outline},0,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for start, end, text in read_srt(srt):
        text = text.replace("{", "").replace("}", "").replace("\n", r"\N")
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    with open(ass, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")


def burn(ffmpeg: str, venc: list[str], video: str, srt: str, font: str,
         font_size: int, margin_v: int, outline: int, output: str) -> int:
    filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " subtitles " not in filters:
        raise SystemExit(f"{ffmpeg} 不支持 subtitles 滤镜（需要 libass）")

    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not width or not height:
        raise SystemExit(f"打不开视频: {video}")

    family, fontsdir = font_family(font)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    ass_path = os.path.splitext(output)[0] + ".ass"
    srt_to_ass(srt, ass_path, family, font_size, margin_v, outline, width, height)
    print(f"font={family} size={font_size} marginV={margin_v} -> {ass_path}")

    vf = f"subtitles=filename='{escape_filter_path(os.path.abspath(ass_path))}'"
    if fontsdir:
        vf += f":fontsdir='{escape_filter_path(fontsdir)}'"

    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-vf", vf, *venc, "-pix_fmt", "yuv420p",
         "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
         "-movflags", "+faststart", output],
        check=True,
    )

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out_frames = subprocess.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", output],
            capture_output=True, text=True,
        ).stdout.strip()
        if out_frames.isdigit() and total and int(out_frames) != total:
            raise SystemExit(f"烧录后帧数不对：输入 {total}，输出 {out_frames}")
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", default="data/xinluT/final_audio.wav")
    parser.add_argument("--text", default="data/xinluT/script.txt")
    parser.add_argument("--srt", default=None, help="默认与音频同名的 .srt")
    parser.add_argument("--video", default="data/xinluT/final_video.mp4")
    parser.add_argument("--font", default="assets/AaRiFuYiRiKeAiYuan-2.ttf",
                        help="字体文件或字体名（Family Name）")
    parser.add_argument("--font-size", type=int, default=96)
    parser.add_argument("--margin-v", type=int, default=204, help="字幕底边离画面底部的像素")
    parser.add_argument("--outline", type=int, default=3)
    parser.add_argument("--burn", action="store_true", help="把字幕烧录进视频")
    parser.add_argument("-o", "--output", default="data/xinluT/final_video_sub.mp4")
    parser.add_argument("--skip-align", action="store_true", help="跳过对齐，直接用已有 SRT")
    parser.add_argument("--model", default="fa-zh")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    srt = args.srt or os.path.splitext(args.audio)[0] + ".srt"

    if args.skip_align:
        if not os.path.isfile(srt):
            raise SystemExit(f"找不到已有字幕: {srt}")
        print(f"跳过对齐，使用 {srt}")
    else:
        align_args = ["--audio", args.audio, "--text", args.text, "-o", srt, "--model", args.model]
        if args.device:
            align_args += ["--device", args.device]
        align_main(align_args)

    if not args.burn:
        print(f"字幕文件: {srt}")
        return 0

    ffmpeg, venc = find_ffmpeg()
    print(f"烧录字幕 -> {args.output}")
    total = burn(ffmpeg, venc, args.video, srt, args.font, args.font_size,
                 args.margin_v, args.outline, args.output)
    print(f"完成: {args.output}（{total} 帧）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
