#!/usr/bin/env python
"""用 FunASR fa-zh 对一段语音和对应文本做强制对齐，输出 SRT。

文本每行输出一条字幕，时间取该行首字符到末字符的对齐结果。

用法（默认就是 data/xinluT 下那对文件）:
    python align_srt.py
    python align_srt.py --audio data/xinluT/gen_sound_with_bg.mp3 \
        --text data/xinluT/script.txt -o data/xinluT/gen_sound_with_bg.srt
"""

import argparse
import os
import shutil
import subprocess
import tempfile

from common_utils.file_utils import load_text, save_text
from common_utils.time_utils import second_to_timecode


def remove_characters(text: str) -> str:
    """移除标点并规范化空白，得到用于强制对齐的纯文本。"""
    text = (
        text.replace("--", "，")
        .replace("——", "，")
        .replace("...", "，")
        .replace("……", "，")
        .replace("…", "，")
    )
    text = text.strip()
    for ch in "，。`·！？：；、,!?:;":
        text = text.replace(ch, " ")
    for ch in "“”‘’《》\"'":
        text = text.replace(ch, "")
    return " ".join(seg for seg in text.split() if seg.strip())


def to_wav(audio_path: str) -> str:
    """转成 16k 单声道 wav（FunASR 更稳）；没有 ffmpeg 时原样返回。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return audio_path
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        check=True,
    )
    return wav


def build_srt(lines: list[str], cleaned: list[str], timestamps: list) -> str:
    """把每行文本映射到 timestamp（按非空格字符数）并拼成 SRT。"""
    total_chars = sum(len(c.replace(" ", "")) for c in cleaned)
    if total_chars == 0 or not timestamps:
        return ""

    def time_at(pos: int):
        # 正常情况 timestamp 和字符一一对应；数量不一致时按比例映射
        idx = pos if len(timestamps) >= total_chars else int(pos * len(timestamps) / total_chars)
        return timestamps[min(max(idx, 0), len(timestamps) - 1)]

    parts, srt_index, pos = [], 1, 0
    for raw, clean in zip(lines, cleaned):
        n = len(clean.replace(" ", ""))
        if n == 0:
            continue
        start = time_at(pos)[0] / 1000.0
        end = time_at(pos + n - 1)[1] / 1000.0
        end = max(end, start + 0.05)
        parts += [str(srt_index), f"{second_to_timecode(start)} --> {second_to_timecode(end)}", raw, ""]
        srt_index += 1
        pos += n
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", default="data/xinluT/gen_sound_with_bg.mp3")
    parser.add_argument("--text", default="data/xinluT/script.txt")
    parser.add_argument("-o", "--output", default=None, help="默认与音频同名的 .srt")
    parser.add_argument("--model", default="fa-zh", help="FunASR 对齐模型")
    parser.add_argument("--device", default=None, help="cpu / cuda:0，默认由 FunASR 决定")
    args = parser.parse_args(argv)

    output = args.output or os.path.splitext(args.audio)[0] + ".srt"
    lines = [ln.strip() for ln in load_text(args.text).splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"文本为空: {args.text}")
    cleaned = [remove_characters(ln) for ln in lines]
    joined = " ".join(cleaned)

    try:
        from funasr import AutoModel
    except ImportError as e:
        raise SystemExit("需要先安装 funasr：pip install funasr") from e

    kwargs = {"model": args.model, "disable_update": True}
    if args.device:
        kwargs["device"] = args.device
    model = AutoModel(**kwargs)

    wav = to_wav(args.audio)
    try:
        result = model.generate(input=(wav, joined), data_type=("sound", "text"))[0]
    finally:
        if wav != args.audio and os.path.exists(wav):
            os.remove(wav)

    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise SystemExit("对齐失败：模型没有返回 timestamp")
    asr_text = (result.get("text") or "").replace(" ", "")
    if joined.replace(" ", "") not in asr_text:
        print(f"警告：对齐返回文本与输入不一致（{len(asr_text)} vs {len(joined.replace(' ', ''))} 字符），时间戳可能不准")

    srt = build_srt(lines, cleaned, timestamps)
    save_text(srt, output)
    print(f"{len(cleaned)} 行文本 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
