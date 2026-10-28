#!/usr/bin/env python
"""一键生成最终视频：

1. 把 seed/还原版.mp3 变速到 bg_music.mp3 的长度，再和背景音乐混合；
2. 按 analyze_376 的逐帧字幕文本，替换原始视频里的字幕区域，生成 gen_video_patched.MP4；
3. 把 patched 视频和新音频合并成 final_video.mp4（视频流直接 copy，不丢帧）。

用法:
    python -m render.make_final_video
    python -m render.make_final_video --speech data/xinluT/seed/还原版.mp3 \
        --output data/xinluT/final_video.mp4
"""

import argparse
import shutil
import subprocess

from render.patch_video_subtitles import find_ffmpeg, patch_video


def need(cmd: str) -> str:
    path = shutil.which(cmd)
    if path is None:
        raise SystemExit(f"找不到 {cmd}")
    return path


def probe_duration(ffprobe: str, path: str) -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def count_frames(ffprobe: str, path: str) -> int:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else -1


def atempo_filter(tempo: float) -> str:
    """atempo 单级只支持 0.5~2.0，超出时串联。"""
    factors = []
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    factors.append(tempo)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def mix_audio(ffmpeg: str, ffprobe: str, speech: str, music: str, music_volume: float,
              output: str) -> None:
    speech_dur = probe_duration(ffprobe, speech)
    music_dur = probe_duration(ffprobe, music)
    tempo = speech_dur / music_dur
    print(f"[1/3] 语音 {speech_dur:.3f}s -> 音乐 {music_dur:.3f}s，atempo={tempo:.6f}")
    filter_complex = (
        f"[0:a]{atempo_filter(tempo)},aresample=44100,aformat=channel_layouts=stereo[voice];"
        f"[1:a]volume={music_volume}[music];"
        f"[voice][music]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[out]"
    )
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", speech, "-i", music, "-filter_complex", filter_complex,
         "-map", "[out]", "-c:a", "pcm_s16le", output],
        check=True,
    )


def mux(ffmpeg: str, video: str, audio: str, output: str, duration: float) -> None:
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", f"{duration:.3f}", "-movflags", "+faststart", output],
        check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/xinluT/376_raw.MP4")
    parser.add_argument("--subtitles", default="outputs/analyze_376/376_raw_subtitles.jsonl")
    parser.add_argument("--clean-image", default="data/xinluT/cleaned_bg_3840x2160.png")
    parser.add_argument("--speech", default="data/xinluT/seed/还原版.mp3")
    parser.add_argument("--music", default="data/xinluT/bg_music.mp3")
    parser.add_argument("--music-volume", type=float, default=0.4)
    parser.add_argument("--audio-out", default="data/xinluT/final_audio.wav")
    parser.add_argument("--patched", default="data/xinluT/gen_video_patched.MP4")
    parser.add_argument("-o", "--output", default="data/xinluT/final_video.mp4")
    parser.add_argument("--similarity", type=float, default=0.8)
    args = parser.parse_args(argv)

    ffmpeg, venc = find_ffmpeg()
    ffprobe = need("ffprobe")

    mix_audio(ffmpeg, ffprobe, args.speech, args.music, args.music_volume, args.audio_out)

    print("[2/3] 替换原始视频的字幕区域...")
    frames, fps, counts = patch_video(
        ffmpeg, venc, args.video, args.subtitles, args.clean_image, args.patched, args.similarity,
    )
    patched_frames = count_frames(ffprobe, args.patched)
    if patched_frames != frames:
        raise SystemExit(f"patched 视频帧数不对：编码 {frames}，文件 {patched_frames}")

    print("[3/3] 合并视频和音频...")
    mux(ffmpeg, args.patched, args.audio_out, args.output, frames / fps)
    out_frames = count_frames(ffprobe, args.output)
    if out_frames != frames:
        raise SystemExit(f"最终视频帧数不对：期望 {frames}，实际 {out_frames}")

    print(f"完成：{args.output}（{out_frames} 帧，{frames / fps:.3f}s）")
    for name, n in counts.most_common():
        print(f"  {n:4d} 帧  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
