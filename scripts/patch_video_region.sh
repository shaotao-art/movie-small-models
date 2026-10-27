#!/usr/bin/env bash
# 用图片左下角的一块区域覆盖视频的对应位置。
# 默认：frame_500.png 的 2700x667 左下角 -> data/xinluT/gen_video.MP4 左下角
#
# 用法:
#   bash scripts/patch_video_region.sh
#   bash scripts/patch_video_region.sh <video> <image> <output> [crop_w] [crop_h]
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:-data/xinluT/gen_video.MP4}"
IMAGE="${2:-frame_500.png}"
OUTPUT="${3:-data/xinluT/gen_video_patched.MP4}"
CROP_W="${4:-2850}"
CROP_H="${5:-363}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }

# 优先找带 libx264 的 ffmpeg；系统 ffmpeg 没有时尝试 imageio-ffmpeg 自带的版本
FFMPEG="$(command -v ffmpeg)"
ENCODERS="$("$FFMPEG" -hide_banner -encoders 2>/dev/null || true)"
if ! grep -q 'libx264' <<< "$ENCODERS"; then
  PYFF="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [ -n "$PYFF" ] && [ -x "$PYFF" ]; then
    PYENC="$("$PYFF" -hide_banner -encoders 2>/dev/null || true)"
    if grep -q 'libx264' <<< "$PYENC"; then
      FFMPEG="$PYFF"
      ENCODERS="$PYENC"
    fi
  fi
fi

if grep -q 'libx264' <<< "$ENCODERS"; then
  VENC=(-c:v libx264 -crf 18 -preset veryfast)
else
  VENC=(-c:v mpeg4 -q:v 2)  # 最后的兜底：系统 ffmpeg 只有坏的 libopenh264 时用
fi
echo "ffmpeg: $FFMPEG"
echo "encoder: ${VENC[*]}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 图片左下角区域
"$FFMPEG" -y -hide_banner -loglevel error -i "$IMAGE" \
  -vf "crop=${CROP_W}:${CROP_H}:0:ih-${CROP_H}" "$TMP/patch.png"

# 覆盖到视频左下角；音频直接 copy，-shortest 对齐时长
"$FFMPEG" -y -hide_banner -loglevel error -i "$VIDEO" -i "$TMP/patch.png" \
  -filter_complex "[0:v][1:v]overlay=0:H-h:format=auto,format=yuv420p[v]" \
  -map "[v]" -map 0:a? -c:a copy "${VENC[@]}" \
  -shortest -movflags +faststart "$OUTPUT"

echo "done: $OUTPUT"
