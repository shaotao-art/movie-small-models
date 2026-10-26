#!/usr/bin/env python
"""把 SRT 台词和无台词场景的 caption 合成一份视频信息 JSON。

结构参考 srt_preprocessor.SrtClip：每条 line/scene 片段都是 SrtClip 的字段，
scene 额外带上 characters（画面里的确认人物）和 sub_captions（每个镜头的 caption）。

用法:
    python build_video_info.py
    python build_video_info.py -c outputs/scene_captions/captions.jsonl -o outputs/video_info.json
"""

import argparse
import json
import os
from dataclasses import asdict

from config import SCENES_PKL, SCENE_CAPTION_DIR, SRT_PATH, VIDEO_INFO_JSON, VIDEO_PATH
from scene_det import load_scene_det_res
from srt_preprocessor import SrtProcesser


def load_captions(path: str) -> dict[str, dict]:
    captions = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    captions[rec["uid"]] = rec
    return captions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--captions", default=os.path.join(SCENE_CAPTION_DIR, "captions.jsonl"),
                        help="make_scene_captions.py 汇总出的 captions.jsonl")
    parser.add_argument("-o", "--output", default=VIDEO_INFO_JSON)
    args = parser.parse_args(argv)

    srt = SrtProcesser(SRT_PATH)
    srt.get_no_line_scenes()
    srt.get_splited_scenes(load_scene_det_res(SCENES_PKL))
    captions = load_captions(args.captions)

    clips = []
    total_subs = found_subs = 0
    for line in srt.lines:
        entry = asdict(line)
        entry["characters"] = []
        entry["sub_captions"] = []
        if line.clip_type == "scene":
            for i, (start, end) in enumerate(line.splited_scene):
                total_subs += 1
                uid = f"{line.uid}_sub_{i:04d}"
                rec = captions.get(uid)
                if not rec or not rec.get("caption"):
                    continue
                found_subs += 1
                entry["sub_captions"].append({
                    "uid": uid,
                    "start": rec.get("start", round(start, 3)),
                    "end": rec.get("end", round(end, 3)),
                    "caption": rec["caption"],
                    "characters": rec.get("characters", []),
                })
            entry["caption"] = " ".join(c["caption"] for c in entry["sub_captions"])
            entry["characters"] = list(dict.fromkeys(
                name for c in entry["sub_captions"] for name in c["characters"]
            ))
        clips.append(entry)

    data = {
        "video": os.path.abspath(VIDEO_PATH),
        "srt_path": SRT_PATH,
        "captions_path": args.captions,
        "total_duration": round(srt.total_duration, 3),
        "total_line_duration": round(srt.total_line_duration, 3),
        "total_no_line_duration": round(srt.total_no_line_duration, 3),
        "num_clips": len(clips),
        "num_lines": sum(c["clip_type"] == "line" for c in clips),
        "num_scenes": sum(c["clip_type"] == "scene" for c in clips),
        "clips": clips,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{data['num_lines']} 条台词 + {data['num_scenes']} 个无台词场景"
          f"（{found_subs}/{total_subs} 个子镜头有 caption）-> {args.output}")
    if found_subs < total_subs:
        print(f"注意：还有 {total_subs - found_subs} 个子镜头没有 caption")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
