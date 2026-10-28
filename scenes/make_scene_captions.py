#!/usr/bin/env python
"""无台词场景：切分 → 人脸识别标注 → 1fps 采样 → qwen35 caption。

1. 用 SRT 找出无台词片段（相邻字幕间隔 > NO_LINE_TIME_THRES）；
2. 按 tlou_scenes.pkl 里的镜头边界切分；
3. 每个片段按 1fps 抽帧，做人脸检测 + 参考图识别，把框和名字画在帧上，
   未识别的脸会继承同一 scene 上一帧 IoU > 0.6 的人名；写出 annotated.mp4；
4. 这些标注帧就是 1fps 采样结果，发给 qwen35 生成画面描述。

用法:
    python -m scenes.make_scene_captions --limit 3   # 先跑前 3 个片段
    python -m scenes.make_scene_captions             # 全部

输出:
    outputs/scene_captions/<uid>/annotated.mp4
    outputs/scene_captions/<uid>/frame_0001.jpg ...
    outputs/scene_captions/<uid>/caption.json
    outputs/scene_captions/captions.jsonl
"""

import argparse
import base64
import glob
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import (
    CAPTION_API_KEY, CAPTION_BASE_URL, CAPTION_FONT, CAPTION_FONT_SIZE,
    CAPTION_MAX_FRAMES, CAPTION_MAX_TOKENS, CAPTION_MODEL, CAPTION_PROMPT,
    CAPTION_SAMPLE_FPS, CAPTION_WORKERS, CHARACTER_REFS, FACE_MATCH_THRESH,
    SCENE_CAPTION_DIR, SCENES_PKL, SRT_PATH, VIDEO_PATH,
)
from faces.face_recognizer import (
    build_gallery, detect_faces, embed_faces, ensure_models, load_models, match_feature,
    propagate_labels_bidirectional,
)
from scenes.scene_det import load_scene_det_res
from subtitles.srt_preprocessor import SrtProcesser

logger = logging.getLogger("scene_caption")


def build_clips() -> list[dict]:
    """无台词片段按 scene pkl 的镜头边界切分。"""
    srt = SrtProcesser(SRT_PATH)
    srt.get_no_line_scenes()
    srt.get_splited_scenes(load_scene_det_res(SCENES_PKL))
    clips = []
    for line in srt.lines:
        if line.clip_type != "scene":
            continue
        for i, (start, end) in enumerate(line.splited_scene):
            if end - start < 0.1:
                continue
            clips.append({
                "uid": f"{line.uid}_sub_{i:04d}",
                "scene_uid": line.uid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            })
    clips.sort(key=lambda c: c["start"])
    return clips


def sample_times(clip: dict) -> list[float]:
    step = 1.0 / CAPTION_SAMPLE_FPS
    times, t = [], clip["start"]
    while t < clip["end"] - 1e-6:
        times.append(round(t, 3))
        t += step
    return times or [clip["start"]]


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def match_faces(frame: np.ndarray, detector, recognizer, gallery) -> list[dict]:
    faces = detect_faces(detector, frame)
    dets = []
    for face, (_, feat) in zip(faces, embed_faces(recognizer, frame, faces)):
        name, score = match_feature(recognizer, np.asarray(feat), gallery)
        known = name is not None and score >= FACE_MATCH_THRESH
        dets.append({"bbox": face[:4], "name": name if known else None,
                     "score": float(score) if known else 0.0})
    return dets


def draw_faces(frame: np.ndarray, dets: list[dict], font) -> np.ndarray:
    if not dets:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for det in dets:
        name, score = det["name"], det["score"]
        label = f"{name} {score:.2f}" if name else "未知"
        if det.get("propagated"):
            label += " *"
        color = (220, 40, 40) if name else (140, 140, 140)
        x, y, w, h = (int(round(v)) for v in det["bbox"])
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        ty = max(0, y - (bottom - top) - 6)
        draw.rectangle([x, ty, x + (right - left) + 8, ty + (bottom - top) + 6], fill=color)
        draw.text((x + 4, ty + 1), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frames_ready(clip_dir: str, expected: int) -> bool:
    """已标注帧数和视频帧数都对得上才算完成，避免上次中断留下半截结果。"""
    frames = glob.glob(os.path.join(clip_dir, "frame_*.jpg"))
    mp4 = os.path.join(clip_dir, "annotated.mp4")
    if len(frames) != expected or not os.path.exists(mp4) or os.path.getsize(mp4) == 0:
        return False
    cap = cv2.VideoCapture(mp4)
    ok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= expected
    cap.release()
    return ok


def _iter_targets(cap, targets, last, bar_total, desc):
    """顺序解码视频，只在目标帧 yield (idx, items, frame, bar)。"""
    idx = 0
    with tqdm(total=bar_total, desc=desc, unit="frame") as bar:
        while idx <= last and cap.grab():
            items = targets.get(idx)
            if items:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, items, frame, bar
            bar.update(1)
            idx += 1


def annotate(clips: list[dict], video: str, detector, recognizer, gallery, font) -> int:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    targets = {}
    for clip in clips:
        if not clip["needs_annotation"]:
            continue
        # 清掉上次中断留下的半截帧/视频
        for name in os.listdir(clip["dir"]):
            if name == "annotated.mp4" or (name.startswith("frame_") and name.endswith(".jpg")):
                os.remove(os.path.join(clip["dir"], name))
        for k, t in enumerate(clip["samples"]):
            targets.setdefault(int(round(t * fps)), []).append((clip, k))
    if not targets:
        cap.release()
        return 0

    last = max(targets)
    total_samples = sum(len(v) for v in targets.values())
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar_total = min(last + 1, total_frames) if total_frames > 0 else last + 1

    # 第一遍：检测 + 识别，先不画
    dets_by_clip = {}
    for _, items, frame, _ in _iter_targets(cap, targets, last, bar_total, "detect"):
        dets = match_faces(frame, detector, recognizer, gallery)
        for clip, k in items:
            dets_by_clip.setdefault(clip["uid"], {})[k] = dets
    cap.release()

    # 每个 scene 内双向扩展人名（中间帧识别到，前后都能继承）
    propagated = 0
    for by_k in dets_by_clip.values():
        seq = [by_k[k] for k in sorted(by_k)]
        propagate_labels_bidirectional(seq)
        propagated += sum(bool(d.get("propagated")) for dets in seq for d in dets)

    # 第二遍：画框 + 写帧/视频
    cap = cv2.VideoCapture(video)
    writer = None
    writer_uid = None
    done = 0
    for idx, items, frame, bar in _iter_targets(cap, targets, last, bar_total, "annotate"):
        for clip, k in items:
            dets = dets_by_clip.get(clip["uid"], {}).get(k, [])
            annotated = draw_faces(frame, dets, font)
            cv2.imwrite(os.path.join(clip["dir"], f"frame_{k + 1:03d}.jpg"), annotated)
            if clip["uid"] != writer_uid:  # 片段按时间顺序处理，只保留一个 writer
                if writer is not None:
                    writer.release()
                writer = cv2.VideoWriter(
                    os.path.join(clip["dir"], "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), CAPTION_SAMPLE_FPS, (width, height),
                )
                writer_uid = clip["uid"]
            writer.write(annotated)
            done += 1
            bar.set_postfix_str(
                f"t={fmt_time(idx / fps)} samples={done}/{total_samples} {clip['uid']}"
            )
    cap.release()
    if writer is not None:
        writer.release()
    return propagated


def load_prompt() -> str:
    with open(CAPTION_PROMPT) as f:
        return f.read().replace("{characters}", "、".join(CHARACTER_REFS)).strip()


def parse_caption(text: str) -> dict:
    """解析模型输出的 JSON；兼容代码围栏/夹带说明，解析失败就整段当描述。"""
    raw = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
    data = None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if isinstance(data, dict) and isinstance(data.get("description"), str):
        characters = [str(c).strip() for c in (data.get("characters") or []) if str(c).strip()]
        return {"caption": data["description"].strip(), "characters": characters}
    return {"caption": raw, "characters": []}


def caption_clip(clip: dict, prompt: str, client: OpenAI) -> dict:
    frames = sorted(glob.glob(os.path.join(clip["dir"], "frame_*.jpg")))
    if not frames:
        raise RuntimeError(f"{clip['uid']}: 没有标注帧，先跑标注")
    if len(frames) > CAPTION_MAX_FRAMES:
        idxs = sorted(set(np.linspace(0, len(frames) - 1, CAPTION_MAX_FRAMES).round().astype(int)))
        frames = [frames[i] for i in idxs]
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in frames]
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=CAPTION_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=CAPTION_MAX_TOKENS,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_caption(resp.choices[0].message.content)
        except Exception as e:  # 网络错误/超时/5xx 都重试
            last_error = e
            logger.warning("attempt %d/3 failed for %s: %s", attempt, clip["uid"], e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{clip['uid']}: {last_error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个片段，0 为全部")
    parser.add_argument("--uid", default=None, help="只跑指定片段 uid")
    parser.add_argument("-o", "--output-dir", default=SCENE_CAPTION_DIR)
    parser.add_argument("--workers", type=int, default=CAPTION_WORKERS)
    parser.add_argument("--no-caption", action="store_true", help="只做标注，不请求 VLM")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.path.isfile(CAPTION_FONT):
        parser.error(f"找不到字体文件: {CAPTION_FONT}")
    font = ImageFont.truetype(CAPTION_FONT, CAPTION_FONT_SIZE)
    started = time.time()

    clips = build_clips()
    if args.uid:
        clips = [c for c in clips if c["uid"] == args.uid]
        if not clips:
            parser.error(f"没有片段 {args.uid}")
    elif args.limit:
        clips = clips[: args.limit]
    logger.info("[1/3] 切分后 %d 个片段，总时长 %.1fs", len(clips), sum(c["duration"] for c in clips))
    os.makedirs(args.output_dir, exist_ok=True)
    for clip in clips:
        clip["dir"] = os.path.join(args.output_dir, clip["uid"])
        os.makedirs(clip["dir"], exist_ok=True)
        clip["samples"] = sample_times(clip)
        caption_path = os.path.join(clip["dir"], "caption.json")
        clip["caption"] = None
        clip["characters"] = []
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                data = json.load(f)
            clip["caption"] = data.get("caption")
            clip["characters"] = data.get("characters", [])
        clip["needs_annotation"] = not clip["caption"] and not frames_ready(clip["dir"], len(clip["samples"]))

    n_annot = sum(c["needs_annotation"] for c in clips)
    n_samples = sum(len(c["samples"]) for c in clips if c["needs_annotation"])
    n_caption = sum(not c["caption"] for c in clips)
    logger.info("[2/3] 需标注 %d 个片段（%d 帧），需 caption %d 个", n_annot, n_samples, n_caption)

    if n_annot:
        detector_path, recognizer_path = ensure_models()
        detector, recognizer = load_models(detector_path, recognizer_path)
        gallery = build_gallery(CHARACTER_REFS, detector, recognizer)
        t0 = time.time()
        propagated = annotate(clips, args.video, detector, recognizer, gallery, font)
        logger.info("[2/3] 标注完成，用时 %.0fs（跟踪扩展 %d 张脸）", time.time() - t0, propagated)

    if not args.no_caption:
        todo = [c for c in clips if not c["caption"]]
        logger.info("[3/3] 请求 caption：%d 个片段，%d 并发", len(todo), args.workers)
        prompt = load_prompt()
        client = OpenAI(api_key=CAPTION_API_KEY, base_url=CAPTION_BASE_URL.rstrip("/") + "/v1", timeout=300)
        ok = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(caption_clip, c, prompt, client): c for c in todo}
            with tqdm(total=len(futures), desc="caption", unit="clip") as bar:
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        result = future.result()
                        clip["caption"] = result["caption"]
                        clip["characters"] = result["characters"]
                    except Exception as e:
                        failed += 1
                        logger.error("caption failed: %s", e)
                    else:
                        ok += 1
                        with open(os.path.join(clip["dir"], "caption.json"), "w") as f:
                            json.dump({
                                "uid": clip["uid"],
                                "scene_uid": clip["scene_uid"],
                                "start": clip["start"],
                                "end": clip["end"],
                                "duration": clip["duration"],
                                "characters": clip["characters"],
                                "caption": clip["caption"],
                            }, f, ensure_ascii=False, indent=2)
                    bar.set_postfix_str(f"ok={ok} fail={failed}")
                    bar.update(1)
                    if (ok + failed) % 25 == 0:
                        logger.info("[3/3] caption %d/%d（ok=%d fail=%d）%.0fs",
                                    ok + failed, len(futures), ok, failed, time.time() - t0)
        logger.info("[3/3] caption 完成，用时 %.0fs（ok=%d fail=%d）", time.time() - t0, ok, failed)
    else:
        logger.info("[3/3] 跳过 caption")

    out = os.path.join(args.output_dir, "captions.jsonl")
    records = {}
    if os.path.exists(out):  # 保留其它片段的已有结果，--uid/--limit 局部重跑不覆盖全量
        with open(out) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records[rec["uid"]] = rec
    for clip in clips:
        new = {
            "uid": clip["uid"],
            "scene_uid": clip["scene_uid"],
            "start": clip["start"],
            "end": clip["end"],
            "duration": clip["duration"],
            "characters": clip["characters"],
            "caption": clip["caption"],
        }
        if new["caption"] or clip["uid"] not in records:
            records[clip["uid"]] = new
    with open(out, "w") as f:
        for rec in sorted(records.values(), key=lambda r: r.get("start", 0)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done = sum(bool(c["caption"]) for c in clips)
    print(f"{done}/{len(clips)} 个片段有 caption -> {out}（总用时 {time.time() - started:.0f}s）")
    if args.no_caption:
        return 0
    return 0 if done == len(clips) else 1


if __name__ == "__main__":
    raise SystemExit(main())
