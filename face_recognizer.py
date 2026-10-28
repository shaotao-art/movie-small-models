"""用人脸检测+识别判断每个片段里出现了哪些人物。

检测用 YuNet、识别用 SFace，两者都在已安装的 opencv-python 里
（cv2.FaceDetectorYN / cv2.FaceRecognizerSF），不需要额外依赖；只缺两个 onnx
模型文件，首次运行会自动从 opencv_zoo 下载到 config.FACE_MODEL_DIR。

参考图见 config.CHARACTER_REFS（一个角色可以挂多张图）。结果写成 json，
key 与其它流水线的片段 uid 一致（整段场景为 scene_0000），并把对齐后的人脸
裁剪图存到 config.FACE_CROPS_DIR，方便人工核对判定对错。

用法:
    python face_recognizer.py --check          # 只检查参考图能否建出 gallery
    python face_recognizer.py --limit 5        # 先跑 5 个片段看看
    python face_recognizer.py --workers 4      # 跑 config.SCENES_DIR 下所有整段场景
"""
import argparse
import logging
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

from common_utils.file_utils import save_image, save_json
import config

logging.basicConfig(level=logging.INFO)

# 每线程一套模型实例，见 load_models
_thread_local = threading.local()


def list_video_clips(video_root: str, include_sub: bool = False) -> list[str]:
    """列出片段目录下的 mp4；默认跳过 cut_videos 产出的 _sub_ 子镜头。"""
    clips = sorted(f for f in os.listdir(video_root) if f.endswith(".mp4"))
    return clips if include_sub else [f for f in clips if "_sub_" not in f]


def ensure_models(model_dir: str = config.FACE_MODEL_DIR) -> tuple[str, str]:
    """确保两个 onnx 模型都在本地，缺了就下载，返回 (检测模型, 识别模型) 路径"""
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for key, url in config.FACE_MODEL_URLS.items():
        path = os.path.join(model_dir, url.rsplit("/", 1)[-1])
        if not os.path.exists(path):
            logging.info(f"[face_recognizer] 下载模型 {path}")
            # 先下成 .part，半途中断不会留下一个看起来可用的坏文件
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        paths[key] = path
    return paths["detector"], paths["recognizer"]


def load_models(detector_path: str, recognizer_path: str) -> tuple:
    """取当前线程的 (detector, recognizer)，没有就建一套

    detector.setInputSize 是实例上的可变状态，多个线程共用一个实例会互相覆盖，
    所以按线程各持一套。SFace 每套约 39MB，几个线程的量级可以接受。
    """
    models = getattr(_thread_local, "models", None)
    if models is None:
        detector = cv2.FaceDetectorYN.create(
            detector_path, "", (320, 320), config.FACE_DETECT_SCORE
        )
        recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        models = (detector, recognizer)
        _thread_local.models = models
    return models


def detect_faces(detector, image_bgr: np.ndarray) -> list[np.ndarray]:
    """检测一帧里的脸，返回 [15 元素的行]：bbox 4 + 五官关键点 10 + score 1"""
    height, width = image_bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    return [f for f in faces if min(f[2], f[3]) >= config.FACE_MIN_FACE_PX]


def embed_faces(recognizer, image_bgr: np.ndarray, faces: list) -> list[tuple]:
    """把检测到的脸对齐后提特征，返回 [(112x112 对齐人脸图, 128 维特征), ...]"""
    out = []
    for face in faces:
        aligned = recognizer.alignCrop(image_bgr, face)
        out.append((aligned, recognizer.feature(aligned)))
    return out


def _ref_paths(value) -> list[str]:
    """config.CHARACTER_REFS 的值允许是单个路径或路径列表"""
    return [value] if isinstance(value, str) else list(value)


def match_feature(recognizer, feature: np.ndarray, gallery: dict) -> tuple[str, float]:
    """取与 gallery 里各角色最像的那个，返回 (角色名, 分数)"""
    best_name, best_score = None, 0.0
    for name, features in gallery.items():
        score = max(
            recognizer.match(feature, ref, cv2.FaceRecognizerSF_FR_COSINE)
            for ref in features
        )
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def box_iou(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def propagate_labels(prev: list[dict], cur: list[dict],
                     iou_thresh: float = config.FACE_TRACK_IOU) -> list[dict]:
    """同一场景相邻帧：未识别的人脸与上一帧已识别框 IoU 超过阈值时，继承上一帧的人名。

    prev/cur 的元素：{"bbox": (x, y, w, h), "name": str|None, "score": float}
    """
    for det in cur:
        if det["name"]:
            continue
        best, best_iou = None, iou_thresh
        for old in prev:
            if not old["name"]:
                continue
            iou = box_iou(det["bbox"], old["bbox"])
            if iou > best_iou:
                best, best_iou = old, iou
        if best is not None:
            det["name"] = best["name"]
            det["score"] = best["score"]
            det["propagated"] = True
    return cur


def propagate_labels_bidirectional(frames_dets: list[list[dict]],
                                   iou_thresh: float = config.FACE_TRACK_IOU) -> list[list[dict]]:
    """在一个场景内先正向、再反向扩展人名：中间帧识别到的角色可以覆盖前后帧。"""
    for i in range(1, len(frames_dets)):
        propagate_labels(frames_dets[i - 1], frames_dets[i], iou_thresh)
    for i in range(len(frames_dets) - 2, -1, -1):
        propagate_labels(frames_dets[i + 1], frames_dets[i], iou_thresh)
    return frames_dets


def build_gallery(character_refs: dict, detector, recognizer) -> dict[str, np.ndarray]:
    """对每个角色的参考图取最大的那张脸提特征，返回 name -> (K, 128) 特征矩阵"""
    gallery = {}
    for name, value in character_refs.items():
        features = []
        for path in _ref_paths(value):
            # 人脸 API 要 BGR，这里不用 file_utils.read_image（那个转成 RGB 了）
            image = cv2.imread(path)
            if image is None:
                logging.warning(f"[face_recognizer] 参考图读不出来: {path}")
                continue
            faces = detect_faces(detector, image)
            if not faces:
                logging.warning(f"[face_recognizer] 参考图里没检测到脸: {path}")
                continue
            # 参考图里主角一般是画面中最大的那张脸
            face = max(faces, key=lambda f: f[2] * f[3])
            _, feature = embed_faces(recognizer, image, [face])[0]
            features.append(feature)
            logging.info(
                f"[face_recognizer] 参考图 {name} <- {path} "
                f"(面积 {face[2] * face[3]:.0f}px, 检测分 {face[14]:.2f})"
            )
        if not features:
            raise ValueError(
                f"[face_recognizer] 角色「{name}」没有任何可用的参考脸，"
                f"检查 config.CHARACTER_REFS 里的图"
            )
        gallery[name] = np.vstack(features)
    _check_gallery(recognizer, gallery)
    return gallery


def _check_gallery(recognizer, gallery: dict) -> None:
    """同一角色的多张参考图之间应该彼此像；不像就说明某张图不适合当参考"""
    for name, features in gallery.items():
        if len(features) < 2:
            continue
        scores = [
            recognizer.match(features[i], features[j], cv2.FaceRecognizerSF_FR_COSINE)
            for i in range(len(features))
            for j in range(i + 1, len(features))
        ]
        if min(scores) < config.FACE_MATCH_THRESH:
            logging.warning(
                f"[face_recognizer] {name} 的参考图之间最差相似度只有 "
                f"{min(scores):.3f}（阈值 {config.FACE_MATCH_THRESH}），可能有图不合格"
            )


def sample_frames(video_path: str, num: int) -> list[tuple[int, np.ndarray]]:
    """均匀抽 num 帧，返回 [(帧号, BGR 图), ...]

    num=1 时取中间那帧（首帧常是黑场或淡入）。
    """
    capture = cv2.VideoCapture(video_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logging.warning(f"[face_recognizer] 读不出帧数: {video_path}")
            return []
        if num == 1:
            idxs = [total // 2]
        else:
            step = max(total // num, 1)
            idxs = list(range(0, total, step))[:num]
        frames = []
        # ponytail: 用 POS_FRAMES 定位，长 GOP 可能落在邻近帧上；判断「有没有某人」
        # 不受影响，要精确帧号再换成逐帧解码
        for idx in idxs:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if ok:
                frames.append((idx, frame))
        return frames
    finally:
        capture.release()


def identify_clip(
    video_path: str,
    detector,
    recognizer,
    gallery: dict,
    uid: str,
    num_frames: int = config.FACE_FRAMES_PER_CLIP,
    match_thresh: float = config.FACE_MATCH_THRESH,
    min_hit_frames: int = config.FACE_MIN_HIT_FRAMES,
    track_iou: float = config.FACE_TRACK_IOU,
    crops_dir: str = None,
    frames_dir: str = None,
) -> dict:
    """判断一个片段里出现了哪些人物

    每帧每张脸与 gallery 比一次，取最像的角色；分数过阈值才算命中。
    未识别的脸会在同一片段（场景）内按前后相邻帧 IoU > track_iou 双向继承人名。
    角色命中帧数 >= min_hit_frames 才认为出现。
    """
    frames = sample_frames(video_path, num_frames)
    all_dets = []
    for _, image in frames:
        faces = detect_faces(detector, image)
        dets = []
        for face, (aligned, feature) in zip(
            faces, embed_faces(recognizer, image, faces)
        ):
            name, score = match_feature(recognizer, feature, gallery)
            if score < match_thresh:
                name = None
            dets.append({"bbox": face[:4], "name": name, "score": float(score),
                         "face": face, "aligned": aligned})
        all_dets.append(dets)
    propagate_labels_bidirectional(all_dets, track_iou)

    hits: dict[str, dict[int, float]] = {name: {} for name in gallery}
    unknown_faces = 0
    for (frame_idx, image), dets in zip(frames, all_dets):
        for det in dets:
            name, score = det["name"], det["score"]
            if crops_dir:
                tag = f"{name}_{score:.2f}" if name else f"未知_{score:.2f}"
                # save_image 要 RGB，aligned 是 BGR
                save_image(
                    cv2.cvtColor(det["aligned"], cv2.COLOR_BGR2RGB),
                    os.path.join(crops_dir, f"{uid}__{tag}_{frame_idx:04d}.jpg"),
                )
            if name is None:
                unknown_faces += 1
            else:
                hits[name][frame_idx] = max(hits[name].get(frame_idx, 0.0), score)
        if frames_dir:
            # 整帧存一份并画上人脸框+判定结果，方便人工核对
            drawn = image.copy()
            for det in dets:
                name, score = det["name"], det["score"]
                x, y, w, h = (int(v) for v in det["face"][:4])
                tag = f"{name} {score:.2f}" if name else f"未知 {score:.2f}"
                if det.get("propagated"):
                    tag += " *"
                cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    drawn, tag, (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
            save_image(
                cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
                os.path.join(frames_dir, f"{uid}_f{frame_idx:04d}.jpg"),
            )
    characters = [
        {
            "name": name,
            "score": round(max(frame_scores.values()), 3),
            "hit_frames": len(frame_scores),
        }
        for name, frame_scores in hits.items()
        if len(frame_scores) >= min_hit_frames
    ]
    characters.sort(key=lambda c: c["score"], reverse=True)
    return {
        "characters": characters,
        "unknown_faces": unknown_faces,
        "frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description="人脸检测+识别，判断片段里出现了哪些人物")
    parser.add_argument("--video-root", default=config.SCENES_DIR)
    parser.add_argument("--out", default=config.FACES_JSON)
    parser.add_argument("--crops-dir", default=config.FACE_CROPS_DIR)
    parser.add_argument("--no-crops", action="store_true", help="不存人脸裁剪图")
    parser.add_argument(
        "--dump-frames",
        default=None,
        help="把抽到的整帧存成图（画上人脸框与判定结果），传目录路径",
    )
    parser.add_argument("--frames", type=int, default=config.FACE_FRAMES_PER_CLIP)
    parser.add_argument("--threshold", type=float, default=config.FACE_MATCH_THRESH)
    parser.add_argument(
        "--min-hit-frames", type=int, default=config.FACE_MIN_HIT_FRAMES
    )
    parser.add_argument("--workers", type=int, default=config.FACE_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个片段，0 为全部")
    parser.add_argument(
        "--include-sub", action="store_true", help="连 _sub_000x 子镜头一起跑"
    )
    parser.add_argument("--check", action="store_true", help="只建 gallery 就退出")
    args = parser.parse_args()

    detector_path, recognizer_path = ensure_models()
    detector, recognizer = load_models(detector_path, recognizer_path)
    gallery = build_gallery(config.CHARACTER_REFS, detector, recognizer)
    logging.info(
        "[face_recognizer] gallery: "
        + ", ".join(f"{name} {len(features)}张" for name, features in gallery.items())
    )
    if args.check:
        return

    video_clips = list_video_clips(args.video_root, include_sub=args.include_sub)
    if args.limit:
        video_clips = video_clips[: args.limit]
    crops_dir = None if args.no_crops else args.crops_dir
    logging.info(
        f"[face_recognizer] 待处理片段 {len(video_clips)} 个，并发 {args.workers}，"
        f"抽帧 {args.frames}/片段"
    )

    def run(video_file: str) -> tuple[str, dict]:
        # 线程内首次调用时建一套模型
        detector, recognizer = load_models(detector_path, recognizer_path)
        uid = video_file[: -len(".mp4")]
        return uid, identify_clip(
            os.path.join(args.video_root, video_file),
            detector,
            recognizer,
            gallery,
            uid,
            num_frames=args.frames,
            match_thresh=args.threshold,
            min_hit_frames=args.min_hit_frames,
            crops_dir=crops_dir,
            frames_dir=args.dump_frames,
        )

    results = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, f): f for f in video_clips}
        with tqdm(total=len(futures), desc="faces", unit="clip") as bar:
            for future in as_completed(futures):
                video_file = futures[future]
                try:
                    uid, result = future.result()
                    results[uid] = {"video_file": video_file, **result}
                    found = "/".join(c["name"] for c in result["characters"]) or "无"
                except Exception as e:
                    failed += 1
                    found = "失败"
                    logging.warning(f"[face_recognizer] {video_file} 失败: {e}")
                bar.set_postfix_str(f"{video_file} [{found}] | 失败 {failed}")
                bar.update(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_json(results, args.out)
    named = sum(1 for r in results.values() if r["characters"])
    logging.info(
        f"[face_recognizer] 完成 {len(results)} 个（有名字的 {named} 个，失败 {failed} 个），"
        f"结果 {args.out}，人脸图 {crops_dir or '(未存)'}"
    )


if __name__ == "__main__":
    main()
