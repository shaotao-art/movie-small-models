#!/usr/bin/env python
"""视频专用评测：DataLoader 后台预取解码，GPU 推理不等 CPU 解码。

IterableDataset + DataLoader(num_workers=1, prefetch_factor=2)：
推理第 N 批时，worker 进程已经在解码第 N+1 批。高分辨率长视频也只需顺序解码一遍。
（环境里没有 decord，解码用 OpenCV；接口与 decord 版一致。）

用法:
    python eval_video.py --ckpt outputs/resnet18/best.pth --video movie.mp4
    python eval_video.py --ckpt outputs/efficientnet_b0/best_weights.pth \
        --config configs/efficientnet_b0.yaml --video movie.mp4 --interval 10 -o outputs/movie_eval.jsonl
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from data import IMAGENET_MEAN, IMAGENET_STD
from eval import load_model

logger = logging.getLogger("eval_video")


class VideoBatchDataset(IterableDataset):
    """在 worker 进程里解码并切好 batch。

    默认按 interval 抽帧；传 indices 时只取这些帧（infer_video 用来只解码预测有字幕的帧）。
    """

    def __init__(self, video, batch_size=64, interval=60.0, indices=None):
        self.video = video
        self.batch_size = batch_size
        self.interval = interval
        self.indices = indices

    def __iter__(self):
        cap = cv2.VideoCapture(self.video)  # 在 worker 内初始化，避免多进程冲突
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps * self.interval))
        wanted = set(self.indices) if self.indices is not None else None

        frames, indices, idx = [], [], 0
        try:
            while cap.grab():  # grab 只解封装不转码，跳过的帧很便宜
                take = idx in wanted if wanted is not None else idx % step == 0
                if take:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    frames.append(frame[:, :, ::-1].copy())  # BGR -> RGB
                    indices.append(idx)
                    if len(frames) == self.batch_size:
                        yield self._batch(frames, indices, fps)
                        frames, indices = [], []
                idx += 1
        finally:
            cap.release()
        if frames:
            yield self._batch(frames, indices, fps)

    @staticmethod
    def _batch(frames, indices, fps):
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # B,C,H,W uint8
        return {
            "images": images,
            "indices": torch.tensor(indices),
            "timestamps": torch.tensor(indices, dtype=torch.float32) / fps,
        }


def preprocess(frames, image_size, device):
    """GPU 上做 Resize(int(size*1.14)) + CenterCrop + Normalize，等价于 data.py 的 val 变换。"""
    x = frames.to(device, non_blocking=True).float().div_(255.0)
    h, w = x.shape[-2:]
    scale = int(image_size * 1.14) / min(h, w)
    x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bilinear", antialias=True)
    top = (x.shape[-2] - image_size) // 2
    left = (x.shape[-1] - image_size) // 2
    x = x[:, :, top:top + image_size, left:left + image_size]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)
    return (x - mean) / std


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="best.pth / last.pth，或导出的 *_weights.pth")
    parser.add_argument("--config", default=None, help="不传则用 ckpt 里保存的配置；纯权重文件必须传")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--classes", default=None, help="逗号分隔的类别顺序，默认从训练集目录推断")
    parser.add_argument("-o", "--output", default="outputs/eval_video.jsonl")
    parser.add_argument("--interval", type=float, default=60.0, help="抽帧间隔（秒），默认 60")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1, help="DataLoader 解码进程数，默认 1")
    parser.add_argument("--prefetch", type=int, default=2, help="每个 worker 预取批数，默认 2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)

    try:
        model, cfg, classes = load_model(args.ckpt, args.config, args.classes)
    except ValueError as e:
        parser.error(str(e))
    model.to(device).eval()
    image_size = cfg["data"]["image_size"]

    dataset = VideoBatchDataset(args.video, args.batch_size, args.interval)
    loader = DataLoader(
        dataset, batch_size=None, num_workers=args.workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch if args.workers > 0 else None,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    amp = device.type == "cuda"
    video = os.path.abspath(args.video)
    n = n_sub = 0
    started = time.time()
    with open(args.output, "w") as f, torch.inference_mode():
        for batch in loader:
            x = preprocess(batch["images"], image_size, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                probs = model(x).softmax(1).float().cpu().tolist()
            for idx, ts, prob in zip(batch["indices"].tolist(), batch["timestamps"].tolist(), probs):
                label = classes[prob.index(max(prob))]
                n_sub += label == "has_subtitle"
                f.write(json.dumps({
                    "video": video,
                    "frame_index": idx,
                    "timestamp": round(ts, 3),
                    "result": {"has_subtitle": label == "has_subtitle", "label": label, "prob": max(prob)},
                    "scores": dict(zip(classes, prob)),
                }, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            logger.info("frames=%d, %.1f frames/s", n, n / (time.time() - started))

    logger.info("%d results (%d has_subtitle) in %.1fs -> %s",
                n, n_sub, time.time() - started, args.output)


if __name__ == "__main__":
    main()
