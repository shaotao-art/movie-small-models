#!/usr/bin/env python
"""用部署好的 SFT VLM 对视频逐帧抽字幕：读取 subtitles/eval_video.py 的结果，只解码判为有字幕的帧。

解码在 DataLoader worker 里后台预取（复用 eval_video.VideoBatchDataset，不回跳视频），
主线程把帧并发送给 VLM。重跑会跳过已完成的 (video, frame_index)。

用法:
    python -m subtitles.infer_video outputs/movie_eval.jsonl --base-url http://127.0.0.1:8080
    python -m subtitles.infer_video outputs/movie_eval.jsonl -o outputs/movie_vlm.jsonl -w 16 --batch-size 16
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from torch.utils.data import DataLoader

from common_utils.img_ops import smart_resize_img
from config import (
    SFT_PROMPT, VLM_API_KEY, VLM_BASE_URL, VLM_BATCH_SIZE, VLM_IMG_FACTOR,
    VLM_MAX_PIXELS, VLM_OUTPUT_SUFFIX, VLM_RETRIES, VLM_TIMEOUT, VLM_WORKERS,
)
from data_making.srt_model.dist_use_qwen38 import SubtitleClient, resolve_model
from subtitles.eval_video import VideoBatchDataset

logger = logging.getLogger("infer_video")


def read_frames(path):
    """按视频分组 {video: {frame_index: record}}，只保留判为有字幕的帧。"""
    groups = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 上次中断留下的半行
            if not (rec.get("result") or {}).get("has_subtitle"):
                continue
            video, idx = rec.get("video"), rec.get("frame_index")
            if video is None or idx is None:
                raise ValueError(f"{path} 不是 subtitles/eval_video.py 的输出（记录缺少 video/frame_index）")
            groups.setdefault(video, {})[idx] = rec
    return groups


def read_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["video"], rec["frame_index"]))
            except (json.JSONDecodeError, KeyError):
                continue  # 上次中断留下的半行
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_jsonl", help="subtitles/eval_video.py 输出的 jsonl")
    parser.add_argument("-o", "--output", default=None, help="默认 <eval_jsonl 同名>_vlm.jsonl")
    parser.add_argument("--base-url", default=VLM_BASE_URL, help="已部署 VLM 的服务地址")
    parser.add_argument("--model", default=None, help="模型名，默认从 /v1/models 自动获取")
    parser.add_argument("--api-key", default=None, help="默认取 VLM_API_KEY；也可用环境变量 AIO_API_KEY")
    parser.add_argument("-w", "--workers", type=int, default=VLM_WORKERS, help="并发请求数")
    parser.add_argument("--batch-size", type=int, default=VLM_BATCH_SIZE, help="解码批大小（4K 帧别开太大）")
    parser.add_argument("--timeout", type=float, default=VLM_TIMEOUT, help="单请求超时秒数")
    parser.add_argument("--retries", type=int, default=VLM_RETRIES, help="失败重试次数")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = args.api_key or VLM_API_KEY
    output = args.output or os.path.splitext(args.eval_jsonl)[0] + VLM_OUTPUT_SUFFIX

    try:
        groups = read_frames(args.eval_jsonl)
    except ValueError as e:
        logger.error("%s", e)
        return 1
    done = read_done(output)
    groups = {v: {i: r for i, r in recs.items() if (v, i) not in done} for v, recs in groups.items()}
    groups = {v: recs for v, recs in groups.items() if recs}
    total = sum(len(recs) for recs in groups.values())
    logger.info("videos=%d todo=%d done=%d output=%s", len(groups), total, len(done), output)
    if not total:
        return 0

    try:
        model = args.model or resolve_model(args.base_url, api_key)
    except Exception as e:
        logger.error("cannot resolve model from %s: %s (pass --model or --api-key)", args.base_url, e)
        return 1
    logger.info("model=%s", model)

    client = SubtitleClient(args.base_url, model, api_key, args.timeout, args.retries, prompt=SFT_PROMPT)
    ok = failed = 0
    started = time.time()
    with open(output, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for video, recs in groups.items():
            logger.info("decoding %s (%d frames)", video, len(recs))
            loader = DataLoader(
                VideoBatchDataset(video, args.batch_size, indices=sorted(recs)),
                # 这里要全分辨率帧给 VLM，跨进程传 1080p 一批会吃满 /dev/shm；
                # 在主进程解码，瓶颈本来也是 VLM 请求
                batch_size=None, num_workers=0,
            )
            it = iter(loader)
            while True:
                try:
                    batch = next(it)  # worker 已提前解码好这一批
                except StopIteration:
                    break
                except Exception as e:
                    logger.error("decode failed for %s: %s", video, e)
                    break

                futures = []
                for pos, idx in enumerate(batch["indices"].tolist()):
                    # 按训练时的像素预算缩放（对齐 32 的倍数、不裁切），再交给 client；
                    # 传 max_side=None 让 encode_pil 别再按 896 长边二次缩小
                    image = batch["images"][pos].permute(1, 2, 0).contiguous().numpy()  # RGB
                    image = smart_resize_img(image, VLM_MAX_PIXELS, factor=VLM_IMG_FACTOR)
                    futures.append((recs[idx], pool.submit(
                        client.predict, Image.fromarray(image), f"{video}:{idx}", max_side=None)))
                for rec, future in futures:
                    try:
                        res = future.result()
                    except Exception as e:
                        failed += 1
                        logger.error("failed %s:%s: %s", video, rec["frame_index"], e)
                        continue
                    fout.write(json.dumps({
                        "video": video,
                        "frame_index": rec["frame_index"],
                        "timestamp": rec.get("timestamp"),
                        "result": res["result"],
                        "raw": res["raw"],
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                    ok += 1
                    if ok % 50 == 0:
                        logger.info("progress %d/%d, %.1f frame/s", ok, total, ok / (time.time() - started))

    logger.info("finish: success=%d failed=%d elapsed=%.1fs", ok, failed, time.time() - started)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
