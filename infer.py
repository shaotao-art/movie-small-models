#!/usr/bin/env python
"""用部署好的 SFT VLM 对 eval.py 的图片结果逐张提取字幕。

输入是 eval.py 输出的 jsonl，只处理判为有字幕的图片。重跑会跳过已完成的图片（断点续跑）。
视频请用 infer_video.py。

用法:
    python infer.py outputs/eval_results.jsonl --base-url http://127.0.0.1:8080
    python infer.py outputs/eval_results.jsonl -o outputs/eval_vlm.jsonl -w 16
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    SFT_PROMPT, VLM_API_KEY, VLM_BASE_URL, VLM_OUTPUT_SUFFIX,
    VLM_RETRIES, VLM_TIMEOUT, VLM_WORKERS,
)
from data_making.build_cls_dataset import resolve_image
from data_making.dist_use_qwen38 import SubtitleClient, read_done, resolve_model

logger = logging.getLogger("infer")


def read_frames(path):
    """eval.py 输出里判为有字幕的图片。"""
    frames = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("result") or {}).get("has_subtitle"):
                frames.append(rec)
    return frames


def resolve(path):
    real = resolve_image(path)
    if real is None:
        raise FileNotFoundError(path)
    return real


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_jsonl", help="eval.py 输出的 jsonl（图片输入）")
    parser.add_argument("-o", "--output", default=None, help="默认 <eval_jsonl 同名>_vlm.jsonl")
    parser.add_argument("--base-url", default=VLM_BASE_URL, help="已部署 VLM 的服务地址")
    parser.add_argument("--model", default=None, help="模型名，默认从 /v1/models 自动获取")
    parser.add_argument("--api-key", default=None, help="默认取 VLM_API_KEY；也可用环境变量 AIO_API_KEY")
    parser.add_argument("-w", "--workers", type=int, default=VLM_WORKERS, help="并发线程数")
    parser.add_argument("--timeout", type=float, default=VLM_TIMEOUT, help="单请求超时秒数")
    parser.add_argument("--retries", type=int, default=VLM_RETRIES, help="失败重试次数")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = args.api_key or VLM_API_KEY
    output = args.output or os.path.splitext(args.eval_jsonl)[0] + VLM_OUTPUT_SUFFIX

    frames = read_frames(args.eval_jsonl)
    done = read_done(output)
    todo = [r for r in frames if r["image"] not in done]
    logger.info("has_subtitle=%d done=%d todo=%d output=%s", len(frames), len(done), len(todo), output)
    if not todo:
        return 0

    try:
        model = args.model or resolve_model(args.base_url, api_key)
    except Exception as e:
        logger.error("cannot resolve model from %s: %s (pass --model or --api-key)", args.base_url, e)
        return 1
    logger.info("model=%s", model)

    client = SubtitleClient(args.base_url, model, api_key, args.timeout, args.retries, prompt=SFT_PROMPT)

    def work(rec):
        return rec, client(resolve(rec["image"]))

    ok = failed = 0
    started = time.time()
    with open(output, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, rec) for rec in todo]
        for future in as_completed(futures):
            try:
                src, res = future.result()
            except Exception as e:
                failed += 1
                logger.error("failed: %s", e)
                continue
            fout.write(json.dumps({
                "image": src["image"],
                "result": res["result"],
                "raw": res["raw"],
            }, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            if ok % 50 == 0:
                logger.info("progress %d/%d, %.1f img/s", ok, len(todo), ok / (time.time() - started))

    logger.info("finish: success=%d failed=%d elapsed=%.1fs", ok, failed, time.time() - started)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
