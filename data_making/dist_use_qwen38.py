#!/usr/bin/env python
"""并发请求 Qwen-VL 提取电影截图字幕。

用法:
    python dist_use_qwen38.py image_list.txt
    python dist_use_qwen38.py image_list.txt -o results.jsonl -w 32
    python dist_use_qwen38.py image_list.txt --api-key sk-xxx

image_list: 每行一个图片路径，空行和 # 开头的行忽略。
输出为 jsonl，每行 {"image": ..., "result": {...}|null, "raw": "模型原始输出"}；
重跑会自动跳过已完成的图片（断点续跑）。
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

logger = logging.getLogger("qwen")

MAX_SIDE = 896


def encode_image(image_path, max_side=MAX_SIDE):
    """长边缩放到 max_side 后编码为 base64 JPEG，减少上传体积和视觉 token。"""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        scale = max_side / max(im.size)
        size = (round(im.width * scale), round(im.height * scale))
        im = im.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

prompt = """提取下面电影截图中的字幕

注意 字幕通常出现在页面的底部
对于字幕，例如是演出者、导演、编剧等信息，则不需要提取

# 输出格式
{"has_subtitle": true, "subtitle_cn": "字幕内容", "subtitle_en": "subtitle content"}

example:
{"has_subtitle": true, "subtitle_cn": "字幕内容", "subtitle_en": "subtitle content"}
{"has_subtitle": true, "subtitle_cn": "字幕内容", "subtitle_en": ""}
{"has_subtitle": true, "subtitle_cn": "", "subtitle_en": "subtitle content"}
{"has_subtitle": false, "subtitle_cn": "", "subtitle_en": ""}
"""


def parse_json(text):
    """模型可能带 ```json 围栏或多余说明，尽量把 JSON 抠出来。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


class SubtitleClient:
    def __init__(self, base_url, model, api_key=None, timeout=120, retries=3, prompt=prompt):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.prompt = prompt
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    def _session(self):
        # requests.Session 不是线程安全的，每个线程一个以复用连接
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def __call__(self, image_path):
        data_url = f"data:image/jpeg;base64,{encode_image(image_path)}"

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
        }

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._session().post(
                    self.url, json=payload, headers=self.headers, timeout=self.timeout
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content)
                parsed = parse_json(content)
                if parsed is None:
                    logger.warning("unparsable output for %s: %r", image_path, content[:200])
                return {"image": image_path, "result": parsed, "raw": content}
            except Exception as e:  # 网络错误/超时/5xx 都重试
                last_error = e
                if attempt < self.retries:
                    logger.warning("attempt %d/%d failed for %s: %s", attempt, self.retries, image_path, e)
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"{image_path}: {last_error}")


def resolve_model(base_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(base_url.rstrip("/") + "/v1/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def read_image_list(path):
    with open(path) as f:
        items = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    return list(dict.fromkeys(items))


def read_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError):
                continue  # 上次中断留下的半行
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image_list", help="图片路径列表文件")
    parser.add_argument("-o", "--output", default=None, help="输出 jsonl，默认与 image_list 同名的 .jsonl")
    parser.add_argument("--base-url", default="http://30.14.1.197:8080", help="模型服务地址")
    parser.add_argument("--model", default=None, help="模型名，默认从 /v1/models 自动获取")
    parser.add_argument("--api-key", default=None, help="API key，默认 dummy；也可用环境变量 AIO_API_KEY")
    parser.add_argument("-w", "--workers", type=int, default=16, help="并发线程数，默认 16")
    parser.add_argument("--timeout", type=float, default=120, help="单请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = args.api_key or os.environ.get("AIO_API_KEY") or "dummy"
    output = args.output or os.path.splitext(args.image_list)[0] + ".jsonl"

    images = read_image_list(args.image_list)
    done = read_done(output)
    todo = [p for p in images if p not in done]
    logger.info("images=%d done=%d todo=%d output=%s", len(images), len(done), len(todo), output)
    if not todo:
        logger.info("nothing to do")
        return 0

    try:
        model = args.model or resolve_model(args.base_url, api_key)
    except Exception as e:
        logger.error("cannot resolve model from %s: %s (pass --model or --api-key)", args.base_url, e)
        return 1
    logger.info("model=%s", model)

    client = SubtitleClient(args.base_url, model, api_key, args.timeout, args.retries)
    ok = failed = 0
    started = time.time()
    with open(output, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(client, path) for path in todo]
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as e:
                failed += 1
                logger.error("failed: %s", e)
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            if ok % 50 == 0:
                logger.info("progress %d/%d, %.1f img/s", ok, len(todo), ok / (time.time() - started))

    logger.info("finish: success=%d failed=%d elapsed=%.1fs", ok, failed, time.time() - started)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
