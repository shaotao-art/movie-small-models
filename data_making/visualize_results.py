#!/usr/bin/env python
"""把 dist_use_qwen38.py 输出的 jsonl 渲染成可浏览的 HTML，人工检查字幕提取效果。

用法:
    python visualize_results.py sample100.jsonl -o check.html
    python visualize_results.py sample100.jsonl -o check.html --movies-dir /path/to/movies
"""

import argparse
import hashlib
import html
import json
import os
import re
from urllib.parse import quote

VIDEO_EXTS = (".mp4", ".mkv", ".rmvb", ".avi", ".mov", ".ts", ".wmv", ".flv", ".webm")


def movie_names(movies_dir):
    """frames/<md5(文件名)> 反查电影名。"""
    names = {}
    if not movies_dir or not os.path.isdir(movies_dir):
        return names
    for root, _, files in os.walk(movies_dir):
        for name in files:
            if name.lower().endswith(VIDEO_EXTS):
                names[hashlib.md5(name.encode()).hexdigest()] = name
    return names


def card(rec, names, base_dir):
    result = rec.get("result") or {}
    has_sub = bool(result.get("has_subtitle"))
    image = rec["image"]
    if not os.path.exists(image) and os.path.exists(os.path.join("data", image)):
        image = os.path.join("data", image)  # 兼容 frames/ 挪到 data/ 后的旧路径
    src = quote(os.path.relpath(os.path.abspath(image), base_dir))
    hash_dir = os.path.basename(os.path.dirname(image))
    movie = names.get(hash_dir, hash_dir[:8])
    frame = os.path.basename(image)
    m = re.search(r"(\d+)", frame)
    minute = (int(m.group(1)) - 1) if m else None

    return f"""
    <div class="card" data-has="{int(has_sub)}">
      <img src="{src}" loading="lazy" alt="">
      <div class="meta">
        <span class="badge {'yes' if has_sub else 'no'}">{'有字幕' if has_sub else '无字幕'}</span>
        <span class="movie">{html.escape(movie)}</span>
        {f'<span class="time">≈{minute} min</span>' if minute is not None else ''}
      </div>
      <div class="cn">{html.escape(result.get('subtitle_cn') or '')}</div>
      <div class="en">{html.escape(result.get('subtitle_en') or '')}</div>
      <div class="path">{html.escape(image)}</div>
    </div>"""


def render(records, names, out_path):
    cards = "\n".join(card(r, names, os.path.dirname(os.path.abspath(out_path))) for r in records)
    n_sub = sum(bool((r.get("result") or {}).get("has_subtitle")) for r in records)
    title = f"字幕提取检查 ({len(records)} 张, 有字幕 {n_sub})"

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 24px; background: #14161a; color: #e6e6e6;
         font-family: -apple-system, "PingFang SC", sans-serif; }}
  h1 {{ font-size: 18px; margin: 0 0 16px; }}
  .filters {{ margin-bottom: 16px; display: flex; gap: 8px; }}
  .filters button {{ background: #24272e; color: #e6e6e6; border: 1px solid #3a3e47;
                     border-radius: 6px; padding: 6px 12px; cursor: pointer; }}
  .filters button.active {{ background: #3b6cf6; border-color: #3b6cf6; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
  .card {{ background: #1d2026; border: 1px solid #2c3038; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; background: #000; }}
  .meta {{ display: flex; align-items: center; gap: 8px; padding: 8px 10px 0; font-size: 12px; }}
  .badge {{ padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
  .badge.yes {{ background: #1f6f43; }}
  .badge.no {{ background: #4a2b2b; }}
  .movie {{ color: #9aa3b2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .time {{ margin-left: auto; color: #6b7381; white-space: nowrap; }}
  .cn {{ padding: 8px 10px 0; font-size: 16px; }}
  .en {{ padding: 4px 10px 10px; font-size: 13px; color: #8fb4ff; }}
  .path {{ padding: 0 10px 10px; font-size: 11px; color: #5c6470; word-break: break-all; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="filters">
  <button class="active" data-f="all">全部 ({len(records)})</button>
  <button data-f="1">有字幕 ({n_sub})</button>
  <button data-f="0">无字幕 ({len(records) - n_sub})</button>
</div>
<div class="grid">
{cards}
</div>
<script>
  document.querySelectorAll('.filters button').forEach(btn => {{
    btn.onclick = () => {{
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const f = btn.dataset.f;
      document.querySelectorAll('.card').forEach(c =>
        c.style.display = (f === 'all' || c.dataset.has === f) ? '' : 'none');
    }};
  }});
</script>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="dist_use_qwen38.py 的输出")
    parser.add_argument("-o", "--output", default=None, help="HTML 输出路径，默认与 jsonl 同名 .html")
    parser.add_argument("--movies-dir", default="/Users/taoshao/Downloads/movies", help="用于反查电影名")
    args = parser.parse_args(argv)

    with open(args.jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]
    output = args.output or os.path.splitext(args.jsonl)[0] + ".html"
    names = movie_names(args.movies_dir)

    with open(output, "w") as f:
        f.write(render(records, names, output))
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    raise SystemExit(main())
