#!/usr/bin/env python
"""把人脸检测/聚类结果渲染成一个 HTML（结果图 + 全部检测明细）。

用法:
    python -m faces.visualize_faces                          # 默认 outputs/face_clusters/<视频名>/
    python -m faces.visualize_faces outputs/face_clusters/xxx
输出:
    <结果目录>/report.html
"""

import argparse
import base64
import html
import json
import mimetypes
import os

from config import FACE_CLUSTER_DIR, VIDEO_PATH


def data_uri(path: str) -> str:
    """图片直接内嵌成 data URI，HTML 拷到哪都能打开。"""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_faces(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def card(label: str, info: dict, kind: str, run_dir: str) -> str:
    crops = "".join(
        f'<img src="{data_uri(os.path.join(run_dir, name))}" alt="">'
        for name in info.get("crops", [])
        if os.path.exists(os.path.join(run_dir, name))
    )
    tag = "未知" if kind == "unknown" else "角色"
    return f"""
    <div class="card">
      <div class="crops">{crops or '<div class="empty">无代表图</div>'}</div>
      <div class="name">{html.escape(label)} <span class="tag {kind}">{tag}</span></div>
      <div class="meta">{info.get('faces', 0)} 张脸 ·
        {fmt_time(info.get('first_seen', 0))} - {fmt_time(info.get('last_seen', 0))}</div>
    </div>"""


def table(faces: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{fmt_time(r['timestamp'])}</td><td>{r['timestamp']:.3f}</td>"
        f"<td>{r.get('frame_index', '')}</td><td>{html.escape(str(r.get('person', '')))}</td>"
        f"<td>{r.get('match', '')}</td><td>{r.get('score', '')}</td>"
        f"<td>{r.get('bbox', '')}</td></tr>"
        for r in faces
    )
    return f"""
    <details>
      <summary>全部检测明细（{len(faces)} 条）</summary>
      <table>
        <thead><tr><th>时间</th><th>秒</th><th>帧号</th><th>人物</th>
        <th>匹配分</th><th>检测分</th><th>bbox</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </details>"""


def render(data: dict, faces: list[dict], run_dir: str) -> str:
    characters = data.get("characters", {})
    unknown = data.get("unknown_clusters", data.get("clusters", {}))
    video = os.path.basename(data.get("video", ""))
    cards = "".join(card(k, v, "character", run_dir) for k, v in characters.items())
    cards += "".join(card(k, v, "unknown", run_dir) for k, v in unknown.items())
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>人脸检测结果 - {html.escape(video)}</title>
<style>
  body {{ margin: 0; padding: 24px; max-width: 1200px; margin: 0 auto;
         font-family: system-ui, -apple-system, "PingFang SC", sans-serif;
         color: #1a1a1a; background: #fcfcfb; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  .summary {{ color: #52514e; font-size: 13px; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }}
  .card {{ border: 1px solid #e1e0d9; border-radius: 10px; overflow: hidden; background: #fff; }}
  .crops {{ display: flex; gap: 2px; background: #f0efec; aspect-ratio: 3 / 1; }}
  .crops img {{ width: 33.33%; height: 100%; object-fit: cover; display: block; }}
  .empty {{ width: 100%; display: flex; align-items: center; justify-content: center; color: #898781; font-size: 12px; }}
  .name {{ padding: 10px 12px 0; font-size: 15px; font-weight: 600; }}
  .tag {{ font-size: 11px; font-weight: 400; padding: 2px 7px; border-radius: 999px; margin-left: 4px; }}
  .tag.character {{ background: #cde2fb; color: #184f95; }}
  .tag.unknown {{ background: #f0efec; color: #52514e; }}
  .meta {{ padding: 4px 12px 12px; font-size: 12px; color: #52514e; }}
  details {{ margin-top: 28px; }}
  summary {{ cursor: pointer; font-size: 14px; color: #2a78d6; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px;
           font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; }}
  th {{ color: #52514e; font-weight: 500; position: sticky; top: 0; background: #fcfcfb; }}
</style>
</head>
<body>
<h1>人脸检测结果 · {html.escape(video)}</h1>
<div class="summary">
  每 {data.get('sample_seconds', '?')}s 抽帧 ·
  共 {data.get('faces_total', len(faces))} 张脸 ·
  识别角色 {len(characters)} 个 · 未知簇 {len(unknown)} 个 ·
  聚类阈值 {data.get('thresh', '?')} / 匹配阈值 {data.get('match_thresh', '?')}
</div>
<div class="grid">{cards}</div>
{table(faces)}
</body>
</html>"""


def main(argv=None):
    stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=os.path.join(FACE_CLUSTER_DIR, stem),
                        help="faces/cluster_faces.py 的输出目录")
    parser.add_argument("-o", "--output", default=None, help="默认 <input>/report.html")
    args = parser.parse_args(argv)

    summary_path = os.path.join(args.input, "clusters.json")
    if not os.path.exists(summary_path):
        parser.error(f"找不到 {summary_path}，先运行 faces/cluster_faces.py")
    output = args.output or os.path.join(args.input, "report.html")

    data = load_json(summary_path)
    faces = load_faces(os.path.join(args.input, "faces.jsonl"))
    with open(output, "w") as f:
        f.write(render(data, faces, args.input))
    print(f"{len(faces)} 条检测结果 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
