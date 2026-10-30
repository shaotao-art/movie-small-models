#!/usr/bin/env python
"""对 speaker_embeddings.pkl 做简单的余弦聚类，并输出可人工试听的 HTML。

用法:
    python -m tools.cluster_speakers --embeddings data/test/segments/speaker_embeddings.pkl
    python -m tools.cluster_speakers --embeddings a.pkl --threshold 0.3 --no-audio

算法：逐个样本与已有簇中心比余弦相似度，够阈值就并入（中心取成员均值后归一化），
否则新开一簇。简单、和输入顺序有关，只适合人工快速检查。
"""

import argparse
import base64
import html
import json
import logging
import mimetypes
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logger = logging.getLogger("cluster_speakers")

PALETTE = ["#2a78d6", "#e5734a", "#3f9e63", "#9b59b6", "#d4a72c",
           "#5aa9e6", "#b5483a", "#6b8e23", "#c2477c", "#4a8f8f"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cluster_embeddings(embeddings, threshold):
    """返回 (labels, centroids)。贪心：余弦相似度达到阈值就并入最近的簇。"""
    centroids, labels, members = [], [], []
    for i, emb in enumerate(embeddings):
        if centroids:
            sims = [float(emb @ c) for c in centroids]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                labels.append(j)
                members[j].append(i)
                centroids[j] = unit(embeddings[members[j]].mean(axis=0))
                continue
        centroids.append(emb)
        members.append([i])
        labels.append(len(centroids) - 1)
    return labels, centroids


def project_2d(embeddings):
    """PCA 到二维（SVD），返回 Nx2 的坐标；样本太少时退化处理。"""
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2))
    x = embeddings - embeddings.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def audio_uri(path):
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def fmt_time(seconds):
    if seconds is None:
        return "--:--"
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def render(items, labels, coords, meta, embed_audio=True):
    """items: 每条记录（含 file/text/start）；labels: 簇 id；coords: PCA 坐标。"""
    n_clusters = max(labels) + 1 if labels else 0
    order = sorted(range(n_clusters), key=lambda c: -labels.count(c))
    remap = {old: new for new, old in enumerate(order)}

    # 散点图
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    pts = []
    for i, (x, y) in enumerate(coords):
        px = 40 + (x - lo[0]) / span[0] * 720
        py = 40 + (hi[1] - y) / span[1] * 300
        c = PALETTE[remap[labels[i]] % len(PALETTE)]
        tip = html.escape(f"簇{remap[labels[i]]} · {items[i].get('text') or items[i]['file']}")
        pts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{c}" fill-opacity="0.8">'
                   f"<title>{tip}</title></circle>")
    scatter = (f'<svg viewBox="0 0 800 380" width="100%" height="380">'
               f'<rect width="800" height="380" fill="#fff"/>{"".join(pts)}</svg>')

    # 每个簇
    blocks = []
    for old in order:
        idxs = [i for i, l in enumerate(labels) if l == old]
        col = PALETTE[remap[old] % len(PALETTE)]
        rows = []
        for i in idxs:
            it = items[i]
            audio = (f'<audio controls preload="none" src="{audio_uri(it["path"])}"></audio>'
                     if embed_audio else f'<code>{html.escape(it["file"])}</code>')
            rows.append(
                f'<tr><td class="t">{fmt_time(it.get("start"))}</td>'
                f'<td class="txt">{html.escape(it.get("text") or "")}</td>'
                f'<td class="a">{audio}</td></tr>')
        blocks.append(
            f'<section class="cluster"><h2><span class="dot" style="background:{col}"></span>'
            f'簇 {remap[old]} <span class="count">{len(idxs)} 条</span></h2>'
            f'<table>{"".join(rows)}</table></section>')

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>说话人聚类 - {html.escape(meta['name'])}</title>
<style>
  body {{ margin: 0 auto; padding: 24px; max-width: 1000px; background: #fcfcfb;
         color: #1a1a1a; font-family: system-ui, -apple-system, "PingFang SC", sans-serif; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }}
  .summary {{ color: #52514e; font-size: 13px; margin-bottom: 16px; }}
  .panel {{ border: 1px solid #e1e0d9; border-radius: 10px; background: #fff; padding: 8px; }}
  .cluster {{ margin-top: 22px; }}
  h2 {{ font-size: 15px; display: flex; align-items: center; gap: 8px; margin: 0 0 8px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .count {{ color: #898781; font-weight: 400; font-size: 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #f0efec; vertical-align: middle; }}
  td.t {{ color: #898781; font-variant-numeric: tabular-nums; white-space: nowrap; width: 60px; }}
  td.txt {{ width: 45%; }}
  audio {{ height: 32px; width: 300px; }}
</style>
</head>
<body>
<h1>说话人聚类 · {html.escape(meta['name'])}</h1>
<div class="summary">
  共 {len(items)} 条语音 · 聚成 {n_clusters} 簇 · 余弦阈值 {meta['threshold']} ·
  embedding {meta['dim']} 维 · 权重 {html.escape(meta['ckpt'])}
</div>
<div class="panel">{scatter}</div>
{"".join(blocks)}
</body>
</html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True, help="tools.extract_speaker_embeddings 输出的 pkl")
    parser.add_argument("--threshold", type=float, default=0.5, help="余弦相似度阈值，越大簇越多（默认 0.5）")
    parser.add_argument("-o", "--output", default=None, help="默认与 pkl 同目录的 speaker_clusters.html")
    parser.add_argument("--no-audio", action="store_true", help="不内嵌音频（片段多时显著减小 HTML）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.isfile(args.embeddings):
        logger.error("找不到 %s", args.embeddings)
        return 1
    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)

    run_dir = os.path.dirname(os.path.abspath(args.embeddings))
    items = []
    for rec in data["items"]:
        item = dict(rec)
        item["path"] = os.path.join(run_dir, rec["file"])
        if not os.path.isfile(item["path"]):
            logger.warning("音频缺失，跳过: %s", item["path"])
            continue
        items.append(item)
    if not items:
        logger.error("没有可用的音频片段")
        return 1

    E = np.stack([unit(np.asarray(it["embedding"], dtype=np.float32)) for it in items])
    labels, _ = cluster_embeddings(E, args.threshold)
    coords = project_2d(E)
    logger.info("%d 条 -> %d 簇（阈值 %.2f）", len(items), max(labels) + 1, args.threshold)

    meta = {"name": os.path.basename(run_dir), "threshold": args.threshold,
            "dim": int(E.shape[1]), "ckpt": data.get("ckpt", "?")}
    page = render(items, labels, coords, meta, embed_audio=not args.no_audio)

    output = args.output or os.path.join(run_dir, "speaker_clusters.html")
    with open(output, "w") as f:
        f.write(page)
    with open(os.path.splitext(output)[0] + ".json", "w") as f:
        json.dump({"threshold": args.threshold,
                   "clusters": {str(c): [items[i]["file"] for i, l in enumerate(labels) if l == c]
                                for c in range(max(labels) + 1)}},
                  f, ensure_ascii=False, indent=2)

    logger.info("已写出 %s（%.1f MB）", output, os.path.getsize(output) / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
