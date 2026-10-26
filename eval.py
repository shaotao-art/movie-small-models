#!/usr/bin/env python
"""对图片 / 图片列表跑分类器，所有推理结果写到一个 jsonl。

用法:
    python eval.py --ckpt outputs/resnet18/best.pth --image data/frames/xxx/frame_000001.jpg
    python eval.py --ckpt outputs/resnet18/best.pth --image-list list.txt -o preds.jsonl
    # 纯权重（tools/export_weights.py 导出）需要 --config
    python eval.py --ckpt outputs/resnet18/best_weights.pth --config configs/resnet18.yaml --image a.jpg
    # 视频请用 eval_video.py
"""

import argparse
import json
import logging
import os

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from config import CLASSIFIER_BATCH_SIZE, EVAL_RESULTS
from data import _transforms
from data_making.build_cls_dataset import resolve_image
from model import build_model
from utils import load_config

logger = logging.getLogger("eval")


def read_list(path):
    """每行一个图片路径；也兼容 {"image": ...} 的 jsonl。"""
    paths = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line)["image"] if line.startswith("{") else line)
    return paths


def resolve(path):
    real = resolve_image(path)  # 兼容 jsonl 里 "frames/..." 的旧路径
    if real is None:
        raise FileNotFoundError(path)
    return real


@torch.no_grad()
def predict(model, paths, tf, device, batch_size):
    probs = []
    for i in range(0, len(paths), batch_size):
        tensors = []
        for p in paths[i:i + batch_size]:
            with Image.open(p) as im:
                tensors.append(tf(im.convert("RGB")))
        probs += model(torch.stack(tensors).to(device)).softmax(1).cpu().tolist()
    return probs


def load_model(ckpt_path, config_path=None, classes=None):
    """加载 checkpoint 或纯权重，返回 (model, cfg, classes)。classes 为逗号分隔字符串或列表。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:  # train.py 的 checkpoint
        state, ckpt_cfg = ckpt["model"], ckpt.get("config")
    else:  # tools/export_weights.py 导出的纯 state_dict
        state, ckpt_cfg = ckpt, None
    cfg = load_config(config_path) if config_path else ckpt_cfg
    if cfg is None:
        raise ValueError("权重文件里没有训练配置，请用 --config 指定")
    if classes:
        classes = classes.split(",") if isinstance(classes, str) else classes
    else:
        train_dir = os.path.join(cfg["data"]["root"], "train")
        if os.path.isdir(train_dir):
            classes = ImageFolder(train_dir).classes
        elif cfg["model"]["num_classes"] == 2:  # 推理机上没有训练集时用 build_cls_dataset 的固定类别
            classes = ["has_subtitle", "no_subtitle"]
            logger.warning("找不到训练集 %s，使用默认类别 %s（可用 --classes 覆盖）", train_dir, classes)
        else:
            raise ValueError(f"找不到训练集 {train_dir}，请用 --classes 指定类别顺序")
    if len(classes) != cfg["model"]["num_classes"]:
        raise ValueError(f"{len(classes)} classes {classes} != model.num_classes {cfg['model']['num_classes']}")

    model = build_model(dict(cfg["model"], pretrained=False))  # 权重来自文件，不下载预训练
    model.load_state_dict(state)
    return model, cfg, classes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="best.pth / last.pth，或导出的 *_weights.pth")
    parser.add_argument("--config", default=None, help="不传则用 ckpt 里保存的配置；纯权重文件必须传")
    parser.add_argument("--image", action="append", default=[], help="单张图片，可重复")
    parser.add_argument("--image-list", action="append", default=[], help="图片列表文件，可重复")
    parser.add_argument("--classes", default=None, help="逗号分隔的类别顺序，默认从训练集目录推断")
    parser.add_argument("-o", "--output", default=EVAL_RESULTS)
    parser.add_argument("--batch-size", type=int, default=CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        model, cfg, classes = load_model(args.ckpt, args.config, args.classes)
    except ValueError as e:
        parser.error(str(e))
    model.to(args.device).eval()
    tf = _transforms(cfg["data"]["image_size"])[1]

    items = []  # (图片路径, 附加字段)
    for image in args.image:
        items.append((resolve(image), {"source": "image"}))
    for list_file in args.image_list:
        items += [(resolve(p), {"source": "image_list", "list": list_file}) for p in read_list(list_file)]
    if not items:
        parser.error("至少需要 --image / --image-list 之一")

    logger.info("%d images, classes=%s", len(items), classes)
    probs = predict(model, [p for p, _ in items], tf, args.device, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_sub = 0
    with open(args.output, "w") as f:
        for (path, extra), prob in zip(items, probs):
            label = classes[prob.index(max(prob))]
            n_sub += label == "has_subtitle"
            rec = {
                "image": path,
                "result": {"has_subtitle": label == "has_subtitle", "label": label, "prob": max(prob)},
                "scores": dict(zip(classes, prob)),
                **extra,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("%d results (%d has_subtitle) -> %s", len(items), n_sub, args.output)


if __name__ == "__main__":
    main()
