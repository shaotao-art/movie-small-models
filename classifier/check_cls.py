#!/usr/bin/env python
"""快速检查数据集和模型：类别顺序、样本量、一个 batch 的形状、前向/反向和吞吐。

用法:
    python -m classifier.check_cls
    python -m classifier.check_cls --config classifier/configs/resnet18.yaml --batches 5
"""

import argparse
import logging
import time

import torch
import torch.nn.functional as F

from classifier.data import build_dataloaders
from classifier.model import build_model
from classifier.utils import load_config

logger = logging.getLogger("check")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="classifier/configs/resnet18.yaml")
    parser.add_argument("--batches", type=int, default=3, help="测几个训练 batch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)

    train_loader, val_loader, _ = build_dataloaders(
        dict(cfg["data"], batch_size=cfg["train"]["global_batch_size"])
    )
    logger.info("classes (index order): %s", train_loader.dataset.classes)
    logger.info("train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(cfg["model"])
    logger.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    started = time.time()
    for step, (images, targets) in enumerate(train_loader):
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logger.info(
            "train batch %d: images=%s labels=%s loss=%.4f",
            step, tuple(images.shape), targets.bincount().tolist(), loss.item(),
        )
        if step + 1 >= args.batches:
            break
    logger.info("%.2fs per batch", (time.time() - started) / args.batches)

    model.eval()
    with torch.no_grad():
        images, targets = next(iter(val_loader))
        logits = model(images)
        logger.info(
            "val batch: images=%s logits=%s acc=%.4f",
            tuple(images.shape), tuple(logits.shape),
            (logits.argmax(1) == targets).float().mean().item(),
        )


if __name__ == "__main__":
    main()
