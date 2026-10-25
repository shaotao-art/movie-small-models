import contextlib
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(rank, log_file=None):
    handlers = [logging.StreamHandler()]
    if rank == 0 and log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("train")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def _reduce_metrics(loss_sum, loss_count, correct, total, device, world_size):
    stats = torch.tensor([loss_sum, loss_count, correct, total], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item(), (stats[2] / stats[3]).item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, log_freq, rank, world_size,
                    autocast=False, scaler=None, accumulate=1):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=autocast):
            logits = model(images)
            loss = criterion(logits, targets)

        # 梯度累加：非边界 micro-batch 不做 DDP 同步，最后一步再 all-reduce
        boundary = (step + 1) % accumulate == 0 or step + 1 == len(loader)
        sync = model.no_sync() if hasattr(model, "no_sync") and not boundary else contextlib.nullcontext()
        with sync:
            if scaler is not None:
                scaler.scale(loss / accumulate).backward()
            else:
                (loss / accumulate).backward()

        if boundary:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), targets.size(0))
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

        if rank == 0 and step % log_freq == 0:
            logger.info(
                "epoch %d step %d/%d loss %.4f acc %.4f",
                epoch, step, len(loader), loss_meter.avg, correct / total,
            )
    return _reduce_metrics(loss_meter.sum, loss_meter.count, correct, total, device, world_size)


@torch.no_grad()
def evaluate(model, loader, criterion, device, world_size, autocast=False):
    model.eval()
    loss_meter = AverageMeter()
    correct = total = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast):
            logits = model(images)
            loss = criterion(logits, targets)
        loss_meter.update(loss.item(), targets.size(0))
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return _reduce_metrics(loss_meter.sum, loss_meter.count, correct, total, device, world_size)
