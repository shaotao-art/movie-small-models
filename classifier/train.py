import argparse
import math
import os
import time

import swanlab
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from classifier.data import build_dataloaders
from classifier.model import build_model, build_pair_model
from classifier.pair_data import build_manifest_dataloaders
from classifier.utils import evaluate, load_config, seed_everything, setup_logging, train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")

    swan_cfg = cfg.get("swanlab", {})
    exp_name = swan_cfg.get("experiment_name") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg["output_dir"], exp_name)

    logger = setup_logging(rank, os.path.join(run_dir, "train.log"))
    seed_everything(cfg["seed"] + rank)

    if rank == 0:
        swanlab.init(project=swan_cfg.get("project", "finetune"), experiment_name=exp_name, config=cfg)
        logger.info("config: %s", cfg)

    # global_batch_size 固定，按卡数拆单卡 batch；单卡仍太大时用梯度累加补齐
    global_bs = cfg["train"]["global_batch_size"]
    per_rank = max(1, global_bs // world_size)
    if cfg["train"].get("max_micro_batch_size"):
        per_rank = min(per_rank, cfg["train"]["max_micro_batch_size"])
    accumulate = max(1, math.ceil(global_bs / (per_rank * world_size)))
    logger.info(
        "global batch %d = per-rank %d x %d ranks x accumulate %d",
        global_bs, per_rank, world_size, accumulate,
    )
    if per_rank * world_size * accumulate != global_bs:
        logger.warning(
            "effective batch %d != global_batch_size %d (adjust max_micro_batch_size or gpu count)",
            per_rank * world_size * accumulate, global_bs,
        )

    data_cfg = dict(cfg["data"], batch_size=per_rank)
    if "manifests" in data_cfg:  # 双图 pair 数据集
        train_loader, val_loader, train_sampler = build_manifest_dataloaders(data_cfg, distributed)
    else:
        train_loader, val_loader, train_sampler = build_dataloaders(data_cfg, distributed)
        if cfg["model"]["num_classes"] != len(train_loader.dataset.classes):
            logger.warning(
                "model.num_classes=%d but dataset has %d classes",
                cfg["model"]["num_classes"], len(train_loader.dataset.classes),
            )
    logger.info("train samples: %d, val samples: %d, world size: %d",
                len(train_loader.dataset), len(val_loader.dataset), world_size)

    build = build_pair_model if cfg["model"].get("pair") else build_model
    model = build(cfg["model"]).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    amp = use_cuda and cfg["train"].get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if use_cuda else None

    def state_dict_of(m):
        return (m.module if hasattr(m, "module") else m).state_dict()

    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            logger, cfg["train"]["log_freq"], rank, world_size,
            autocast=amp, scaler=scaler, accumulate=accumulate,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, world_size, autocast=amp)
        scheduler.step()

        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch %d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.2e",
                epoch, train_loss, train_acc, val_loss, val_acc, lr,
            )
            swanlab.log(
                {"train/loss": train_loss, "train/acc": train_acc,
                 "val/loss": val_loss, "val/acc": val_acc, "lr": lr},
                step=epoch,
            )
            ckpt = {"epoch": epoch, "model": state_dict_of(model),
                    "optimizer": optimizer.state_dict(), "config": cfg}
            torch.save(ckpt, os.path.join(run_dir, "last.pth"))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(ckpt, os.path.join(run_dir, "best.pth"))
                logger.info("new best val acc: %.4f", best_acc)

    if rank == 0:
        logger.info("done. best val acc: %.4f, checkpoints in %s", best_acc, run_dir)
        swanlab.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
