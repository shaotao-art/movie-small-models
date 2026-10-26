import os

from config import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _transforms(image_size):
    train_tf = transforms.Compose([
        # transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(int(image_size * RESIZE_RATIO)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * RESIZE_RATIO)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_dataloaders(cfg, distributed=False):
    """cfg is the `data` section of the yaml. Layout: <root>/train/<class>/*.jpg, <root>/val/<class>/*.jpg"""
    root = cfg["root"]
    train_tf, val_tf = _transforms(cfg["image_size"])
    train_ds = ImageFolder(os.path.join(root, "train"), train_tf)
    val_ds = ImageFolder(os.path.join(root, "val"), val_tf)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    common = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(
        train_ds, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=distributed, **common,
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader, train_sampler
