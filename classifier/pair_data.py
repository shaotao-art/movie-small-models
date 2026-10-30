"""「两帧台词是否相同」的二分类数据集。

PairDataset 在线组样本，比例固定为 正:难负:易负 = 2:1:1：
    正样本   同一 cue 内时间跨度最大的两帧（画面差异大、台词相同）
    难负样本 同一部片子的下一条 cue（同场景，台词不同）
    易负样本 随机一条别的 cue（场景不同）

样本形状 [2, 3, H, W]，默认 collate 后是 [B, 2, 3, H, W]，模型侧 flatten(0, 1) 即可。
"""

import glob
import json
import os
import random
import re

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD

_PUNCT = re.compile(r"[\s　，。！？、；：,.!?;:\"'“”‘’（）()《》<>…—-]+")


def normalize_text(text):
    """去标点/空白、统一小写，只用来判断两句台词是否相同。"""
    return _PUNCT.sub("", (text or "").lower())


def build_transform(train):
    ops = [transforms.ColorJitter(0.1, 0.1)] if train else []
    return transforms.Compose(ops + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


DEFAULT_CROP = (0.125, 0.8, 0.875, 1.0)  # x0, y0, x1, y1：底部 1/5，左右各去掉 1/8


def load_image(root, rel, size, crop, tf):
    """读图 -> 按 crop 比例裁剪（字幕区）-> 缩放到固定尺寸 -> 张量化。"""
    img = Image.open(os.path.join(root, rel)).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = crop
    img = img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    return tf(img.resize(size, Image.BICUBIC))


def load_cues(paths):
    """读 cues jsonl（可为目录），返回 [{..., 'root': 帧路径根目录}]，按 movie + cue_id 排序。"""
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        files.extend(glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else [p])
    cues = []
    for f in sorted(files):
        root = os.path.dirname(os.path.dirname(os.path.abspath(f)))  # <out>/cues/x.jsonl -> <out>
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    cues.append(dict(json.loads(line), root=root))
    if not cues:
        raise ValueError(f"没有找到 cue：{paths}")
    cues.sort(key=lambda c: (c["movie"], c["cue_id"]))
    return cues


class PairDataset(Dataset):
    def __init__(self, cues, size=(448, 128), crop=DEFAULT_CROP, train=True, seed=0):
        self.cues = cues
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.crop = tuple(crop)
        self.train = train
        self.rng = random.Random(seed)
        self.groups = {}  # movie -> [cue 在 self.cues 中的下标]
        for i, c in enumerate(cues):
            self.groups.setdefault(c["movie"], []).append(i)
        self.movies = sorted(self.groups)
        self.pairs = self._fixed_pairs() if not train else None
        self.tf = build_transform(train)

    # ---------- 采样 ----------
    def _fixed_pairs(self):
        """验证集：固定的 (i, j, label) 列表，保证每轮可比。"""
        rng = random.Random(1234)
        pairs = []
        for i in range(len(self.cues)):
            pairs.append((i, i, 1))  # 正样本：同 cue 首尾两帧
            j = self._hard_negative(i)
            if j is not None:
                pairs.append((i, j, 0))
            pairs.append((i, self._easy_negative(i, rng), 0))
        return pairs

    def _hard_negative(self, i):
        group = self.groups[self.cues[i]["movie"]]
        pos = group.index(i)
        text = normalize_text(self.cues[i]["text"])
        for nxt in group[pos + 1: pos + 4]:  # 拆分句子的相邻 cue 跳过
            if normalize_text(self.cues[nxt]["text"]) != text:
                return nxt
        return None

    def _easy_negative(self, i, rng):
        text = normalize_text(self.cues[i]["text"])
        for _ in range(20):
            j = rng.randrange(len(self.cues))
            if j != i and self.cues[j]["movie"] != self.cues[i]["movie"] \
                    and normalize_text(self.cues[j]["text"]) != text:
                return j
        return (i + len(self.cues) // 2) % len(self.cues)

    def __len__(self):
        return len(self.pairs) if not self.train else 4 * len(self.cues)

    def __getitem__(self, idx):
        if self.train:
            i, kind = idx // 4, idx % 4
            if kind in (0, 1):          # 正样本
                j, label = i, 1
            elif kind == 2:             # 难负
                j = self._hard_negative(i)
                if j is None:
                    j = self._easy_negative(i, self.rng)
                label = 0
            else:                       # 易负
                j, label = self._easy_negative(i, self.rng), 0
        else:
            i, j, label = self.pairs[idx]

        ci, cj = self.cues[i], self.cues[j]
        if i == j:  # 同一 cue：取时间跨度最大的两帧
            fa = random.choice(ci["frames"][: max(1, len(ci["frames"]) // 2)])
            fb = random.choice(ci["frames"][len(ci["frames"]) // 2:])
        else:
            fa, fb = random.choice(ci["frames"]), random.choice(cj["frames"])
        a, b = self._load(ci["root"], fa), self._load(cj["root"], fb)
        return torch.stack([a, b]), label

    # ---------- 读图 ----------
    def _load(self, root, rel):
        return load_image(root, rel, self.size, self.crop, self.tf)


def build_pair_dataloaders(cfg, distributed=False):
    """cfg 来自 yaml 的 data 段：cues / size / crop / batch_size / num_workers / val_ratio。"""
    cues = load_cues(cfg["cues"])
    movies = sorted({c["movie"] for c in cues})
    val_movies = set(cfg.get("val_movies") or movies[-max(1, round(len(movies) * cfg.get("val_ratio", 0.15))):])
    train_cues = [c for c in cues if c["movie"] not in val_movies]
    val_cues = [c for c in cues if c["movie"] in val_movies]
    if not train_cues or not val_cues:
        raise ValueError(f"按电影切分后 train={len(train_cues)} val={len(val_cues)}，至少各要 1 条")

    kw = dict(size=cfg.get("size", (448, 128)), crop=cfg.get("crop", DEFAULT_CROP))
    train_ds = PairDataset(train_cues, train=True, **kw)
    val_ds = PairDataset(val_cues, train=False, **kw)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None,
                              drop_last=distributed, **common)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader, train_sampler


class ManifestPairDataset(Dataset):
    """读 data_making/srt_model/build_pairs.py 生成的 train.jsonl / val.jsonl。"""

    def __init__(self, manifest, root, size=(448, 128), crop=DEFAULT_CROP, train=False):
        with open(manifest) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"空清单: {manifest}")
        self.root = root
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.crop = tuple(crop)
        self.tf = build_transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        a = load_image(self.root, row["a"], self.size, self.crop, self.tf)
        b = load_image(self.root, row["b"], self.size, self.crop, self.tf)
        return torch.stack([a, b]), row["label"]


def build_manifest_dataloaders(cfg, distributed=False):
    """cfg 的 data 段：root / manifests{train,val} / size / crop / batch_size / num_workers。"""
    kw = dict(size=cfg.get("size", (448, 128)), crop=cfg.get("crop", DEFAULT_CROP))
    root = cfg.get("root", "data/pairs")
    train_ds = ManifestPairDataset(cfg["manifests"]["train"], root, train=True, **kw)
    val_ds = ManifestPairDataset(cfg["manifests"]["val"], root, train=False, **kw)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None,
                              drop_last=distributed, **common)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader, train_sampler
