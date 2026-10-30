import torch
import torch.nn as nn
from torchvision import models


def _head(builder, cfg, attr):
    """换掉分类头：attr 为 'fc' 或 'classifier'（取最后那个 Linear）。"""
    net = builder(weights="DEFAULT" if cfg["pretrained"] else None)
    if attr == "fc":
        net.fc = nn.Linear(net.fc.in_features, cfg["num_classes"])
    else:
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, cfg["num_classes"])
    return net


# Add new architectures here: name -> (cfg) -> nn.Module
MODELS = {
    "resnet18": lambda cfg: _head(models.resnet18, cfg, "fc"),
    "resnet50": lambda cfg: _head(models.resnet50, cfg, "fc"),
    "regnet_x_400mf": lambda cfg: _head(models.regnet_x_400mf, cfg, "fc"),
    "mobilenet_v3_small": lambda cfg: _head(models.mobilenet_v3_small, cfg, "classifier"),
    "mobilenet_v3_large": lambda cfg: _head(models.mobilenet_v3_large, cfg, "classifier"),
    "efficientnet_b0": lambda cfg: _head(models.efficientnet_b0, cfg, "classifier"),
    "mnasnet0_75": lambda cfg: _head(models.mnasnet0_75, cfg, "classifier"),
}


def build_model(cfg):
    if cfg["name"] not in MODELS:
        raise ValueError(f"unknown model {cfg['name']!r}, available: {list(MODELS)}")
    return MODELS[cfg["name"]](cfg)


# ---------------- 双图（两帧台词是否相同） ----------------
# name -> (torchvision 工厂, features 输出通道数)
PAIR_BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
    "resnet18": (models.resnet18, 512),
}


class PairModel(nn.Module):
    """两帧共享 backbone，拼接 [f1, f2, |f1-f2|, f1*f2] 后做二分类。

    输入 [B, 2, 3, H, W]（DataLoader 默认 collate 的形状），输出 [B, num_classes]。
    """

    def __init__(self, backbone, feat_dim, num_classes=2, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, images):
        if images.dim() == 4:  # 单条样本 [2,3,H,W]
            images = images.unsqueeze(0)
        b, p = images.shape[:2]
        feat = self.backbone(images.flatten(0, 1)).view(b, p, -1)
        f1, f2 = feat[:, 0], feat[:, 1]
        return self.head(torch.cat([f1, f2, (f1 - f2).abs(), f1 * f2], dim=1))


def build_pair_model(cfg):
    name = cfg["name"]
    if name not in PAIR_BACKBONES:
        raise ValueError(f"{name!r} 没有注册双图 backbone，可选: {list(PAIR_BACKBONES)}")
    builder, feat_dim = PAIR_BACKBONES[name]
    net = builder(weights="DEFAULT" if cfg.get("pretrained") else None)
    if hasattr(net, "features"):  # mobilenet / efficientnet：features + 全局池化
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:  # resnet：去掉 fc，保留 avgpool
        backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
    return PairModel(backbone, feat_dim, cfg.get("num_classes", 2), cfg.get("dropout", 0.2))
