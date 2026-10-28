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
