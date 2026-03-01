from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {
    "alexnet",
    "vgg16",
    "resnet50",
    "inception_v3",
    "mobilenet_v3_large",
}


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    pretrained: bool = True
    freeze_backbone: bool = True  # transfer learning (only head trains)
    dropout: float = 0.2          # used in some heads


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def build_model(cfg: ModelConfig) -> Tuple[nn.Module, int]:
    """
    Returns: (model, input_size)
    input_size used by transforms (224 usually; inception wants 299).
    """
    name = cfg.name.lower().strip()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{cfg.name}'. Supported: {sorted(SUPPORTED_MODELS)}")

    # NOTE: In torchvision>=0.13, use weights=... . To keep compatibility we use try/except.
    if name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if cfg.pretrained else None)
        if cfg.freeze_backbone:
            _freeze_all(model)
        in_feats = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feats, cfg.num_classes)
        input_size = 224

    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if cfg.pretrained else None)
        if cfg.freeze_backbone:
            _freeze_all(model)
        in_feats = model.classifier[6].in_features
        # optionally add dropout in head
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_feats, cfg.num_classes)
        )
        input_size = 224

    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if cfg.pretrained else None)
        if cfg.freeze_backbone:
            _freeze_all(model)
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_feats, cfg.num_classes)
        )
        input_size = 224

    elif name == "inception_v3":
        # Inception expects 299x299
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if cfg.pretrained else None,
            aux_logits=True
        )
        if cfg.freeze_backbone:
            _freeze_all(model)

        # Replace main classifier
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_feats, cfg.num_classes)
        )

        # Replace aux classifier (only used during training when aux_logits=True)
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, cfg.num_classes)

        input_size = 299

    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained else None
        )
        if cfg.freeze_backbone:
            _freeze_all(model)
        in_feats = model.classifier[3].in_features
        # classifier = [Linear, Hardswish, Dropout, Linear]
        model.classifier[3] = nn.Linear(in_feats, cfg.num_classes)
        input_size = 224

    # Ensure head params are trainable even if backbone frozen
    for p in model.parameters():
        # some heads might still be frozen due to _freeze_all
        pass
    # Explicitly unfreeze the last layers we replaced
    # (safe approach: unfreeze all, then freeze backbone if needed)
    # But we already froze everything above. So just set replaced head params trainable:
    for p in model.parameters():
        # leave as-is
        ...
    # Better: just make sure all params in model are correct:
    # We'll mark all params in head as trainable by scanning common attribute names.
    def _set_trainable(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    if name in {"resnet50", "inception_v3"}:
        _set_trainable(model.fc)
        if name == "inception_v3" and model.AuxLogits is not None:
            _set_trainable(model.AuxLogits)
    elif name in {"alexnet", "vgg16", "mobilenet_v3_large"}:
        _set_trainable(model.classifier)

    return model, input_size


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]