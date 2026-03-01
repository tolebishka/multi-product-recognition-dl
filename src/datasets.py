from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: str = "data/splits"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # split names
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Standard ImageNet-style transforms for transfer learning.
    """
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # ~256 when image_size=224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def _assert_split_dir(path: Path, split: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Split folder not found: {path}\n"
            f"Expected structure: data/splits/{split}/<class_name>/*.jpg"
        )
    # Check at least one class directory exists
    has_class_dir = any(p.is_dir() for p in path.iterdir())
    if not has_class_dir:
        raise FileNotFoundError(
            f"No class folders inside: {path}\n"
            f"Create folders like: data/splits/{split}/apple, data/splits/{split}/banana, ..."
        )


def create_datasets(cfg: DataConfig):
    base = Path(cfg.data_dir)

    train_dir = base / cfg.train_split
    val_dir = base / cfg.val_split
    test_dir = base / cfg.test_split

    _assert_split_dir(train_dir, cfg.train_split)
    _assert_split_dir(val_dir, cfg.val_split)
    _assert_split_dir(test_dir, cfg.test_split)

    tfms = get_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=tfms["train"])
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=tfms["val"])
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=tfms["test"])

    return train_ds, val_ds, test_ds


def create_dataloaders(cfg: DataConfig):
    train_ds, val_ds, test_ds = create_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    # class mapping to save in runs
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class