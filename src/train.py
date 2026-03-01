from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets import DataConfig, create_dataloaders
from models import ModelConfig, build_model, get_trainable_params


def get_device() -> torch.device:
    # Mac Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    # logits: [B, C], y: [B]
    topk = logits.topk(k, dim=1).indices  # [B, k]
    correct = topk.eq(y.view(-1, 1)).any(dim=1).float().sum().item()
    return correct / y.size(0)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    is_inception: bool = False,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if is_inception:
            # inception returns (logits, aux_logits) in train mode
            logits, aux_logits = model(x)
            loss = criterion(logits, y) + 0.4 * criterion(aux_logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        b = y.size(0)
        running_loss += loss.item() * b
        running_top1 += accuracy_topk(logits, y, k=1) * b
        running_top5 += accuracy_topk(logits, y, k=5) * b
        n += b

    return {
        "loss": running_loss / n,
        "top1": running_top1 / n,
        "top5": running_top5 / n,
    }


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    is_inception: bool = False,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # in eval mode inception returns only logits
        loss = criterion(logits, y)

        b = y.size(0)
        running_loss += loss.item() * b
        running_top1 += accuracy_topk(logits, y, k=1) * b
        running_top5 += accuracy_topk(logits, y, k=5) * b
        n += b

    return {
        "loss": running_loss / n,
        "top1": running_top1 / n,
        "top5": running_top5 / n,
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["alexnet", "vgg16", "resnet50", "inception_v3", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device()
    print("Device:", device)

    # Create loaders
    # NOTE: input_size differs for inception (299)
    # We'll build model first, then set transforms size accordingly by passing cfg.image_size.
    tmp_cfg = ModelConfig(name=args.model, num_classes=2)  # placeholder
    _, input_size = build_model(tmp_cfg)

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=input_size,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=False if device.type == "mps" else True
    )

    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = create_dataloaders(data_cfg)
    num_classes = len(class_to_idx)
    print("Classes:", num_classes, class_to_idx)

    # Build model with correct num_classes
    freeze = True
    if args.unfreeze_backbone:
        freeze = False

    model_cfg = ModelConfig(
        name=args.model,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=freeze,
        dropout=0.2
    )

    model, _ = build_model(model_cfg)
    model = model.to(device)

    # Train only trainable params (head if backbone frozen)
    params = get_trainable_params(model)
    optimizer = Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    is_inception = (args.model == "inception_v3")

    # Run folder
    run_id = f"{args.model}_bs{args.batch_size}_lr{args.lr}_{int(time.time())}"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "class_to_idx.json", class_to_idx)
    save_json(run_dir / "idx_to_class.json", idx_to_class)
    save_json(run_dir / "configs.json", {"data": asdict(data_cfg), "model": asdict(model_cfg), "args": vars(args)})

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, is_inception=is_inception)
        val_metrics = eval_one_epoch(model, val_loader, criterion, device, is_inception=is_inception)

        scheduler.step(val_metrics["loss"])

        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 2),
            "train_loss": round(train_metrics["loss"], 4),
            "train_top1": round(train_metrics["top1"], 4),
            "train_top5": round(train_metrics["top5"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_top1": round(val_metrics["top1"], 4),
            "val_top5": round(val_metrics["top5"], 4),
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {row['train_loss']} top1 {row['train_top1']:.3f} | "
            f"Val loss {row['val_loss']} top1 {row['val_top1']:.3f} | "
            f"lr {row['lr']}"
        )

        # Save best
        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "model_cfg": asdict(model_cfg)}, run_dir / "best.pt")
        else:
            no_improve += 1

        save_json(run_dir / "history.json", history)

        # Early stopping
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
            break

    print(f"Training done. Best epoch: {best_epoch} val_loss={best_val_loss:.4f}")
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()