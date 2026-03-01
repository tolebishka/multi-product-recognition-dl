from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from datasets import DataConfig, create_dataloaders
from models import ModelConfig, build_model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(y.view(-1, 1)).any(dim=1).float().sum().item()
    return correct / y.size(0)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def benchmark_inference(model: nn.Module, device: torch.device, input_size: int, iters: int = 200) -> Dict[str, float]:
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)

    # warmup
    for _ in range(20):
        _ = model(x)

    # timed
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    t = time.time() - t0

    ms_per_img = (t / iters) * 1000.0
    fps = 1000.0 / ms_per_img
    return {"ms_per_image": ms_per_img, "fps": fps}


def model_size_mb(model: nn.Module) -> float:
    # size of parameters only
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run folder, e.g. runs/resnet50_bs32_lr...")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    args = parser.parse_args()

    device = get_device()
    run_dir = Path(args.run_dir)

    cfg_path = run_dir / "configs.json"
    ckpt_path = run_dir / "best.pt"
    idx_to_class_path = run_dir / "idx_to_class.json"

    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing configs.json or best.pt in {run_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    idx_to_class = json.loads(idx_to_class_path.read_text(encoding="utf-8"))

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    # Rebuild model
    model_cfg = ModelConfig(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,            # weights loaded from checkpoint
        freeze_backbone=False,
        dropout=cfg["model"].get("dropout", 0.2)
    )
    model, input_size = build_model(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Dataloaders (use correct input size)
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=input_size,
        batch_size=32,
        num_workers=2,
        pin_memory=False if device.type == "mps" else True,
    )
    _, _, test_loader, class_to_idx, _ = create_dataloaders(data_cfg)

    y_true = []
    y_pred = []
    y_prob = []

    top1_sum = 0.0
    top5_sum = 0.0
    n = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred = logits.argmax(dim=1)

        b = y.size(0)
        top1_sum += accuracy_topk(logits, y, k=1) * b
        top5_sum += accuracy_topk(logits, y, k=5) * b
        n += b

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    top1 = top1_sum / n
    top5 = top5_sum / n

    # classification report
    labels_sorted = list(range(num_classes))
    target_names = [idx_to_class[str(i)] for i in labels_sorted]

    report = classification_report(
        y_true, y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    # save confusion matrix as csv
    cm_csv = run_dir / "confusion_matrix.csv"
    with cm_csv.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(target_names) + "\n")
        for i, row in enumerate(cm):
            f.write(target_names[i] + "," + ",".join(map(str, row.tolist())) + "\n")

    # benchmark
    speed = benchmark_inference(model, device, input_size, iters=200)
    size_mb = model_size_mb(model)

    metrics = {
        "model": model_name,
        "num_classes": num_classes,
        "test_top1": top1,
        "test_top5": top5,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {k: v for k, v in report.items() if k in target_names},
        "speed": speed,
        "model_size_mb": size_mb
    }

    save_json(run_dir / "metrics.json", metrics)

    print("Saved metrics to:", run_dir / "metrics.json")
    print("Saved confusion matrix to:", cm_csv)
    print(f"TEST top1={top1:.4f} top5={top5:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"Speed: {speed['ms_per_image']:.2f} ms/img ({speed['fps']:.1f} FPS) | Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()