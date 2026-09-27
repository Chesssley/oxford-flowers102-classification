"""Train a CNN using train only and select checkpoints using validation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import get_dataset, make_loader
from .engine import run_epoch
from .models import build_model, parameter_counts
from .utils import OUTPUT_DIR, environment_info, save_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    """All user-adjustable training parameters saved with the checkpoint."""

    model: str
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    patience: int
    seed: int
    smoke_test: bool


def train_experiment(config: TrainConfig) -> dict[str, Any]:
    """Train one model, save its best validation checkpoint and history."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = get_dataset("train")
    val_data = get_dataset("val")
    train_loader = make_loader(
        train_data, batch_size=config.batch_size, workers=config.workers, shuffle=True, seed=config.seed
    )
    val_loader = make_loader(
        val_data, batch_size=config.batch_size, workers=config.workers, shuffle=False, seed=config.seed
    )
    model = build_model(config.model, pretrained=config.model == "resnet18").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    training_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    evaluation_loss = nn.CrossEntropyLoss()
    max_batches = 2 if config.smoke_test else None
    epoch_count = 1 if config.smoke_test else config.epochs
    if config.smoke_test:
        run_dir = OUTPUT_DIR / "smoke" / config.model
        checkpoint_path = run_dir / "best.pt"
        history_path = run_dir / "history.json"
    else:
        checkpoint_path = OUTPUT_DIR / "checkpoints" / f"{config.model}-best.pt"
        history_path = OUTPUT_DIR / "metrics" / f"{config.model}-history.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    info = environment_info(device)
    print(f"model={config.model} device={device} gpu={info['gpu']} batch={config.batch_size} smoke={config.smoke_test}", flush=True)

    for epoch in range(1, epoch_count + 1):
        train_stats = run_epoch(
            model, train_loader, training_loss, device,
            optimizer=optimizer, scaler=scaler, top5=config.model == "resnet18", max_batches=max_batches,
        )
        val_stats = run_epoch(
            model, val_loader, evaluation_loss, device,
            top5=config.model == "resnet18", max_batches=max_batches,
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_top5_accuracy": train_stats["top5_accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_top5_accuracy": val_stats["top5_accuracy"],
        }
        history.append(row)
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch}/{epoch_count} train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"val loss={row['val_loss']:.4f} acc={row['val_accuracy']:.4f} "
            f"lr={row['learning_rate']:.2e} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if val_stats["accuracy"] > best_accuracy:
            best_accuracy = val_stats["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "validation_metrics": {
                    "loss": val_stats["loss"],
                    "accuracy": val_stats["accuracy"],
                    "top5_accuracy": val_stats["top5_accuracy"],
                },
                "model_name": config.model,
                "class_count": 102,
                "config": asdict(config),
                "environment": info,
                "dataset_sizes": {"train": len(train_data), "val": len(val_data)},
                "parameter_counts": parameter_counts(model),
            }
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
        else:
            stale_epochs += 1
        scheduler.step()
        save_json(
            history_path,
            {
                "model": config.model,
                "config": asdict(config),
                "environment": info,
                "epochs": history,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "training_time_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint_path.relative_to(OUTPUT_DIR.parent)),
                "smoke_test": config.smoke_test,
            },
        )
        if not config.smoke_test and stale_epochs >= config.patience:
            print(f"early stopping after {stale_epochs} epochs without validation improvement", flush=True)
            break

    from .plots import plot_training_history

    figure_dir = (OUTPUT_DIR / "smoke" / config.model) if config.smoke_test else (OUTPUT_DIR / "figures")
    plot_training_history(history, config.model, figure_dir)
    print(f"best epoch={best_epoch} val accuracy={best_accuracy:.4f} checkpoint={checkpoint_path}", flush=True)
    return {"best_epoch": best_epoch, "best_val_accuracy": best_accuracy, "checkpoint": str(checkpoint_path)}
