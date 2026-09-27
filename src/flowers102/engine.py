"""Shared training and evaluation epoch logic."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    top5: bool = False,
    collect_predictions: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run an epoch and return sample-weighted loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    correct5 = 0
    sample_count = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        loss_sum += loss.detach().item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        if top5:
            top_indices = logits.topk(5, dim=1).indices
            correct5 += int(top_indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item())
        sample_count += batch_size
        if collect_predictions:
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            confidences_all.extend(logits.softmax(dim=1).max(dim=1).values.cpu().tolist())

    if sample_count == 0:
        raise ValueError("The loader yielded no samples")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "top5_accuracy": correct5 / sample_count if top5 else None,
        "sample_count": sample_count,
        "targets": targets_all if collect_predictions else None,
        "predictions": predictions_all if collect_predictions else None,
        "confidences": confidences_all if collect_predictions else None,
    }
