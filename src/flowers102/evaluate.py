"""Load the best validation checkpoint and evaluate one official split."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.datasets import Flowers102

from .data import CLASS_COUNT, get_dataset, make_loader
from .engine import run_epoch
from .models import build_model, parameter_counts
from .utils import DATA_DIR, OUTPUT_DIR, environment_info, save_json


def _classification_metrics(targets: list[int], predictions: list[int]) -> tuple[dict[str, float], np.ndarray]:
    """Compute classwise macro scores from aligned targets and predictions."""
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(matrix, (np.asarray(targets), np.asarray(predictions)), 1)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(CLASS_COUNT), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros(CLASS_COUNT), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(CLASS_COUNT), where=precision + recall != 0)
    return {
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }, matrix


def evaluate_checkpoint(
    model_name: str,
    *,
    split: str,
    smoke_test: bool,
    batch_size: int = 32,
    workers: int = 2,
) -> dict[str, Any]:
    """Evaluate a saved state_dict without fitting or selecting on test data."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if smoke_test and split == "test":
        raise ValueError("Smoke checkpoints cannot be evaluated on test")
    checkpoint_path = (
        OUTPUT_DIR / "smoke" / model_name / "best.pt"
        if smoke_test else OUTPUT_DIR / "checkpoints" / f"{model_name}-best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != model_name or checkpoint["class_count"] != CLASS_COUNT:
        raise ValueError("Checkpoint does not match requested model or class count")
    if bool(checkpoint["config"]["smoke_test"]) != smoke_test:
        raise ValueError("Smoke/final checkpoint mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = get_dataset(split)
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False, seed=checkpoint["config"]["seed"])
    started = time.perf_counter()
    stats = run_epoch(
        model, loader, nn.CrossEntropyLoss(), device,
        top5=model_name == "resnet18", collect_predictions=True,
        max_batches=2 if smoke_test else None,
    )
    targets = stats["targets"]
    predictions = stats["predictions"]
    confidences = stats["confidences"]
    assert targets is not None and predictions is not None and confidences is not None
    macro_scores, matrix = _classification_metrics(targets, predictions)
    history_path = (
        OUTPUT_DIR / "smoke" / model_name / "history.json"
        if smoke_test else OUTPUT_DIR / "metrics" / f"{model_name}-history.json"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_row = history["epochs"][checkpoint["epoch"] - 1]
    matrix_no_diagonal = matrix.copy()
    np.fill_diagonal(matrix_no_diagonal, 0)
    top_pairs = sorted(
        (
            (int(matrix_no_diagonal[true, pred]), int(true), int(pred))
            for true, pred in zip(*np.nonzero(matrix_no_diagonal))
        ),
        reverse=True,
    )[:10]
    result: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "smoke_test": smoke_test,
        "best_epoch": checkpoint["epoch"],
        "train_accuracy": best_row["train_accuracy"],
        "val_accuracy": checkpoint["validation_metrics"]["accuracy"],
        "val_loss": checkpoint["validation_metrics"]["loss"],
        "test_loss": stats["loss"] if split == "test" else None,
        "test_accuracy": stats["accuracy"] if split == "test" else None,
        "test_top5_accuracy": stats["top5_accuracy"] if split == "test" else None,
        "validation_check_accuracy": stats["accuracy"] if split == "val" else None,
        "macro_precision": macro_scores["macro_precision"] if split == "test" else None,
        "macro_recall": macro_scores["macro_recall"] if split == "test" else None,
        "macro_f1": macro_scores["macro_f1"] if split == "test" else None,
        "parameter_count": parameter_counts(model)["parameter_count"],
        "trainable_parameter_count": parameter_counts(model)["trainable_parameter_count"],
        "training_time_seconds": history["training_time_seconds"],
        "evaluation_time_seconds": time.perf_counter() - started,
        "evaluated_samples": stats["sample_count"],
        "checkpoint": str(checkpoint_path.relative_to(OUTPUT_DIR.parent)),
        "config": checkpoint["config"],
        "environment": environment_info(device),
        "top_confusions": [
            {
                "count": count,
                "true_class": true,
                "true_name": Flowers102.classes[true],
                "predicted_class": pred,
                "predicted_name": Flowers102.classes[pred],
            }
            for count, true, pred in top_pairs
        ],
    }
    if smoke_test:
        output_path = OUTPUT_DIR / "smoke" / model_name / "evaluation-val.json"
    else:
        from .plots import plot_confusion_matrix, plot_sample_predictions

        figures = OUTPUT_DIR / "figures" if split == "test" else OUTPUT_DIR / "validation"
        confusion_path = plot_confusion_matrix(matrix, model_name, figures, split=split)
        raw_dataset = Flowers102(root=DATA_DIR, split=split)
        prediction_path, examples = plot_sample_predictions(
            raw_dataset, targets, predictions, confidences, model_name, figures
        )
        result["figures"] = [
            str(confusion_path.relative_to(OUTPUT_DIR.parent)),
            str(prediction_path.relative_to(OUTPUT_DIR.parent)),
        ]
        result["prediction_examples"] = examples
        output_path = (
            OUTPUT_DIR / "metrics" / f"{model_name}.json"
            if split == "test" else OUTPUT_DIR / "validation" / f"{model_name}.json"
        )
    save_json(output_path, result)
    print(
        f"{model_name} {split} loss={stats['loss']:.4f} top1={stats['accuracy']:.4f} "
        f"top5={stats['top5_accuracy']} samples={stats['sample_count']} saved={output_path}",
        flush=True,
    )
    return result
