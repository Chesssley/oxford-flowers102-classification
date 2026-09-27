"""Report-ready training curves and held-out evaluation figures."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import Flowers102


def plot_training_history(history: list[dict[str, Any]], model_name: str, folder: Path) -> None:
    """Save train/validation loss and top-1 curves from recorded epochs."""
    folder.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    for measure, ylabel, filename in (
        ("loss", "Loss (train smoothed; val standard)", "training-loss"),
        ("accuracy", "Top-1 accuracy", "training-accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [row[f"train_{measure}"] for row in history], marker="o", label="Train")
        ax.plot(epochs, [row[f"val_{measure}"] for row in history], marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{model_name}: {ylabel}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"{model_name}-{filename}.png", dpi=180)
        plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, model_name: str, folder: Path, *, split: str) -> Path:
    """Render a readable 102-class confusion matrix with numeric class IDs."""
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    ticks = np.arange(0, matrix.shape[0], 5)
    ax.set_xticks(ticks, ticks, fontsize=7, rotation=90)
    ax.set_yticks(ticks, ticks, fontsize=7)
    ax.set(xlabel="Predicted class ID (0–101)", ylabel="True class ID (0–101)", title=f"{model_name}: {split} confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Images")
    fig.tight_layout()
    path = folder / f"{model_name}-confusion-matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sample_predictions(
    dataset: Flowers102,
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    model_name: str,
    folder: Path,
    *,
    seed: int = 42,
) -> tuple[Path, list[dict[str, Any]]]:
    """Show fixed held-out examples with labels, predictions, and confidence."""
    if not (len(dataset) == len(targets) == len(predictions) == len(confidences)):
        raise ValueError("Prediction arrays must align with dataset order")
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    incorrect = [index for index, (true, pred) in enumerate(zip(targets, predictions)) if true != pred]
    correct = [index for index, (true, pred) in enumerate(zip(targets, predictions)) if true == pred]
    selected = rng.sample(incorrect, min(6, len(incorrect)))
    selected += rng.sample(correct, min(12 - len(selected), len(correct)))
    if len(selected) < 12:
        remainder = [index for index in range(len(dataset)) if index not in selected]
        selected += rng.sample(remainder, min(12 - len(selected), len(remainder)))
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    examples = []
    for ax, index in zip(axes.flat, selected):
        image, label = dataset[index]
        assert label == targets[index]
        prediction = predictions[index]
        ax.imshow(image)
        ax.set_title(
            f"GT {label}: {Flowers102.classes[label]}\n"
            f"Pred {prediction}: {Flowers102.classes[prediction]}\n"
            f"Confidence {confidences[index]:.1%}",
            fontsize=8,
            color="green" if label == prediction else "darkred",
        )
        ax.axis("off")
        examples.append(
            {
                "index": index,
                "ground_truth": label,
                "ground_truth_name": Flowers102.classes[label],
                "prediction": prediction,
                "prediction_name": Flowers102.classes[prediction],
                "confidence": confidences[index],
                "correct": label == prediction,
            }
        )
    fig.tight_layout()
    path = folder / f"{model_name}-sample-predictions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, examples
