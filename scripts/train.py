"""Train a supported model on official Flowers102 train/val splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowers102.utils import configure_project_caches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple-cnn", "resnet18", "mobilenetv3-small"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    defaults = {"simple-cnn": (20, 1e-3), "resnet18": (15, 1e-4), "mobilenetv3-small": (15, 1e-4)}
    default_epochs, default_lr = defaults[args.model]
    epochs = default_epochs if args.epochs is None else args.epochs
    learning_rate = default_lr if args.learning_rate is None else args.learning_rate
    if args.batch_size < 1 or args.workers < 0 or args.patience < 1 or epochs < 1:
        parser.error("batch-size, patience and epochs must be positive; workers cannot be negative")
    if learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        parser.error("learning-rate must be positive, weight-decay nonnegative, label-smoothing in [0, 1)")
    configure_project_caches()
    from flowers102.train import TrainConfig, train_experiment

    config = TrainConfig(
        model=args.model,
        epochs=epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    train_experiment(config)


if __name__ == "__main__":
    main()
