"""Evaluate the best saved checkpoint on validation or held-out test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowers102.utils import configure_project_caches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple-cnn", "resnet18"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive; workers cannot be negative")
    configure_project_caches()
    from flowers102.evaluate import evaluate_checkpoint

    evaluate_checkpoint(
        args.model, split=args.split, smoke_test=args.smoke_test,
        batch_size=args.batch_size, workers=args.workers,
    )


if __name__ == "__main__":
    main()
