"""Download and validate the official Oxford Flowers102 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowers102.utils import OUTPUT_DIR, configure_project_caches, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download into project data/ if absent")
    args = parser.parse_args()
    configure_project_caches()
    from flowers102.data import inspect_official_data

    result = inspect_official_data(download=args.download)
    save_json(OUTPUT_DIR / "metrics" / "dataset.json", result)
    for split, detail in result["splits"].items():
        print(f"{split}: {detail['count']} images, {detail['class_count']} classes, labels {detail['label_min']}..{detail['label_max']}")
    print(f"total: {result['total_images']}, disjoint: {result['disjoint_splits']}")


if __name__ == "__main__":
    main()
