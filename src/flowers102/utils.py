"""Project paths, local caches, seeding, and JSON output."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def configure_project_caches() -> None:
    """Keep model downloads and temporary files inside this project."""
    temp_dir = PROJECT_ROOT / ".venv" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp_dir)
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".venv" / "mplconfig")
    torch.hub.set_dir(str(DATA_DIR / "torch-hub"))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_info(device: torch.device) -> dict[str, Any]:
    """Return versions and hardware details saved with every experiment."""
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_build": torch.version.cuda,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write human-readable UTF-8 JSON below the project root."""
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Output outside project: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
