"""Official Flowers102 splits and ImageNet-compatible image transforms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import ResNet18_Weights

from .utils import DATA_DIR

CLASS_COUNT = 102
IMAGE_SIZE = 224
RESIZE_SIZE = 256
SPLITS = ("train", "val", "test")
SPLIT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}


def image_transforms(training: bool) -> transforms.Compose:
    """Use the ImageNet normalization required by the ResNet18 weights."""
    weights_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    normalize = transforms.Normalize(
        mean=weights_transform.mean,
        std=weights_transform.std,
    )
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_dataset(split: str, *, download: bool = False) -> Flowers102:
    """Construct one unmodified official Flowers102 split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return Flowers102(
        root=DATA_DIR,
        split=split,
        transform=image_transforms(split == "train"),
        download=download,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Flowers102,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic loader with optional Windows worker processes."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None,
        generator=generator,
    )


def inspect_official_data(*, download: bool, seed: int = 42) -> dict[str, Any]:
    """Check official IDs, labels, image files, and transformed samples."""
    datasets = {
        split: get_dataset(split, download=download and split == "train")
        for split in SPLITS
    }
    base = DATA_DIR / "flowers-102"
    set_ids = loadmat(base / "setid.mat", squeeze_me=True)
    one_based_labels = loadmat(base / "imagelabels.mat", squeeze_me=True)["labels"]
    labels = np.asarray(one_based_labels, dtype=np.int64) - 1
    assert len(Flowers102.classes) == CLASS_COUNT
    assert np.array_equal(np.unique(labels), np.arange(CLASS_COUNT))

    split_ids: dict[str, set[int]] = {}
    details: dict[str, Any] = {}
    rng = random.Random(seed)
    for split, dataset in datasets.items():
        ids = np.atleast_1d(set_ids[SPLIT_KEYS[split]]).astype(np.int64)
        split_ids[split] = set(ids.tolist())
        split_labels = labels[ids - 1]
        assert len(dataset) == len(ids)
        assert split_labels.min() == 0 and split_labels.max() == CLASS_COUNT - 1
        assert len(np.unique(split_labels)) == CLASS_COUNT
        missing = [int(i) for i in ids if not (base / "jpg" / f"image_{i:05d}.jpg").is_file()]
        assert not missing, f"Missing {split} images: {missing[:5]}"
        samples = []
        for index in rng.sample(range(len(dataset)), min(3, len(dataset))):
            image, label = dataset[index]
            assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
            assert image.dtype == torch.float32 and torch.isfinite(image).all()
            assert label == int(split_labels[index])
            samples.append({"index": index, "label": label, "shape": list(image.shape), "dtype": str(image.dtype)})
        details[split] = {
            "count": len(dataset),
            "class_count": len(np.unique(split_labels)),
            "label_min": int(split_labels.min()),
            "label_max": int(split_labels.max()),
            "samples": samples,
        }
    assert all(
        split_ids[first].isdisjoint(split_ids[second])
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    assert len(set.union(*split_ids.values())) == len(labels)
    return {
        "source": "torchvision.datasets.Flowers102 / Oxford official setid.mat",
        "total_images": len(labels),
        "class_count": CLASS_COUNT,
        "splits": details,
        "disjoint_splits": True,
    }
