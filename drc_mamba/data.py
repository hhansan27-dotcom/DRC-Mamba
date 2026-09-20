"""Dataset utilities for infrared small target detection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


def read_split(dataset_dir: str | Path, split_name: str) -> tuple[list[str], list[str]]:
    """Read ``train.txt`` and ``test.txt`` from a dataset split directory."""
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split_name
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"
    if not train_file.is_file() or not test_file.is_file():
        raise FileNotFoundError(
            f"Expected split files at {train_file} and {test_file}. "
            "See dataset/README.md for the required layout."
        )
    train_ids = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]
    test_ids = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]
    return train_ids, test_ids


def _image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    array = array / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _mask_to_tensor(mask: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    # IRSTD masks appear as either {0,1} or {0,255}; thresholding handles both.
    array = (array > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class TrainDataset(Dataset):
    """Training dataset with the augmentation used by the original experiments.

    Images remain single-channel. Random scale, horizontal flip, crop, padding, and
    optional Gaussian blur are applied synchronously to image and mask where needed.
    ``cache_images`` only changes I/O behavior; it does not change augmentation.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        image_ids: Sequence[str],
        base_size: int = 512,
        crop_size: int = 512,
        suffix: str = ".png",
        cache_images: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.masks_dir = self.dataset_dir / "masks"
        self.image_ids = list(image_ids)
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)
        self.suffix = suffix
        self.cache_images = bool(cache_images)
        self._cache: dict[str, tuple[Image.Image, Image.Image]] = {}

        if self.cache_images:
            for image_id in self.image_ids:
                self._cache[image_id] = self._load_pair(image_id)

    def _load_pair(self, image_id: str) -> tuple[Image.Image, Image.Image]:
        image_path = self.images_dir / f"{image_id}{self.suffix}"
        mask_path = self.masks_dir / f"{image_id}{self.suffix}"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask pair: {image_path}, {mask_path}")
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _get_pair(self, image_id: str) -> tuple[Image.Image, Image.Image]:
        if self.cache_images:
            image, mask = self._cache[image_id]
            return image.copy(), mask.copy()
        return self._load_pair(image_id)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            out_h = long_size
            out_w = int(width * long_size / height + 0.5)
        else:
            out_w = long_size
            out_h = int(height * long_size / width + 0.5)

        image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
        mask = mask.resize((out_w, out_h), Image.Resampling.NEAREST)

        pad_w = max(0, self.crop_size - out_w)
        pad_h = max(0, self.crop_size - out_h)
        if pad_w or pad_h:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        width, height = image.size
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        image = image.crop(box)
        mask = mask.crop(box)

        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        image, mask = self._get_pair(image_id)
        image, mask = self._augment(image, mask)
        return _image_to_tensor(image), _mask_to_tensor(mask)

    def __len__(self) -> int:
        return len(self.image_ids)


class EvalDataset(Dataset):
    """Evaluation dataset returning both resized and original-resolution masks."""

    def __init__(
        self,
        dataset_dir: str | Path,
        image_ids: Sequence[str],
        input_size: int = 512,
        suffix: str = ".png",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.masks_dir = self.dataset_dir / "masks"
        self.image_ids = list(image_ids)
        self.input_size = int(input_size)
        self.suffix = suffix

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = self.images_dir / f"{image_id}{self.suffix}"
        mask_path = self.masks_dir / f"{image_id}{self.suffix}"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask pair: {image_path}, {mask_path}")

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        original_width, original_height = image.size

        resized_image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        resized_mask = mask.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)

        return (
            _image_to_tensor(resized_image),
            _mask_to_tensor(resized_mask),
            torch.tensor([original_height, original_width], dtype=torch.int64),
            image_id,
            _mask_to_tensor(mask),
        )

    def __len__(self) -> int:
        return len(self.image_ids)
