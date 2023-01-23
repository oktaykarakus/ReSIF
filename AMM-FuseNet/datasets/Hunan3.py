# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Potsdam dataset."""

import os
from typing import Callable, Dict, Optional
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from einops import rearrange
from .geo import VisionDataset

from .utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)


class Hunan3(VisionDataset):

    colormap = [
        (197, 90, 17),  # Cropland
        (51, 130, 88),  # Forest
        (178, 206, 61),  # Grassland
        (229, 84, 96),  # Wetland
        (91, 155, 215),  # Water
        (240, 160, 2),  # bare land
        (226, 175, 110),  # others
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Potsdam dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.splits = ["train", "val", "test"]
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.s1_root = "s1_dir"
        self.s2_root = "s2_dir"
        self.dem_root = "dem_dir"
        self.mask_root = "ann_dir"
        self._verify()

        self.files = []
        if split == 'train':
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "train", "*.tif"), recursive=False
            )

            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "lc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                dem = image.replace(self.s1_root, self.dem_root)
                dem = dem.replace("s1", "topo")
                self.files.append(dict(s1=image, s2=s2, dem=dem, mask=mask))
        elif split == "test":
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "test", "*.tif"), recursive=False
            )

            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "lc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                dem = image.replace(self.s1_root, self.dem_root)
                dem = dem.replace("s1", "topo")
                self.files.append(dict(s1=image, s2=s2, dem=dem, mask=mask))
        else:
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "val", "*.tif"), recursive=False
            )

            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "lc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                dem = image.replace(self.s1_root, self.dem_root)
                dem = dem.replace("s1", "topo")
                self.files.append(dict(s1=image, s2=s2, dem=dem, mask=mask))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        s1 = self._load_image_s1(index)
        s2 = self._load_image_s2(index)
        dem = self._load_image_dem(index)
        dem = rearrange(dem[0], " h w -> () h w")
        # image = torch.cat(tensors=[s1, s2], dim=0)
        mask = self._load_target(index)
        sample = {"modality1": s2, "modality2": s1, "modality3": dem, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image_s1(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["s1"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            # array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_image_s2(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["s2"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            # array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_image_dem(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["dem"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            # array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """

        # with rasterio.open(path) as f:
        #     array: "np.typing.NDArray[np.int_]" = f.read(
        #         indexes=1, out_dtype="int32", resampling=Resampling.bilinear
        #     )
        #     tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        #     tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
        #     return tensor

        igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])
        path = self.files[index]["mask"]
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.uint8]" = np.array(img)
            array[array == 255] = 12
            array = igbp2hunan[array]
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
        return tensor
        # with rasterio.open(path) as img:
        #     array: "np.typing.NDArray[np.int_]" = img.read(
        #                  out_dtype="int32"
        #             )
        #     # array = rgb_to_mask(array, self.colormap)
        #     tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        #     # Convert from HxWxC to CxHxW
        #     tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
        # return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.s1_root)):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different" +
            " `root` directory or manually download the dataset to this directory."
        )

    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        cmap = np.zeros((256, 3), dtype='uint8')
        for i, c in enumerate(cls.colormap):
            cmap[i] = np.array(list(c))
        return cmap[mask]
