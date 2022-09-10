
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

from .geo import VisionDataset
from .utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)
DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    6,  # 12 --> 6
    7,  # 13 --> 7
    6,  # 14 --> 6
    8,
    9,
    10
    ]
class DFC20_dual_2D(VisionDataset):

    colormap = [
        (0, 0, 0),
        (0, 153, 0),  # Forest
        (198, 176, 68),  # Shrubland
        (251, 255, 19),  # Savanna
        (182, 255, 5),  # Grassland
        (39, 255, 135),  # Wetlands
        (194, 79, 68),  # Croplands
        (165, 165, 165),  # Urban/Built-up
        (105, 255, 248),  # Snow/Ice
        (249, 255, 164),  # Barren
        (28, 13, 255),  # Water
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
        self.splits=["train", "val", "test"]
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.s1_root = "s1_dir"
        self.s2_root = "s2_dir"
        self.mask_root = "ann_dir"
        self._verify()

        self.files = []
        if split == 'train':
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "train", "**", "*.tif"), recursive=False
            )
            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "lc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                self.files.append(dict(s1=image, s2=s2, mask=mask))
        elif split == "test":
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "test", "*.tif"), recursive=False
            )
            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "lc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                self.files.append(dict(s1=image, s2=s2, mask=mask))
        else:
            s1 = glob.glob(
                os.path.join(self.root, self.s1_root, "val", "*.tif"), recursive=False
            )
            for image in sorted(s1):
                mask = image.replace(self.s1_root, self.mask_root)
                mask = mask.replace("s1", "dfc")
                s2 = image.replace(self.s1_root, self.s2_root)
                s2 = s2.replace("s1", "s2")
                self.files.append(dict(s1=image, s2=s2, mask=mask))
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        s1 = self._load_image_s1(index)
        s2 = self._load_image_s2(index)
        # image = torch.cat(tensors=[s1, s2], dim=0)
        mask = self._load_target(index)
        sample = {"modality1": s2, "modality2": s1, "mask": mask}

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
        path = self.files[index]["s1"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            s1 = np.nan_to_num(array)
            s1 = np.clip(s1, -25, 0)
            s1 /= 25
            s1 += 1
            s1 = s1.astype(np.float32)
            # array = f.read()
            tensor: Tensor = torch.from_numpy(s1)  # type: ignore[attr-defined]
            return tensor

    def _load_image_s2(self, index: int) -> Tensor:
        path = self.files[index]["s2"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(out_dtype="float32")
            s2 = np.clip(array, 0, 10000)
            s2 /= 10000
            s2 = s2.astype(np.float32)
            tensor: Tensor = torch.from_numpy(s2)  # type: ignore[attr-defined]
            return tensor

    def _load_target(self, index: int) -> Tensor:
        path = self.files[index]["mask"]
        with rasterio.open(path) as data:
            lc = data.read(1)
            # print(lc.max())
            if self.split == 'train' or self.split == 'test':
                lc = np.take(DFC2020_CLASSES, lc)
            else:
                lc = lc.astype(np.int64)

            # adjust class scheme to ignore class savanna
            # if no_savanna:
            #     lc[lc == 3] = 0
            #     lc[lc > 3] -= 1

            # convert to zero-based labels and set ignore mask
            lc -= 1
            lc[lc == -1] = 255
            tensor: Tensor = torch.from_numpy(lc)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
        return tensor

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
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        cmap = np.zeros((256, 3), dtype='uint8')
        for i, c in enumerate(cls.colormap):
            cmap[i]=np.array(list(c))
        return cmap[mask]