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

from .geo import VisionDataset
from .utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)


class Potsdam2D(VisionDataset):
    """Potsdam 2D Semantic Segmentation dataset.

    The `Potsdam <https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/>`_
    dataset is a dataset for urban semantic segmentation used in the 2D Semantic Labeling
    Contest - Potsdam. This dataset uses the "4_Ortho_RGBIR.zip" and "5_Labels_all.zip"
    files to create the train/test sets used in the challenge. The dataset can be
    requested at the challenge homepage. Note, the server contains additional data
    for 3D Semantic Labeling which are currently not supported.

    Dataset format:

    * images are 4-channel geotiffs
    * masks are 3-channel geotiffs with unique RGB values representing the class

    Dataset classes:

    0. Clutter/background
    1. Impervious surfaces
    2. Building
    3. Low Vegetation
    4. Tree
    5. Car

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprsannals-I-3-293-2012

    .. versionadded:: 0.2
    """  # noqa: E501

    filenames = ["4_Ortho_RGBIR.zip", "5_Labels_all.zip"]
    md5s = ["c4a8f7d8c7196dd4eba4addd0aae10c1", "cf7403c1a97c0d279414db"]


    colormap = [
        (255, 0, 0),
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        num_tra_sam: str = "train",
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
        self.splits=["train", "val", "test", "unlabeled_train"]
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.image_root = "img_dir"
        self.mask_root = "ann_dir"
        self.dsm_root = "dsm_dir"
        self.num_tra_sam = num_tra_sam
        self._verify()

        self.files = []
        self.unl_files = []
        if split == 'train':
            images = glob.glob(
                os.path.join(self.root, self.image_root, self.num_tra_sam, "*.tif"), recursive=False
            )
            for image in sorted(images):
                mask = image.replace(self.image_root, self.mask_root)
                mask = mask.replace("RGBIR", "label")
                dsm = image.replace(self.image_root, self.dsm_root)
                dsm = dsm.replace("RGBIR", "dsm")
                self.files.append(dict(image=image, mask=mask, dsm=dsm))
        elif split == 'unlabeled_train':
            images = glob.glob(
                os.path.join(self.root, self.image_root, self.num_tra_sam, "*.tif"), recursive=False
            )
            unl_images = glob.glob(
                os.path.join(self.root, self.image_root, "train", "*.tif"), recursive=False
            )
            images_name = [item.split('/')[-1] for item in images]
            for image in sorted(unl_images):
                if image.split('/')[-1] not in images_name:
                    # mask = image.replace(self.image_root, self.mask_root)
                    # mask = mask.replace("RGBIR", "label")
                    dsm = image.replace(self.image_root, self.dsm_root)
                    dsm = dsm.replace("RGBIR", "dsm")
                    self.files.append(dict(image=image, dsm=dsm))
        elif split == "test":
            images = glob.glob(
                os.path.join(self.root, self.image_root, "test", "*.tif"), recursive=False
            )
            for image in sorted(images):
                mask = image.replace(self.image_root, self.mask_root)
                mask = mask.replace("RGBIR", "label")
                dsm = image.replace(self.image_root, self.dsm_root)
                dsm = dsm.replace("RGBIR", "dsm")
                self.files.append(dict(image=image, mask=mask, dsm=dsm))
        else:
            images = glob.glob(
                os.path.join(self.root, self.image_root, "val", "*.tif"), recursive=False
            )
            for image in sorted(images):
                mask = image.replace(self.image_root, self.mask_root)
                mask = mask.replace("RGBIR", "label")
                dsm = image.replace(self.image_root, self.dsm_root)
                dsm = dsm.replace("RGBIR", "dsm")
                self.files.append(dict(image=image, mask=mask, dsm=dsm))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        dsm_image = self._load_dsm_image(index)
        if self.split == "unlabeled_train":
            image = torch.cat(tensors=[image, dsm_image], dim=0)
            sample = {"image": image}
        else:
            mask = self._load_target(index)
            image = torch.cat(tensors=[image, dsm_image], dim=0)
            sample = {"image": image, "label": mask}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["image"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            # array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_dsm_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """ 
        path = self.files[index]["dsm"]
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
        path = self.files[index]["mask"]
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.uint8]" = np.array(img.convert("RGB"))
            array = rgb_to_mask(array, self.colormap)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
        return tensor
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        cmap = np.zeros((256, 3), dtype='uint8')
        for i, c in enumerate(cls.colormap):
            cmap[i]=np.array(list(c))
        return cmap[mask]
    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.image_root)):
            return

        # Check if .zip files already exists (if so extract)
        exists = []
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 1
        image1 = draw_semantic_segmentation_masks(
            sample["image"][:3], sample["mask"], alpha=alpha, colors=self.colormap
        )
        if "prediction" in sample:
            ncols += 1
            image2 = draw_semantic_segmentation_masks(
                sample["image"][:3],
                sample["prediction"],
                alpha=alpha,
                colors=self.colormap,
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols > 1:
            (ax0, ax1) = axs
        else:
            ax0 = axs

        ax0.imshow(image1)
        ax0.axis("off")
        if ncols > 1:
            ax1.imshow(image2)
            ax1.axis("off")

        if show_titles:
            ax0.set_title("Ground Truth")
            if ncols > 1:
                ax1.set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
