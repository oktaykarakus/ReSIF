import os
from typing import Callable, Dict, Optional
import glob

import numpy as np
import rasterio
import torch

from PIL import Image
from torch import Tensor

from .geo import VisionDataset
from .utils import (
    check_integrity,
    draw_semantic_segmentation_masks,
    extract_archive,
    rgb_to_mask,
)

s2_name = "s2"
planet_name = "planet"
dem_name = "dem"
wind_name = "wind"
windthrow_name = "ann"


class Passau_quad(VisionDataset):

    colormap = [
        (255, 255, 255),  # damage
        (0, 0, 0)       # no damage
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new dataset instance.

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
        self.s2_root = "s2_dir"
        self.planet_root = "planet_dir"
        self.dem_root = "dem_dir"
        self.wind_root = "wind_dir"
        self.mask_root = "ann_dir"
        self._verify()

        self.files = []
        if split == 'train':
            s2 = glob.glob(
                os.path.join(self.root, self.s2_root, "train", "*.tif"), recursive=False
            )

            for s2_image in sorted(s2):
                mask = s2_image.replace(self.s2_root, self.mask_root)
                mask = mask.replace(s2_name, windthrow_name)
                planet = s2_image.replace(self.s2_root, self.planet_root)
                planet = planet.replace(s2_name, planet_name)
                dem = s2_image.replace(self.s2_root, self.dem_root)
                dem = dem.replace(s2_name, dem_name)
                wind = s2_image.replace(self.s2_root, self.wind_root)
                wind = wind.replace(s2_name, wind_name)
                self.files.append(dict(s2=s2_image, planet=planet, dem=dem, wind=wind, mask=mask))
        elif split == "test":
            s2 = glob.glob(
                os.path.join(self.root, self.s2_root, "test", "*.tif"), recursive=False
            )

            for s2_image in sorted(s2):
                mask = s2_image.replace(self.s2_root, self.mask_root)
                mask = mask.replace(s2_name, windthrow_name)
                planet = s2_image.replace(self.s2_root, self.planet_root)
                planet = planet.replace(s2_name, planet_name)
                dem = s2_image.replace(self.s2_root, self.dem_root)
                dem = dem.replace(s2_name, dem_name)
                wind = s2_image.replace(self.s2_root, self.wind_root)
                wind = wind.replace(s2_name, wind_name)
                self.files.append(dict(s2=s2_image, planet=planet, dem=dem, wind=wind, mask=mask))
        else:
            s2 = glob.glob(
                os.path.join(self.root, self.s2_root, "val", "*.tif"), recursive=False
            )

            for s2_image in sorted(s2):
                mask = s2_image.replace(self.s2_root, self.mask_root)
                mask = mask.replace(s2_name, windthrow_name)
                planet = s2_image.replace(self.s2_root, self.planet_root)
                planet = planet.replace(s2_name, planet_name)
                dem = s2_image.replace(self.s2_root, self.dem_root)
                dem = dem.replace(s2_name, dem_name)
                wind = s2_image.replace(self.s2_root, self.wind_root)
                wind = wind.replace(s2_name, wind_name)
                self.files.append(dict(s2=s2_image, planet=planet, dem=dem, wind=wind, mask=mask))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        s2 = self._load_image_s2(index)
        planet = self._load_image_planet(index)
        dem = self._load_dem_patch(index)
        wind = self._load_wind_patch(index)
        mask = self._load_target(index)

        sample = {"modality1": s2, "modality2": planet, "modality3": dem, "modality4": wind, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

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

    def _load_image_planet(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["planet"]
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_dtype="float32"
            )
            # array = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_dem_patch(self, index: int) -> Tensor:
        """Load a single DEM patch.

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

    def _load_wind_patch(self, index: int) -> Tensor:
        """Load a single wind data patch.

        Args:
            index: index to return

        Returns:
            the image
        """
        path = self.files[index]["wind"]
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

        # TODO: adjust target data format

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
