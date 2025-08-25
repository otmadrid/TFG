import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal


import albumentations.augmentations.functional as AF
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
from einops import einsum, pack
from PIL import Image
from torchvision.datasets import VisionDataset
from albumentations.augmentations.crops.transforms import RandomCrop
#from .serialize import TorchSerializedList
import albumentations as A
import imageio
import os
#type AlbuTransform = BasicTransform | Compose

class SyntheticUIBSelection(ABC, VisionDataset):
    def __init__(self,
                 root: str,
                 split: Literal["train", "val"],
                 name_noisy: str,
                 dims_crop: list,
                 do_crop: bool = True,):
        

        self.transform_crop = A.Compose([RandomCrop(dims_crop[0], dims_crop[1], p=1)],
            additional_targets={'image_noisy': 'image'}
            )
        self.dim_crop = dims_crop
        self.do_crop = do_crop  
        self._to_tensor = ToTensorV2()
        self.split = split
        root_data = Path(root) / split
        gt_list = sorted(root_data.iterdir())
        self._gt_list = gt_list
        self.name_noisy = name_noisy    

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = np.array(Image.open(str(file_name)).convert("RGB"), dtype=np.float32) / 255.0
        img_noisy = np.array(Image.open(str(file_name).replace(self.split, self.split + self.name_noisy)).convert("RGB"),
                dtype=np.float32) / 255.0

            # Afegim un print per verificar els valors mínims i màxims
        #print(f"GT min/max: {img.min()}/{img.max()}, dtype: {img.dtype}")
        #print(f"Noisy min/max: {img_noisy.min()}/{img_noisy.max()}, dtype: {img_noisy.dtype}")

        return img, img_noisy

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number, original_dataset, orig_name = file_name.stem.split(
            "_", maxsplit=3)
        return {"dataset": "UIBSelection",
                "scene": original_dataset,
                "frame": int(frame_number)}

    def __len__(self) -> int:
        return len(self._gt_list)
    

    def __getitem__(self, index: int):
        image_path = self._gt_list[index]
        img_gt, img_noisy = self._read_image(image_path)
        metadata = self._extract_metadata(image_path)
        width_im, height_im = img_gt.shape[:2]

        if self.do_crop and width_im >= self.dim_crop[0] and height_im >= self.dim_crop[1]:
            transformed = self.transform_crop(image=img_gt, image_noisy=img_noisy)
            img_gt = transformed["image"]
            img_noisy = transformed["image_noisy"]
        return img_noisy, img_gt


    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:04g}"

def save_image(images, path, name):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = np.clip(images[0], 0, 1)
    images = (images * 255).astype(np.uint8)
    
    imageio.imwrite(os.path.join(path, f"{name}.png"), images)
