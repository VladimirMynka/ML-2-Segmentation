import os
import random
import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms_t
import cv2


class ImageDataset(Dataset):
    def __init__(
        self,
        img_path: os.PathLike | str,
        mask_path: os.PathLike | str,
        transforms: torch.nn.Module,
        affine: t.Tuple[int, int, bool, bool] = None,
        image_size: t.Tuple = (384, 384),
    ):
        """
        :param img_path: Path to folder with source images
        :param mask_path: Path with masks. Each of masks must be named as image in `img_path`
        :param transforms: transforms for transforming images
        :param affine: set of params for affine transformation. None if affine transformation is not needed. Params in the next order: rotate_max, move_max, vertical_flip, horizontal_flip
        :param image_size: size of image for resizing
        """
        self.horizontal_flip = False
        self.vertical_flip = False
        self.move_Y = 0
        self.move_X = 0
        self.rotate = 0

        self.affine = affine
        self.img_path = img_path
        self.mask_path = mask_path
        self.image_size = image_size
        self.transforms = transforms

        self.names = os.listdir(img_path)

    def read_file(self, path):
        file = cv2.imread(path)[:, :, ::-1]
        file = cv2.resize(file, self.image_size, interpolation=cv2.INTER_NEAREST)
        return file

    def apply_random_affine(self, image: torch.tensor, update_random=True):
        pass
        if self.affine is None:
            return image
        if update_random:
            self.update_affine()
        image = transforms_t.affine(
            image,
            angle=self.rotate,
            translate=[self.move_X, self.move_Y],
            scale=1,
            shear=[0, 0]
        )
        return image

    def update_affine(self):
        rotate_max, move_max, vertical_flip, horizontal_flip = self.affine
        self.rotate = random.randint(-rotate_max, rotate_max)
        self.move_X = random.randint(-move_max, move_max)
        self.move_Y = random.randint(-move_max, move_max)
        self.vertical_flip = (random.random() > 0.5) and vertical_flip
        self.horizontal_flip = (random.random() > 0.5) and horizontal_flip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, self.names[index])
        image = self.read_file(image_path)
        image = self.transforms(image)
        image = self.apply_random_affine(image, update_random=True)

        mask_path = os.path.join(self.mask_path, self.names[index])
        gt_mask = self.read_file(mask_path).astype(np.int32)
        _mask = np.zeros((*self.image_size, 2), dtype=np.float32)

        # BACKGROUND
        _mask[:, :, 0] = np.where(gt_mask[:, :, 0] == 0, 1.0, 0.0)
        # CLOD
        _mask[:, :, 1] = np.where(gt_mask[:, :, 0] == 255, 1.0, 0.0)
        mask = torch.from_numpy(_mask).permute(2, 0, 1)
        mask = self.apply_random_affine(mask, update_random=False)

        return image, mask
