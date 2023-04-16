import os

import cv2
import typing
from scipy.ndimage import label
import torch
import numpy as np

from src.core.data_utilities.transforms import get_common_transforms
from src.core.models.model import Model
from src.config_and_utils.config import TransformsConfig, ModelConfig


class Predictor:
    def __init__(
        self,
        model: Model,
        size_threshold: float,
        transforms_config: TransformsConfig = None
    ):
        self.model = model.eval()
        self.threshold = size_threshold
        self.transforms_config = transforms_config if transforms_config is not None else TransformsConfig()
        self.image_size = self.transforms_config.image_size

    def get_mask(self, image: np.uint8):
        preprocess_transforms = get_common_transforms(self.transforms_config)

        image_model_np = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)

        image_model = preprocess_transforms(image_model_np)
        image_model = torch.unsqueeze(image_model, dim=0)

        with torch.no_grad():
            out = self.model(image_model)["out"].cpu()

        return out.numpy().astype(np.uint8), image_model_np

    def predict_count_and_the_biggest_size(self, image: np.uint8):
        masks, _ = self.get_mask(image)  # (1, 2, im_size, im_size)
        mask = masks[0, 1]
        mask = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2RGB)
        mask[mask > 100] = 255
        mask[mask <= 100] = 0
        return self.calculate_count_and_the_biggest_size_by_mask(mask)

    def calculate_count_and_the_biggest_size_by_mask(self, mask: np.uint8, resize: bool = False):
        if resize:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        labeled_array, num_labels = self.get_areas(mask)
        sizes, the_biggest = self.get_sizes(labeled_array, num_labels)
        the_biggest_size = sizes[the_biggest]
        count = len([size for size in sizes if size > self.threshold])

        return count, the_biggest_size

    @staticmethod
    def get_areas(mask: np.uint8) -> typing.Tuple[np.ndarray, int]:
        return label(mask)

    @staticmethod
    def get_areas_and_bboxes(mask: np.uint8):
        labeled_array, num_labels = label(mask)
        # loop through each labeled region and get the bounding box coordinates
        bbox_list = []
        for i in range(1, num_labels + 1):
            # get the coordinates of all pixels in the current labeled region
            coordinates = np.where(labeled_array == i)
            # get the minimum and maximum coordinates for each dimension
            min_row, min_col, _ = np.min(coordinates, axis=1)
            max_row, max_col, _ = np.max(coordinates, axis=1)
            # add the bounding box coordinates to the list
            bbox_list.append((min_col, min_row, max_col, max_row))

        return bbox_list, labeled_array

    @staticmethod
    def get_sizes(labeled_array, num_labels):
        sizes = []
        for i in range(1, num_labels + 1):
            sh = labeled_array.shape
            size = np.sum(labeled_array == i + 1) // 3 / (sh[0] * sh[1]) * 1000
            sizes.append(size)
        the_biggest = np.argmax(sizes)
        return sizes, the_biggest

    @classmethod
    def with_preloaded_model(
        cls,
        size_threshold: float,
        model_config: ModelConfig,
        transforms_config: TransformsConfig = None
    ):
        model = Model(model_config)
        weights = torch.load(os.path.join(model_config.save_path, "best_model.pt"))
        model.load_state_dict(weights)
        return Predictor(model, size_threshold, transforms_config)
