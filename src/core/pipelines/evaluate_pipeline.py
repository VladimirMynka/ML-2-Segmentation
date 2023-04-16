import os
from logging import Logger

import cv2
import numpy as np
from tqdm import tqdm

from src.config_and_utils.config import EvaluatePipelineConfig
from src.config_and_utils.utils import get_last_dataset
from src.core.pipelines.predictor import Predictor


class EvaluatePipeline:
    def __init__(self, logger: Logger, config: EvaluatePipelineConfig = None):
        self.logger = logger
        if config is not None:
            self.config = config
        else:
            self.config = EvaluatePipelineConfig()

    def run(self):
        self.logger.info("Loads model...")
        predictor = Predictor.with_preloaded_model(
            self.config.clod_size_threshold,
            self.config.model_config,
            self.config.transforms_config
        )

        self.logger.info("Loads data...")
        valid_path_x, valid_path_y = self._get_folder()
        size_predicts, count_predicts, size_trues, count_trues = self._get_predicts_and_trues(
            predictor,
            valid_path_x,
            valid_path_y
        )
        size_mse = self.calculate_mse(size_trues, size_predicts)
        count_mse = self.calculate_mse(count_trues, count_predicts)

        self.logger.info(f"Count MSE: {count_mse}; Size MSE: {size_mse}")
        self.logger.info(f"Evaluate pipeline ended.")

    @staticmethod
    def _get_predicts_and_trues(
        predictor: Predictor,
        valid_path_x: os.PathLike | str,
        valid_path_y: os.PathLike | str
    ):
        names = os.listdir(valid_path_x)
        size_preds, count_preds, size_trues, count_trues = [], [], [], []
        for name in tqdm(names):
            path_x = os.path.join(valid_path_x, name)
            path_y = os.path.join(valid_path_y, name)

            image = cv2.imread(path_x)[:, :, ::-1]
            mask = cv2.imread(path_y)[:, :, ::-1]

            predict_count, predict_max_size = predictor.predict_count_and_the_biggest_size(image)
            true_count, true_max_size = predictor.calculate_count_and_the_biggest_size_by_mask(mask, resize=True)

            size_preds.append(predict_max_size)
            size_trues.append(true_max_size)
            count_preds.append(predict_count)
            count_trues.append(true_count)

        return size_preds, count_preds, size_trues, count_trues

    @staticmethod
    def calculate_mse(trues, predicts):
        trues = np.array(trues)
        predicts = np.array(predicts)
        return ((trues - predicts) ** 2).mean()

    def _get_folder(self):
        if self.config.dataset_path is None:
            dataset_path = get_last_dataset(self.config.data_preparation_config.dataset_path)
        else:
            dataset_path = self.config.dataset_path

        valid_path_x = os.path.join(dataset_path, "val", "x")
        valid_path_y = os.path.join(dataset_path, "val", "y")

        return valid_path_x, valid_path_y
