import datetime
import json
import os

import cv2
import numpy as np
from fire import Fire
from tqdm import tqdm

from src.config_and_utils.config import DataPreparationConfig


class DatasetPreparationPipeline:
    MAX_ITERATIONS_TO_BREAK = 100000
    """
    Use run method to run pipeline
    """

    def __init__(self):
        self.shape = (0, 0, 3)
        self.max_index = 0
        self.frame_id_to_frame: dict[int, int] = {}
        self.validation_id_to_frame_id: dict[int, int] = {}
        self.indexes: list[int] = []
        self.config = DataPreparationConfig()

    def run(self, movie_name: str = None):
        """
        Make data preparation pipeline:
            - load json with segmentation after validation
            - get frames indexes from their names. Name of each image must be in format
                {some text}_{frame index}.{type}
            - generate mask by polygon segmentations
            - extract frames with needed indexes from video
            - create folders "dataset/{current_datetime}/train/" and "dataset/{current_datetime}/val/"
            - for each of these folders create "~/x/" and "~/y/" directories with images which names are equal
        :param movie_name: name of movie from that will be extracted frames
        :return:
        """
        if movie_name is None:
            movie_name = self.config.movie_name
        movie_path = self.config.path / movie_name
        with open(self.config.path / 'result.json') as validation_json:
            info = json.load(validation_json)

        self.indexes = self.get_indexes(info)
        self.max_index = max(self.indexes)

        video = cv2.VideoCapture(str(movie_path))
        self.frame_id_to_frame = self.extract_frames(video)
        self.shape = list(self.frame_id_to_frame.values())[0].shape

        self.validation_id_to_frame_id = self._get_frame_mappings(info)

        save_path = self.create_directory_structure()

        self.generate_masks(info, save_path)

    @staticmethod
    def get_indexes(info):
        return [int(elem["file_name"].split("_")[-1].split(".")[0]) for elem in info["images"]]

    def extract_frames(self, video):
        frames = {}
        with tqdm() as pbar:
            for i in range(self.MAX_ITERATIONS_TO_BREAK):
                if not video.isOpened():
                    break
                one, frame = video.read()
                if not one:
                    break
                if i not in self.indexes:
                    continue
                frames[i] = frame
                pbar.update()
        return frames

    def _get_frame_mappings(self, info):
        validation_id_to_frame_id = {info["images"][i]["id"]: self.indexes[i] for i in range(len(self.indexes))}
        return validation_id_to_frame_id

    def create_directory_structure(self):
        date = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        save_path = self.config.dataset_path / date
        os.makedirs(save_path / 'train' / 'x')
        os.makedirs(save_path / 'train' / 'y')
        os.makedirs(save_path / 'val' / 'x')
        os.makedirs(save_path / 'val' / 'y')
        return save_path

    def generate_masks(self, info, save_path):
        img = self.get_new_black_image()
        last_index = 0

        for row in tqdm(info['annotations']):
            if row['image_id'] != last_index:
                self.save_images(img, last_index, save_path)

                last_index = row['image_id']
                img = self.get_new_black_image()

            cv2.fillPoly(img, np.int32([row['segmentation']]).reshape(1, -1, 2), (255, 255, 255))

        self.save_images(img, last_index, save_path)

    def get_new_black_image(self):
        return np.zeros(self.shape).astype(np.uint8)

    def save_images(self, img, last_index, save_path):
        source_image_id = self.validation_id_to_frame_id[last_index]
        source_image = self.frame_id_to_frame[source_image_id]

        folder = 'train' if source_image_id < self.config.split_k * self.max_index else 'val'

        cv2.imwrite(str(save_path / folder / 'y' / f"{source_image_id}_{last_index}.png"), img)
        cv2.imwrite(str(save_path / folder / 'x' / f"{source_image_id}_{last_index}.png"), source_image)


if __name__ == "__main__":
    Fire(DatasetPreparationPipeline)
