import logging
from os import PathLike

from fire import Fire

from src.config_and_utils.config import TrainPipelineConfig
from src.config_and_utils.utils import init_logging
from src.core.train_pipeline import TrainPipeline
from src.dataset_preparation.dataset_preparation_pipeline import DatasetPreparationPipeline


class CLI:
    def __init__(self):
        init_logging()
        self.logger = logging.getLogger(__name__)

    def train(self, dataset: PathLike | str = None) -> None:
        """
        Receives the dataset folder. Performs model training. Saves the artifacts to ./model/. Logs the results.
        :param dataset: path to train folder contains folder with images and json description
        """
        config = TrainPipelineConfig()
        if dataset is not None:
            config.dataset_path = dataset
        trainer = TrainPipeline(self.logger, config)
        trainer.run()

    def evaluate(self, dataset: PathLike):
        """
        Receives the dataset folder. Loads the model from ./data/model/. Evaluates the model with the provided dataset,
        prints the results and saves it to the log
        :param dataset: path to evaluating dataset contains folder with images and json description
        """
        pass

    def demo(self, video: PathLike):
        """
        Runs real-time demo with provided image or video file
        :return:
        """
        pass

    @staticmethod
    def prepare_dataset(movie_name: str = None):
        """
        Prepare dataset by validated data
        :param movie_name:
        :return:
        """
        preparer = DatasetPreparationPipeline()
        preparer.run(movie_name)


if __name__ == "__main__":
    Fire(CLI)
