import logging
from os import PathLike

from fire import Fire

from src.config_and_utils.config import TrainPipelineConfig, EvaluatePipelineConfig, DemoPipelineConfig
from src.config_and_utils.utils import init_logging
from src.core.pipelines.train_pipeline import TrainPipeline
from src.core.pipelines.evaluate_pipeline import EvaluatePipeline
from src.dataset_preparation.dataset_preparation_pipeline import DatasetPreparationPipeline
from src.demo.demo_pipeline import DemoPipeline


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

    def evaluate(self, dataset: PathLike | str = None):
        """
        Receives the dataset folder. Loads the model from ./data/model/. Evaluates the model with the provided dataset,
        prints the results and saves it to the log
        :param dataset: path to evaluating dataset contains folder with images and json description
        """
        config = EvaluatePipelineConfig()
        if dataset is not None:
            config.dataset_path = dataset
        evaluater = EvaluatePipeline(self.logger, config)
        evaluater.run()

    def demo(self, video: PathLike | str = None):
        """
        Runs real-time demo with provided image or video file
        :param video: path to video that must be processed
        """
        config = DemoPipelineConfig()
        if video is not None:
            config.movie_path = video
        demo = DemoPipeline(self.logger, config)
        demo.run()

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
