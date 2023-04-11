from os import PathLike
from fire import Fire


class CLI:
    def train(self, dataset: PathLike) -> None:
        """
        Receives the dataset folder. Performs model training. Saves the artifacts to ./model/. Logs the results.
        :param dataset: path to train folder contains folder with images and json description
        """
        pass

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


if __name__ == "__main__":
    Fire(CLI)
