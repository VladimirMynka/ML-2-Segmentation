import os
from pathlib import Path
from dataclasses import dataclass

root = Path(__file__).parent.parent.parent


@dataclass
class DataPreparationConfig:
    dataset_path: Path = root / 'data' / 'dataset'
    path: Path = dataset_path / 'markup'
    movie_name: str = "movie.avi"

    split_k: float = 0.6  # percent of train data

