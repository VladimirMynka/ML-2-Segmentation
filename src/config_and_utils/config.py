import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

root = Path(__file__).parent.parent.parent


@dataclass
class LoggerConfig:
    log_file: os.PathLike | str = root / 'data' / 'log_file.log'
    encoding: str = 'utf-8'
    level: str = 'INFO'
    format: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    date_format: str = "%d/%b/%Y %H:%M:%S"


@dataclass
class DataPreparationConfig:
    dataset_path: Path = root / 'data' / 'dataset'
    path: Path = dataset_path / 'markup'
    movie_name: str = "movie.avi"

    split_k: float = 0.6  # percent of train data


@dataclass
class ModelConfig:
    weights: str = "DEFAULT"
    backbone_model: str = "mbv3"
    n_classes: int = 2

    save_path: Path = root / 'data' / 'model'


@dataclass
class TransformsConfig:
    mean: t.Tuple[float] = (0.4611, 0.4359, 0.3905)
    std: t.Tuple[float] = (0.2193, 0.2150, 0.2109)
    image_size: t.Tuple[int, int] = (384, 384)
    max_rotate: int = 15
    max_translate: int = 30
    do_vertical_flip: bool = True
    do_horizontal_flip: bool = True


@dataclass
class TrainPipelineConfig:
    data_preparation_config: DataPreparationConfig = DataPreparationConfig()
    model_config: ModelConfig = ModelConfig()
    transforms_config: TransformsConfig = TransformsConfig()

    dataset_path: os.PathLike | str = None
    batch_size: int = 2
    n_epochs: int = 20
    device: str = 'cpu'

    metric_name: str = 'iou'  # 'iou' | 'dice'


class EvaluatePipelineConfig:
    data_preparation_config: DataPreparationConfig = DataPreparationConfig()
    model_config: ModelConfig = ModelConfig()
    transforms_config: TransformsConfig = TransformsConfig()

    dataset_path: os.PathLike | str = None
    device: str = 'cpu'

    clod_size_threshold: float = 0.5
