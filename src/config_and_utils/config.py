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
class EvaluatePipelineConfig:
    data_preparation_config: DataPreparationConfig = DataPreparationConfig()
    model_config: ModelConfig = ModelConfig()
    transforms_config: TransformsConfig = TransformsConfig()

    dataset_path: os.PathLike | str = None
    device: str = 'cpu'

    clod_size_threshold: float = 0.5


@dataclass
class TrainPipelineConfig:
    data_preparation_config: DataPreparationConfig = DataPreparationConfig()
    model_config: ModelConfig = ModelConfig()
    transforms_config: TransformsConfig = TransformsConfig()
    evaluate_config: EvaluatePipelineConfig = EvaluatePipelineConfig()

    dataset_path: os.PathLike | str = None
    batch_size: int = 2
    n_epochs: int = 20
    device: str = 'cpu'
    do_evaluate: bool = True

    metric_name: str = 'iou'  # 'iou' | 'dice'


@dataclass
class ColorConfig:
    bbox_color: t.Tuple[int, int, int] = (255, 0, 0)
    text_color: t.Tuple[int, int, int] = (255, 255, 0)


@dataclass
class DemoPipelineConfig:
    model_config: ModelConfig = ModelConfig()
    transforms_config: TransformsConfig = TransformsConfig()

    movie_path: str | os.PathLike = DataPreparationConfig.dataset_path / "movie.avi"
    clod_size_threshold: float = 0.5
    skip_frames: int = 3
    size_type: str = 'model'  # 'model' | 'source'
    source_size: t.Tuple[int, int] = (1280, 1028)
    draw_mask: bool = False
    save_video: bool | str = DataPreparationConfig.dataset_path / "processed.mp4"  # False or path to file

    biggest_color_config: ColorConfig = ColorConfig(bbox_color=(0, 0, 255), text_color=(0, 255, 255))
    usual_color_config: ColorConfig = ColorConfig(bbox_color=(255, 0, 0), text_color=(255, 0, 0))
    biggest_text_bold: int = 2
    usual_text_bold: int = 1
